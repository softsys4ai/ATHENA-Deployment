from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
from tensorflow import keras

import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset
#from yolov3_tf2 import dataset as dataset #this could be a bug
import sys


flags.DEFINE_string('dataset', './data/coco2017_train.tfrecord', 'path to dataset')
flags.DEFINE_string('val_dataset', './data/coco2017_train.tfrecord', 'path to validation dataset')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', './checkpoints/yolov3/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov3_trash/yolov3_trash_dark_3.tf',
                    'path to save')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'none',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune', 'yes'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only'
                  'yes: resume training')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 50, 'number of epochs')
flags.DEFINE_integer('batch_size', 16, 'batch size')
flags.DEFINE_float('learning_rate', 1e-5, 'learning rate')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')
flags.DEFINE_integer('gpu', None, 'set which gpu to use')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if (physical_devices != []) and (FLAGS.gpu is not None):
        tf.config.experimental.set_visible_devices(physical_devices[FLAGS.gpu], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[FLAGS.gpu], True)
    else:
        tf.config.set_visible_devices([], 'GPU')

    if FLAGS.tiny:
        model = YoloV3Tiny(FLAGS.size, training=True,
                           classes=FLAGS.num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    if FLAGS.dataset:
        train_dataset = dataset.load_tfrecord_dataset(
            FLAGS.dataset, FLAGS.classes, FLAGS.size)
    else:
        train_dataset = dataset.load_fake_dataset()

    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))

    logging.info("tensor? " + str(tf.executing_eagerly()))

    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    if FLAGS.val_dataset:
        val_dataset = dataset.load_tfrecord_dataset(
            FLAGS.val_dataset, FLAGS.classes, FLAGS.size)
    else:
        val_dataset = dataset.load_fake_dataset()
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
    print(val_dataset)

    # Configure the model for transfer learning
    if FLAGS.transfer == 'none':
        pass  # Nothing to do
    elif FLAGS.transfer in ['darknet', 'no_output']:
        # Darknet transfer is a special case that works
        # with incompatible number of classes

        # reset top layers
        if FLAGS.tiny:
            model_pretrained = YoloV3Tiny(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        else:
            model_pretrained = YoloV3(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        model_pretrained.load_weights(FLAGS.weights)

        if FLAGS.transfer == 'darknet':
            model.get_layer('yolo_darknet').set_weights(
                model_pretrained.get_layer('yolo_darknet').get_weights())
            freeze_all(model.get_layer('yolo_darknet'))

        elif FLAGS.transfer == 'no_output':
            for l in model.layers:
                if not l.name.startswith('yolo_output'):
                    l.set_weights(model_pretrained.get_layer(
                        l.name).get_weights())
                    freeze_all(l)
    elif FLAGS.transfer == 'yes':
        model.load_weights(FLAGS.weights)
    else:
        # All other transfer require matching classes
        model.load_weights(FLAGS.weights)
        if FLAGS.transfer == 'fine_tune':
            # freeze darknet and fine tune other layers
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        elif FLAGS.transfer == 'frozen':
            # freeze everything
            freeze_all(model)

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    #optimizer = tf.keras.optimizers.SGD(momentum=0.9127, learning_rate=0.00001)
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
            for mask in anchor_masks]

    if FLAGS.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        for epoch in range(1, FLAGS.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                logging.info("{}_train_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_loss.update_state(total_loss)

            for batch, (images, labels) in enumerate(val_dataset):
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                logging.info("{}_val_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_val_loss.update_state(total_loss)

            logging.info("{}, train: {}, val: {}".format(
                epoch,
                avg_loss.result().numpy(),
                avg_val_loss.result().numpy()))

            avg_loss.reset_states()
            avg_val_loss.reset_states()
            model.save_weights(FLAGS.output[:-3] + f'_{epoch}.tf')

                #'checkpoints/yolov3_train_{}.tf'.format(epoch))
    else:
        model.compile(optimizer=optimizer, loss=loss,
                      run_eagerly=(FLAGS.mode == 'eager_fit'))
# checkpoints/yolov3_train_{epoch}.tf



        class lr(keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                lr = self.model.optimizer.lr
                tf.print(lr)

        class save(keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                self.model.save(FLAGS.output)
                tf.print("saving")

        class burn_in(keras.callbacks.Callback):

            def __init__(self):
                self.burn = False

            def on_epoch_begin(self, epoch, logs=None):
                if epoch < 3:
                    slope = FLAGS.learning_rate / 2
                    self.model.optimizer.lr = slope * epoch
                    logging.info(self.model.optimizer.lr)
            #def on_epoch_begin(self, epoch, logs=None):
            #    if epoch < 1:
            #        self.burn = True
            #        #self.model.optimizer.lr = 0.1
            #    else:
            #        self.burn = False

            #def on_batch_begin(self, batch, logs=None):
            #    if self.burn:
            #        slope = FLAGS.learning_rate / 4
            #        self.model.optimizer.lr = slope * batch
            #        logging.info(self.model.optimizer.lr)

        train_dataset = train_dataset.take(5)
        val_dataset = train_dataset #val_dataset.take(1)
        while True:
            print(model.evaluate(val_dataset))
        logs = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1)
        callbacks = [
            ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1),
            EarlyStopping(patience=50, verbose=1),
            ModelCheckpoint(FLAGS.output[:-3] + '_{epoch}.tf',
                            verbose=1, save_weights_only=True, save_freq=1),
            logs,
            lr()
        ]
        #TensorBoard(log_dir='logs2')
        history = model.fit(train_dataset,
                            epochs=FLAGS.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

#python train.py --dataset ./data/voc2012_train.tfrecord --val_dataset ./data/voc2012_val.tfrecord --classes ./data/voc2012.names --num_classes 20 --mode fit --transfer darknet --batch_size 16 --epochs 10 --weights ./checkpoints/yolov3/yolov3.tf --weights_num_classes 80 --output ./checkpoints/yolov3_voc2017/yolov3_voc2017.tf

#python train.py --dataset ./data/coco2017_train.tfrecord --val_dataset ./data/coco2017_train.tfrecord --classes ./data/coco.names --num_classes 80 --mode fit --transfer darknet --batch_size 16 --epochs 2 --weights ./checkpoints/yolov3/yolov3.tf --weights_num_classes 80 --output ./checkpoints/yolov3_coco2017_trash/yolov3_coco2017_trash.tf

#((None, 416, 416, 3), ((None, 13, 13, 3, 6), (None, 26, 26, 3, 6), (None, 52, 52, 3, 6)))
#((None, 320, 320, 3), ((None, 10, 10, 3, 6), (None, 20, 20, 3, 6), (None, 40, 40, 3, 6)))
#((None, 320, 320, 3), ((None, 10, 10, 3, 6), (None, 20, 20, 3, 6), (None, 40, 40, 3, 6)))


#monitor="val_loss", factor=0.1, patience=3, verbose=1