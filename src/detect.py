import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
import skimage
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs, draw_outputs_bbox_deltas, majority_voting

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3_clean/yolov3_clean.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/meme.jpg', 'path to input image')
flags.DEFINE_string('tfrecord', './data/coco2017_train.tfrecord', 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('gpu', None, 'set which gpu to use')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if (physical_devices != []) and (FLAGS.gpu is not None):
        tf.config.experimental.set_visible_devices(physical_devices[FLAGS.gpu], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[FLAGS.gpu], True)
    else:
        tf.config.set_visible_devices([], 'GPU')

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
        print('yuh', img_raw)
    else:
        img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)

    print(img_raw/255)
    img1 = tf.expand_dims(img_raw, 0)
    #print(img1 / 255)
    img1 = transform_images(img1, FLAGS.size)
    img2_raw = skimage.util.random_noise(img_raw.numpy()/255, mode='gaussian')
    img2 = tf.expand_dims(img2_raw, 0)
    img2 = tf.image.resize(img2, (FLAGS.size, FLAGS.size))
    print(img2)


    t1 = time.time()
    boxes, scores, classes, nums = yolo(img1)
    boxes2, scores2, classes2, nums2 = yolo(img2)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))


    #majority_voting((boxes, scores, classes, nums), class_names, FLAGS.size, 10)
    img1 = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img1 = draw_outputs(img1, (boxes, scores, classes, nums), class_names)
    img2 = cv2.cvtColor((img2_raw * 255).astype('float32'), cv2.COLOR_RGB2BGR)
    img2 = draw_outputs(img2, (boxes2, scores2, classes2, nums2), class_names)
    #img2 = draw_outputs_bbox_deltas(img2, (boxes, scores, classes, nums), class_names)
    img_all = np.concatenate((img1, img2), axis=1)
    cv2.imwrite(FLAGS.output, img_all)
    logging.info('output saved to: {}'.format(FLAGS.output))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
