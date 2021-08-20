import tensorflow as tf
from yolov3_tf2.dataset import load_tfrecord_dataset
from absl import app, flags, logging
from absl.flags import FLAGS
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
import cv2
import numpy as np
import skimage
import yolov3_tf2.dataset as dataset
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('dataset', './data/coco2017_train.tfrecord', 'path to dataset')
flags.DEFINE_integer('batch_size', 16, 'batch size')


#@tf.function
def nope(_argv):
    x = tf.constant(1)
    tf.print(x)
    y = tf.constant([1, 2])
    tf.print(y)

    z = []
    tf.print(type(z))
    z.append(y)
    tf.print(type(z))
    tf.print(z)
    tf.print(type(z[0]))
    tf.print(z[0])

    dataset = load_tfrecord_dataset(FLAGS.dataset, FLAGS.classes, FLAGS.size)
    data = dataset.take(1)
    tf.print(data)
    tf.print(type(data))
    tf.print(data)
    for img, lable in data:
        img = img.numpy()
        img = img / 255
        tf.print(img)
        #tf.print(img)
        img = np.round(img, 8)
        tf.print(type(img))
        img = skimage.util.random_noise(mode='salt', image=img, amount=0.05)
        #img = img * 255
        #img = np.round(img, 7)
        tf.print(img)
        #cv2.imwrite('output.jpg', img)
        #tf.image.adjust_contrast


    tf.print(tf.executing_eagerly())



def main(_argv):
    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks

    if FLAGS.dataset:
        og_dataset = dataset.load_tfrecord_dataset(
            FLAGS.dataset, FLAGS.classes, FLAGS.size)
    og_dataset = og_dataset.shuffle(buffer_size=512, seed=7)
    og_dataset = og_dataset.batch(FLAGS.batch_size)
    og_dataset = og_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
    #og_dataset = og_dataset.unbatch()
    og_dataset = og_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    for img, lable in og_dataset.take(1):
        #print('start')
        print(img)
        img = img.numpy()
        #print(img, type(img), np.shape(img))
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #print(img, type(img), np.shape(img))
        #print('next')
        #cv2.imshow('out', img)
        #cv2.waitKey(2000)


    def augmentation2(x):
        def map_func(img):
            img = img.numpy()
            img = np.flip(img, axis=1)
            img = np.flip(img, axis=0)
            img = img / 255
            return img
        augmented_imgs = tf.map_fn(lambda img: map_func(img), x)
        augmented_imgs = tf.image.resize(augmented_imgs, (FLAGS.size, FLAGS.size))
        augmented_imgs = tf.image.random_hue(augmented_imgs, 0.25)
        return augmented_imgs

    def augmentation(x):
        def map_func(img):
            img = img.numpy()
            img = img / 255
            img = skimage.util.random_noise(img, mode='salt', clip=True)
            return img

        augmented_imgs = tf.map_fn(lambda img: map_func(img), x)
        augmented_imgs = tf.image.resize(augmented_imgs, (FLAGS.size, FLAGS.size))
        augmented_imgs = tf.image.random_hue(augmented_imgs, 0.15)
        augmented_imgs = tf.image.random_saturation(augmented_imgs, lower=1, upper=10)
        return augmented_imgs

    if FLAGS.dataset:
        raw_dataset = dataset.load_tfrecord_dataset(
            FLAGS.dataset, FLAGS.classes, FLAGS.size)

    raw_dataset = raw_dataset.shuffle(buffer_size=512, seed=7)
    raw_dataset = raw_dataset.batch(FLAGS.batch_size)
    salt_dataset = raw_dataset.map(lambda x, y: (
        tf.py_function(func=augmentation, inp=[x], Tout=tf.float32),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))  # transform dataset

    #salt_dataset = salt_dataset.unbatch()
    salt_dataset = salt_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    for img, lable in salt_dataset.take(1):
        print(img)
        img = img.numpy()
        img = img[0]
        #img[:10][:10] = [1.5, 1.5, 1.5]
        #img = np.squeeze(img, axis=1)
        print(img)
        #print(img, type(img), np.shape(img))
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #print(img, type(img), np.shape(img))
        #print('next')
        #cv2.imshow('out', img)
        #cv2.waitKey(5000)

    og_imgs = []
    salt_imgs = []
    for x, _ in og_dataset.take(1):
        og_imgs = x
        og_imgs = og_imgs.numpy()
    for x, _ in salt_dataset.take(1):
        salt_imgs = x
        salt_imgs = salt_imgs.numpy()

    for i in range(5):
        img1 = og_imgs[i]
        img2 = salt_imgs[i]
        img_all = np.concatenate((img1, img2), axis=1)
        cv2.imshow('output', img_all)
        cv2.waitKey(100)

    og_imgs = []
    salt_imgs = []
    temp = []
    for x, _ in og_dataset.take(1):
        og_imgs = x
        temp = _
        og_imgs = og_imgs.numpy()
    for x, _ in salt_dataset.take(1):
        salt_imgs = x
        salt_imgs = salt_imgs.numpy()
    count = 0
    for img, lable in og_imgs, temp:
        count += 1

    print(count)


    for i in range(5):
        img1 = og_imgs[i]
        img2 = salt_imgs[i]
        img_all = np.concatenate((img1, img2), axis=1)
        cv2.imshow('output', img_all)
        cv2.waitKey(100)


if __name__ == '__main__':
    app.run(main)
