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

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('dataset', './data/coco2017_train.tfrecord', 'path to dataset')


#@tf.function
def main(_argv):
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



if __name__ == '__main__':
    app.run(main)
