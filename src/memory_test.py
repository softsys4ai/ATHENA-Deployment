import sys
from yolov3_tf2.weak_defences import WeakDefence
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from memory_profiler import *


flags.DEFINE_string('weights', './checkpoints/yolov3_trash/yolov3_trash_dark_1.tf',
                    'path to weights file')
flags.DEFINE_integer('amt', 1, 'number of models to load')
flags.DEFINE_string('wd', 'clean', 'weak defence to use')
flags.DEFINE_integer('size', 416, 'resolution of models')

@profile
def main(_argv):
    yolo = YoloV3(classes=80)
    yolo.load_weights(FLAGS.weights)
    wrapped_yolo = WeakDefence(yolo, FLAGS.wd, FLAGS.size)

    yolo2 = YoloV3(classes=80)
    yolo2.load_weights(FLAGS.weights)
    wrapped_yolo2 = WeakDefence(yolo2, FLAGS.wd, FLAGS.size)

    yolo3 = YoloV3(classes=80)
    yolo3.load_weights(FLAGS.weights)
    wrapped_yolo3 = WeakDefence(yolo3, FLAGS.wd, FLAGS.size)

    yolo4 = YoloV3(classes=80)
    yolo4.load_weights(FLAGS.weights)
    wrapped_yolo4 = WeakDefence(yolo4, FLAGS.wd, FLAGS.size)

    yolo5 = YoloV3(classes=80)
    yolo5.load_weights(FLAGS.weights)
    wrapped_yolo5 = WeakDefence(yolo5, FLAGS.wd, FLAGS.size)

    yolo6 = YoloV3(classes=80)
    yolo6.load_weights(FLAGS.weights)
    wrapped_yolo6 = WeakDefence(yolo6, FLAGS.wd, FLAGS.size)

    list = [wrapped_yolo, wrapped_yolo2, wrapped_yolo3, wrapped_yolo4, wrapped_yolo5, wrapped_yolo6]



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass