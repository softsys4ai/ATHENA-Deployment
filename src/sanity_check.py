import time

from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.weak_defences import WeakDefence
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
import numpy as np #my thing to flip image
from memory_profiler import profile

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', '0',
                    'path to video file or number for webcam)')
#flags.DEFINE_string('tfrecord', 'clean_test.tfrecord', 'tfrecord instead of video')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('rotate', 0, 'degrees to rotate image')
flags.DEFINE_integer('gpu', None, 'set which gpu to use')
flags.DEFINE_integer('test_duration', 30, 'set the amount of time to test each cenario')


#goals normal speed, wd speed, wd works
#vid.read (0,255), tfrecord (0,255)



def main(_argv):
    #set gpu use
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if (physical_devices != []) and (FLAGS.gpu is not None):
        tf.config.experimental.set_visible_devices(physical_devices[FLAGS.gpu], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[FLAGS.gpu], True)
    else:
        tf.config.set_visible_devices([], 'GPU')

    #load classes for data
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')


    #------------Video----------------------------------------------------------------------

    #get imgs
    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    #test normal model
    yolo = YoloV3(classes=FLAGS.num_classes)
    yolo.load_weights(FLAGS.weights)

    # without tf func
    times = []
    global_time = time.time()
    while True:
        _, img = vid.read()
        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        #preprocess image
        img = tf.expand_dims(img, 0)
        img = transform_images(img, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img)
        t2 = time.time()
        times.append(t2 - t1)
        times = times[-20:]
        if time.time() - global_time > FLAGS.test_duration:
            total = sum(times)/len(times)*1000
            print(f'Time: {total}ms')
            break

    # with tf func
    @tf.function  # (input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),))
    def serve(x):
        print('tracing')
        return yolo(x, training=False)

    times = []
    global_time = time.time()
    while True:
        _, img = vid.read()
        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        #preprocess image
        img = tf.expand_dims(img, 0)
        img = transform_images(img, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = serve(img)
        t2 = time.time()
        times.append(t2 - t1)
        times = times[-20:]
        if time.time() - global_time > FLAGS.test_duration:
            total = sum(times)/len(times)*1000
            print(f'Time: {total}ms')
            break

    # test WD
    wrapped_yolo = WeakDefence(yolo, 'clean', FLAGS.size)
    # without tf func
    times = []
    global_time = time.time()
    while True:
        _, img = vid.read()
        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        # preprocess image
        img = tf.expand_dims(img, 0)
        img = transform_images(img, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = wrapped_yolo.predict_old(img)
        t2 = time.time()
        times.append(t2 - t1)
        times = times[-20:]
        if time.time() - global_time > FLAGS.test_duration:
            total = sum(times) / len(times) * 1000
            print(f'Time: {total}ms')
            break

    # with tf func
    @tf.function  # (input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),))
    def serve_test(x):
        print('tracing')
        return wrapped_yolo.predict(x, training=False)

    times = []
    global_time = time.time()
    while True:
        _, img = vid.read()
        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        # preprocess image
        img = tf.expand_dims(img, 0)
        img = transform_images(img, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = serve(img)
        t2 = time.time()
        times.append(t2 - t1)
        times = times[-20:]
        if time.time() - global_time > FLAGS.test_duration:
            total = sum(times) / len(times) * 1000
            print(f'Time: {total}ms')
            break

    #with interneal tf function
    times = []
    global_time = time.time()
    while True:
        _, img = vid.read()
        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        # preprocess image
        img = tf.expand_dims(img, 0)
        img = transform_images(img, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = wrapped_yolo.predict_test(img)
        t2 = time.time()
        times.append(t2 - t1)
        times = times[-20:]
        if time.time() - global_time > FLAGS.test_duration:
            total = sum(times) / len(times) * 1000
            print(f'Time: {total}ms')
            break


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
