import time

import skimage
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, majority_voting

import numpy as np #my thing to flip image
from yolov3_tf2.weak_defences import WeakDefence
import copy

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3_trash/yolov3_trash_1.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', '0', #'./data/Office-Parkour.mp4'
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('rotate', 0, 'degrees to rotate image')
flags.DEFINE_integer('gpu', None, 'set which gpu to use')

def main(_argv):
    tf.config.run_functions_eagerly(False)
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
        yolo2 = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights('./checkpoints/yolov3/yolov3.tf')
    yolo2.load_weights(FLAGS.weights)

    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    wrapped_yolo = WeakDefence(yolo, 'clean', FLAGS.size) #TODO: make shure that each WD does not contaminate the other. some operations coppy while others dont.
    wrapped_yolo2 = WeakDefence(yolo2, 'clean', FLAGS.size)


    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        _, img = vid.read()
        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        def rotate_image(image, angle):
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
            return result

        img = skimage.util.img_as_float32(img)
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        t1 = time.time()
        boxes, scores, classes, nums = wrapped_yolo.predict(copy.copy(img_in))
        boxes2, scores2, classes2, nums2 = wrapped_yolo2.predict(copy.copy(img_in))
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]


        output1 = draw_outputs(wrapped_yolo.get_image(copy.copy(img)), (boxes, scores, classes, nums), class_names)
        output2 = draw_outputs(wrapped_yolo2.get_image(copy.copy(img)), (boxes2, scores2, classes2, nums2), class_names)
        img_all = np.concatenate((output1, output2), axis=1)
        img_all = cv2.putText(img_all, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        if FLAGS.output:
            out.write(img)
        cv2.imshow('output', img_all)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
