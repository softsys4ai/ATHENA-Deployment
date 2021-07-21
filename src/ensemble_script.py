import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs

import numpy as np #my thing to flip image
from multiprocessing import Process, Array, shared_memory
import random

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', '0',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('rotate', 0, 'degrees to rotate image')
flags.DEFINE_integer('wds', 1, 'number of WDs')
flags.DEFINE_integer('gpu', None, 'set which gpu to use')

def main(_argv):
    _argv = _argv[0]
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

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    frames_buffer = shared_memory.SharedMemory(name='videoFrames')
    while True:
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

        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue
        else:
            frames = np.ndarray(img.shape, dtype=np.array(img).dtype, buffer=frames_buffer.buf)
            vid.release()
            break

    while True:
        img = frames.copy()
        if img is None:
            continue

        def rotate_image(image, angle):
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
            return result
        img = rotate_image(img, FLAGS.rotate) if FLAGS.rotate != 0 else img

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        #print(tf.executing_eagerly)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]

        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (random.randint(0,255), random.randint(0,255), random.randint(0,255)), 2)
        if FLAGS.output:
            out.write(img)
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()



if __name__ == '__main__':
    try:
        while True:
            vid = cv2.VideoCapture(0)
            _, frame = vid.read()
            if frame is None:
                continue
            frames_buffer = shared_memory.SharedMemory(name='videoFrames', create=True, size=frame.nbytes)
            output = np.ndarray(frame.shape, dtype=frame.dtype, buffer=frames_buffer.buf)
            output[:] = np.array(frame)[:]
            vid.release()
            break
        models = []
        for i in range(2):
            models.append(Process(target=app.run, args=(main,)))
            models[-1].daemon = True
            models[-1].start()
            print('reeeeeeee', i)

        time.sleep(30)
        print("starting")
        while True:
            time.sleep(0.3)
            vid = cv2.VideoCapture(0)
            _, frame = vid.read()
            if not frame is None:
                output[:] = np.array(frame)[:]

    except SystemExit:
        pass





#python ensemble_script.py --video 0 TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
 # - defaults/win-64::tensorflow==2.3.0=mkl_py38h8c0d9a2_0
 # - defaults/win-64::tensorflow-base==2.3.0=eigen_py38h75a453f_0
 # - defaults/noarch::tensorflow-estimator==2.5.0=pyh7b7c402_0
#python ensemble_script.py --video 0 --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit
#TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" ensemble_script.py
#python ensemble_script.py TF_XLA_FLAGS=--tf_xla_cpu_global_jit