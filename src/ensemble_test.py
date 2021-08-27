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
import time

flags.DEFINE_list('wds', ['clean'], 'type the desired weak defence. type the name multiple times for multiple '
                                         'instances of WD')
flags.DEFINE_integer('gpu', None, 'set which gpu to use')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('input', '0', 'path to input image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_float('sensitivity', 0.5, 'controls the sensitivity of majority voting')

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def get_weighted_box(boxes):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box (label, score, x1, y1, x2, y2)
    """

    box = np.zeros(6, dtype=np.float32)
    conf = 0
    conf_list = []
    label_list = []
    w = 0
    for b in boxes:
        box[2:] += (b[1] * b[2:])
        conf += b[1]
        conf_list.append(b[1])
        label_list.append(b[0])
    # get the most common class for this box
    box[0] = np.bincount(label_list).argmax()
    # get the average confidence
    box[1] = conf / len(boxes)
    box[2:] /= conf  # divide by the sum of weights to get weighted average of bbox coordinates
    print(box)
    return box

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if (physical_devices != []) and (FLAGS.gpu is not None):
        tf.config.experimental.set_visible_devices(physical_devices[FLAGS.gpu], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[FLAGS.gpu], True)
    else:
        tf.config.set_visible_devices([], 'GPU')

    #img_raw = tf.image.decode_image(
    #    open(FLAGS.image, 'rb').read(), channels=3)
    #img = tf.expand_dims(img_raw, 0)
    #img = transform_images(img, FLAGS.size)
    #img = cv2.imread(FLAGS.image)
    #img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img_in = tf.expand_dims(img_in, 0)
    #img_in = transform_images(img_in, FLAGS.size)


    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    models = []
    for wd in FLAGS.wds:
        wd_model = YoloV3(classes=FLAGS.num_classes)
        weights = f'./checkpoints/yolov3_{wd}/yolov3_{wd}.tf'
        wd_model.load_weights(weights).expect_partial()
        models.append(WeakDefence(wd_model, wd, FLAGS.size))
    logging.info('ensemble loaded')

    try:
        vid = cv2.VideoCapture(int(FLAGS.input))
    except:
        vid = cv2.VideoCapture(FLAGS.input)

    while True:
        _, img = vid.read()
        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        time1 = time.time()
        boxes = []
        scores = []
        classes = []
        for model in models:
            boxes_temp, scores_temp, classes_temp, _ = model.predict(tf.identity(img_in))
            boxes = np.concatenate((boxes, boxes_temp), axis=1) if np.size(boxes) else boxes_temp
            scores = np.concatenate((scores, scores_temp), axis=1) if np.size(scores) else scores_temp
            classes = np.concatenate((classes, classes_temp), axis=1) if np.size(classes) else [79 - x for x in classes_temp]

        boxes = np.squeeze(boxes, axis=0)
        scores = np.squeeze(scores, axis=0)
        classes = np.squeeze(classes, axis=0)

        box_clusters = []
        for i in range(len(boxes) - 1):
            cluster = []
            for j in range(i+1, len(boxes)):
                if bb_intersection_over_union(boxes[i], boxes[j]) > FLAGS.sensitivity:
                    temp_box = [classes[j], scores[j]].extend(boxes[j])
                    cluster.append(temp_box)
                    np.delete(classes, j)
                    np.delete(scores, j)
                    np.delete(boxes, j)
                    j -= 1
            cluster.append([classes[i], scores[i]].extend(boxes[i]))
            box_clusters.append(cluster)

        boxes, scores, classes, valid_detections = [], [], [], 0
        for cluster in box_clusters:
            prediction = get_weighted_box(cluster)
            classes.append(prediction[0])
            scores.append(prediction[1])
            boxes.append(prediction[2:])
            valid_detections += 1

        boxes = tf.expand_dims(boxes, axis=0)
        scores = tf.expand_dims(scores, axis=0)
        classes = tf.expand_dims(classes, axis=0)
        valid_detections = tf.expand_dims(valid_detections, axis=0)

        #results = majority_voting((boxes, scores, classes, valid_detections), FLAGS.size, FLAGS.sensitivity)
        results = (boxes, scores, classes, valid_detections)
        time2 = time.time()
        fps = 1 / (time2 - time1)
        output = draw_outputs(copy.copy(img/255), results, class_names)
        output = cv2.putText(output, f'FPS: {fps}', (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.imshow('ensemble', output)
        if cv2.waitKey(1) == ord('q'):
            break
        #cv2.imwrite('output.jpg', img)






if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass






