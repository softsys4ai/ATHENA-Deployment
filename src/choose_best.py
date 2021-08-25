import copy
import cv2
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
from yolov3_tf2.dataset import load_tfrecord_dataset, transform_images
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.utils import draw_outputs
import time
from yolov3_tf2.weak_defences import WeakDefence

flags.DEFINE_string('weights', './checkpoints/yolov3_clean/yolov3_clean.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('dataset', './data/coco2017_val.tfrecord', 'path to dataset')
flags.DEFINE_string('output', '/nfs/general/lane/best_model.txt', 'path to output image')
flags.DEFINE_float('accuracy', None, 'what is the minimum iou')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('gpu', None, 'set which gpu to use')
flags.DEFINE_boolean('show_img', False, 'controls weather or not images are shown')
flags.DEFINE_string('wd', 'clean', 'controls the wd to be used')


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
    wrapped_yolo = WeakDefence(yolo, FLAGS.wd, FLAGS.size)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    tp = 0
    fp = 0
    average_precision = 0
    average_recall = 0
    x_coordinates = []
    y_coordinates = []
    coordinates = []
    mAP = 0
    latency = 0
    average_counter = 0
    #total_mAP = 0
    #total_precission = 0
    #total_recall = 0

    dataset = load_tfrecord_dataset(FLAGS.dataset, FLAGS.classes, FLAGS.size)
    #dataset = dataset.shuffle(512, seed=0)  # 7 is e+01
    time1 = time.time()
    for accuracy in range(10):
        for image, labels in dataset.take(100):
            boxes = []
            scores = []
            classes = []
            for x1, y1, x2, y2, label in labels:
                if x1 == 0 and x2 == 0:
                    continue

                boxes.append((x1, y1, x2, y2))
                scores.append(1)
                classes.append(label)
            nums = [len(boxes)]
            boxes = [boxes]
            scores = [scores]
            classes = [classes]

            #get ground truth
            gt_boxes, gt_scores, gt_classes, gt_nums = boxes[0], scores[0], classes[0], nums[0]

            #get prediction
            # boxes2, scores2, classes2, nums2 = boxes[0], scores[0], classes[0], nums[0]
            img = tf.expand_dims(image, 0)
            img = transform_images(img, FLAGS.size)
            boxes2, scores2, classes2, nums2 = wrapped_yolo.predict(img)
            pred_boxes, pred_scores, pred_classes, pred_nums = boxes2[0], scores2[0], classes2[0], nums2[0]


            # calculate the number of miss classified objects
            #fp += nums2 - nums

            for i in range(pred_nums):
                for j in range(gt_nums):
                    gt_cls = tf.cast(gt_classes[j], dtype=tf.int64)
                    pred_cls = tf.cast(pred_classes[i], dtype=tf.int64)
                    if tf.math.not_equal(gt_cls, pred_cls): #tf.math.not_equal(classes[j], classes2[i])
                        pass
                    iou = bb_intersection_over_union(gt_boxes[j], pred_boxes[i])
                    if iou >= (0.5 + (0.05 * accuracy)):
                        tp += 1
                        fp -= 1
                        break
                fp += 1

            precision = tp / (tp + fp) if (fp + tp) != 0 else 0
            recall = tp / gt_nums if gt_nums != 0 else 0
            #print('TP = ', tp, 'FP = ', fp)
            #print('precision = ', precision, 'recall = ', recall)

            if FLAGS.show_img:
                image = cv2.cvtColor(image.numpy() / 255, cv2.COLOR_RGB2BGR)
                img1 = draw_outputs(copy.copy(image), (boxes, scores, classes, nums), class_names)
                img2 = draw_outputs(copy.copy(image), (boxes2, scores2, classes2, nums2), class_names)
                img_all = np.concatenate((img1, img2), axis=1)
                img_all = cv2.putText(img_all, f'TP = {tp} FP = {fp}\nprecision = {precision} recall = {recall}', (0, 30),
                                      cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

                while True:
                    cv2.imshow('out', img_all)
                    if cv2.waitKey(1) == ord('q'):
                        break

            tp = 0
            fp = 0
            average_precision += precision
            average_recall += recall
            coordinates.append([recall, precision])
            average_counter += 1

        coordinates = sorted(coordinates, key=lambda k: [k[0], k[1]])
        x_coordinates, y_coordinates = list(map(lambda c: c[0], coordinates)), list(map(lambda c: c[1], coordinates))

        mAP += np.trapz(y=y_coordinates, x=x_coordinates)
        time2 = time.time()
        latency += time2 - time1

    file_object = open(FLAGS.output, 'a')
    mAP = mAP / 10
    latency = latency / 10
    average_precision = average_precision / average_counter
    average_recall = average_recall / average_counter
    file_object.write(f'\n{FLAGS.weights[14:]} mAP: {mAP} average_precision: {average_precision} average_recall: {average_recall} latency: {latency}')
    file_object.close()
    #print(f'mAP: {mAP} average_precision: {average_precision} average_recall: {average_recall} latency: {latency}')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
