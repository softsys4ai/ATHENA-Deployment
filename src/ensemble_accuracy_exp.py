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
from yolov3_tf2.dataset import load_tfrecord_dataset, transform_images


flags.DEFINE_list('wds', ['clean'], 'type the desired weak defence. type the name multiple times for multiple '
                                         'instances of WD')
flags.DEFINE_integer('gpu', None, 'set which gpu to use')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
#flags.DEFINE_string('input', './data/meme.jpg', 'path to input image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_string('dataset', './data/clean_test_set.tfrecord', 'path to dataset')
flags.DEFINE_string('output', 'ensemble_mAP.txt', 'path to output image')
flags.DEFINE_integer('sensitivity', 10, 'controls the sensitivity of majority voting')
flags.DEFINE_boolean('show_img', False, 'controls weather or not images are shown')



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

def doThis(ensemble):
    models = []
    for wd in ensemble:
        wd_model = YoloV3(classes=FLAGS.num_classes)
        weights = f'./checkpoints/yolov3_{wd}/yolov3_{wd}.tf'
        wd_model.load_weights(weights).expect_partial()
        models.append(WeakDefence(wd_model, wd, FLAGS.size))
    logging.info('ensemble loaded')

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
    for accuracy in range(10): #TODO: remove for loop and have iou calculated at all levels at the same time
        time1 = time.time()
        for image, labels in dataset.take(100):#100):
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

            #boxes2, scores2, classes2, nums2 = wrapped_yolo.predict(img)
            boxes2 = []
            scores2 = []
            classes2 = []
            for model in models:
                boxes_temp, scores_temp, classes_temp, _ = model.predict(tf.identity(img))
                boxes2 = np.concatenate((boxes2, boxes_temp), axis=1) if np.size(boxes2) else boxes_temp
                scores2 = np.concatenate((scores2, scores_temp), axis=1) if np.size(scores2) else scores_temp
                classes2 = np.concatenate((classes2, classes_temp), axis=1) if np.size(classes2) else classes_temp

            boxes2 = np.squeeze(boxes2, axis=0)
            scores2 = np.squeeze(scores2, axis=0)
            classes2 = np.squeeze(classes2, axis=0)

            selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
                boxes2, scores2, max_output_size=100, iou_threshold=0.5, score_threshold=0.5, soft_nms_sigma=0.5)

            num_valid_nms_boxes = tf.shape(selected_indices)[0]

            selected_indices = tf.concat(
                [selected_indices, tf.zeros(FLAGS.yolo_max_boxes - num_valid_nms_boxes, tf.int32)], 0)
            selected_scores = tf.concat(
                [selected_scores, tf.zeros(FLAGS.yolo_max_boxes - num_valid_nms_boxes, tf.float32)], -1)

            boxes2 = tf.gather(boxes2, selected_indices)
            boxes2 = tf.expand_dims(boxes2, axis=0)
            scores2 = selected_scores
            scores2 = tf.expand_dims(scores2, axis=0)
            classes2 = tf.gather(classes2, selected_indices)
            classes2 = tf.expand_dims(classes2, axis=0)
            valid_detections = num_valid_nms_boxes
            valid_detections = tf.expand_dims(valid_detections, axis=0)
            #print(boxes2, scores2, classes2, valid_detections)
            if tf.equal(valid_detections,0):
                #boxes2, scores2, classes2, valid_detections = boxes, scores, classes, valid_detections
                boxes2, scores2, classes2, valid_detections = tf.zeros([1,100,4], tf.float32), tf.zeros([1,100], tf.float32), tf.zeros([1,100], tf.int64), tf.zeros([1,], tf.int32)
            else:
                boxes2, scores2, classes2, valid_detections = majority_voting((boxes2, scores2, classes2, valid_detections), FLAGS.size, FLAGS.sensitivity)

            pred_boxes, pred_scores, pred_classes, pred_nums = boxes2[0], scores2[0], classes2[0], valid_detections[0]


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
                img2 = draw_outputs(copy.copy(image), (boxes2, scores2, classes2, valid_detections), class_names)
                img_all = np.concatenate((img1, img2), axis=1)
                img_all = cv2.putText(img_all, f'TP = {tp} FP = {fp} precision = {precision} recall = {recall}', (0, 30),
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
    file_object.write(f'\n{FLAGS.wds} mAP: {mAP} average_precision: {average_precision} average_recall: {average_recall} latency: {latency}')
    file_object.close()







if __name__ == '__main__':
    try:
        app.run(main)

    except SystemExit:
        pass

    doThis(['clean', 'salt', 'pepper', 'gaussian', 'poisson', 'compress_png_8'])
    doThis(['salt', 'pepper', 'gaussian', 'poisson', 'compress_png_8'])
    doThis(['pepper', 'gaussian', 'poisson', 'compress_png_8'])
    doThis(['gaussian', 'poisson', 'compress_png_8'])
    doThis(['poisson', 'compress_png_8'])
    doThis(['clean', 'salt', 'pepper', 'gaussian', 'poisson'])
    doThis(['clean', 'salt', 'pepper', 'gaussian'])
    doThis(['clean', 'salt', 'pepper'])
    doThis(['clean', 'salt'])
    doThis(['salt', 'pepper', 'gaussian'])
    doThis(['salt', 'pepper'])
    doThis(['salt'])
    doThis(['pepper', 'gaussian', 'poisson'])
    doThis(['gaussian', 'poisson'])
    doThis(['poisson'])
    doThis(['salt', 'pepper', 'gaussian', 'poisson'])

    doThis(['clean'])
    doThis(['salt'])
    doThis(['pepper'])
    doThis(['gaussian'])
    doThis(['compress_png_8'])






