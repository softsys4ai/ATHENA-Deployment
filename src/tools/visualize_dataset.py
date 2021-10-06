import sys
sys.path.append("./")
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import load_tfrecord_dataset #TODO fix this
from yolov3_tf2.utils import draw_outputs #TODO fix this no import

flags.DEFINE_string('classes', '../data/coco.names', 'path to classes file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('dataset', '../data/coco2017_train.tfrecord', 'path to dataset')
flags.DEFINE_string('output', '../output.jpg', 'path to output image')


def main(_argv):
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    dataset = load_tfrecord_dataset(FLAGS.dataset, FLAGS.classes, FLAGS.size)
    #dataset = dataset.shuffle(512, seed=7) #7 is e+01
    #print(dataset)
    for image, labels in dataset.take(7):
        print(image)
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

        logging.info('labels:')
        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                               np.array(scores[0][i]),
                                               np.array(boxes[0][i])))

        img = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
        #img = img / 255
        #print(img)
        #print(type(img))
        #print(np.shape(img))
        #print(type(img[0][0][0]))

        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)

        #cv2.imshow('out', img)
        #cv2.waitKey(9000)
        cv2.imwrite(FLAGS.output, img)
        logging.info('output saved to: {}'.format(FLAGS.output))


if __name__ == '__main__':
    app.run(main)
