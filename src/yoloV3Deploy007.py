import cv2
import numpy as np
from keras.models import load_model
from keras_preprocessing.image import img_to_array
from src.yolo3_one_file_to_detect_them_all import decode_netout, correct_yolo_boxes, do_nms
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import tensorflow as tf

import os

os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["OMP_NUM_THREADS"] = '4'
config = tf.compat.v1.ConfigProto()
config.intra_op_parallelism_threads = 4
config.inter_op_parallelism_threads = 1
tf.compat.v1.Session(config=config)
# Hide GPU from visible devices
# tf.config.set_visible_devices([], 'GPU')


import time


def resize_image(img, size=(28, 28)):
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1

    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0] + size[1]) // 2 else cv2.INTER_CUBIC

    x_pos = (dif - w) // 2
    y_pos = (dif - h) // 2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)


# load image with preprocessing
def load_image_pixels(image, shape):
    # load the image to get its shape
    height, width, x = image.shape

    # load the image with the required size
    image = resize_image(image, shape)

    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0

    # add a dimension so that we have one sample
    image = np.expand_dims(image, 0)

    return image, width, height


# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i] * 100)
            # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores


# draw all results
def draw_boxes(frame, v_boxes, v_labels, v_scores, box_color):
    # load the image
    # data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(frame)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color=box_color)
        # draw the box
        ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        pyplot.text(x1, y1, label, color=box_color)
    # show the plot
    pyplot.show()


if __name__ == "__main__":
    # --- set up all the things needed to do one image ---
    # configuration = tf.compat.v1.ConfigProto(device_count={"GPU": 0}, inter_op_parallelism_threads=4)
    # session = tf.compat.v1.Session(config=configuration)
    # load yolov3 model
    model = load_model('yolov3-tiny.h5')
    # define the expected input shape for the model
    input_w, input_h = 608, 608
    # input_w, input_h = 100, 100
    # define the anchors
    anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
    # define the probability threshold for detected objects
    class_threshold = 0.15
    # labels
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    # list of image names
    images = ["knife1.jpg", "pizza1.jpg", "pizza2.jpg", "zebra.jpg"]

    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0

    # time things
    time1 = 0
    time2 = 0

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("We cannot open webcam")
    
    while True:

        ret, frame = cap.read()
        # load image
        image, image_w, image_h = load_image_pixels(frame, (input_w, input_h))

        # make prediction
        yhat = model.predict(image)

        boxes = list()
        for i in range(len(yhat)):
            # decode the output of the network
            boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)

        # correct the sizes of the bounding boxes for the shape of the image
        correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

        # suppress non-maximal boxes
        do_nms(boxes, 0.5)  # SLOW! 1.30
        # boxes = non_max_suppression_fast(boxes, 0.5)

        # get the details of the detected objects
        v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)  # 0.79

        # summarize what we found
        for i in range(len(v_boxes)):
            # if False:
            # i = 0
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # fontScale
            fontScale = 0.5
            # Blue color in BGR
            color = (0, 0, 255)
            # Line thickness of 2 px
            thickness = 2

            # print(v_labels[i], v_scores[i])
            box = v_boxes[i]
            # get coordinates
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            # Using cv2.putText() method
            # origin of text
            org = (x1, y1 - 3)
            frame = cv2.putText(frame, v_labels[i] + " (" + str(int(v_scores[i])) + "%)", org, font, fontScale, color,
                                thickness - 1, cv2.LINE_AA)

            new_frame_time = time.time()
            # Calculating the fps
            # fps will be number of frame processed in given time frame
            # since their will be most of time error of 0.001 second
            # we will be subtracting it to get more accurate result
            fps = 1 / (new_frame_time - prev_frame_time) if new_frame_time - prev_frame_time != 0 else 0
            prev_frame_time = new_frame_time

            # converting the fps to string so that we can display it on frame
            # by using putText function
            # fps = str(fps)

            # display the fps
            frame = cv2.putText(frame, "{0:.1f}".format(fps), (10, 20), font, fontScale, color,
                                thickness - 1, cv2.LINE_AA)
            print(fps)
        # time2 = time.time()
        # print(time2 - time1)
        cv2.imshow("Web cam input", frame)
        cv2.waitKey(1)

        # draw what we found
        # draw_boxes(frame, v_boxes, v_labels, v_scores, 'red')
