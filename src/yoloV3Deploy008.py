import cv2
import numpy as np
from keras.models import load_model
from keras_preprocessing.image import img_to_array
from src.yolo3_one_file_to_detect_them_all import decode_netout, correct_yolo_boxes, do_nms
import tensorflow as tf
from multiprocessing import Process, Array, shared_memory
import ctypes
import sys
import time

import os
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["OMP_NUM_THREADS"] = '4'
config = tf.compat.v1.ConfigProto()
config.intra_op_parallelism_threads = 4
config.inter_op_parallelism_threads = 1
tf.compat.v1.Session(config=config)

def resize_image(img, size):

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)

def shared_array(dtype, shape):
    """
    Form a shared memory numpy array.

    https://stackoverflow.com/q/5549190/2506522
    """
    size = sys.getsizeof(dtype())
    for dim in shape:
        size *= dim

    shared_array_base = Array(ctypes.c_double, size)
    shared_array = np.ndarray(shape, dtype=dtype, buffer=shared_array_base.get_obj())
    shared_array = shared_array.reshape(*shape)
    return shared_array




#load image with preprocessing
def load_image_pixels(image, shape):
    # load the image to get its shape
    height, width, x = image.shape

    # load the image with the required size
    image = resize_image(image, shape)
    #image = cv2.resize(image, shape, interpolation=cv2.INTER_AREA)
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
				v_scores.append(box.classes[i]*100)
				# don't break, many labels may trigger for one box
	return v_boxes, v_labels, v_scores


def WDProcess(i):
    child = shared_memory.SharedMemory(name='videoFrames')
    child2 = shared_memory.SharedMemory(name='videoFrames2')
    frame_data, WD = i
    child_frames = np.ndarray(frame_data.shape, dtype=frame_data.dtype, buffer=child2.buf)

    # --- set up all the things needed to do one image ---
    # load yolov3 model
    model = load_model('src\yolov3-tiny.h5')
    # define the expected input shape for the model
    #input_w, input_h = 416, 416
    input_w, input_h = 608, 608
    # define the anchors
    anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
    # define the probability threshold for detected objects
    class_threshold = 0.4
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

    child_output = np.ndarray(child_frames[WD].shape, dtype=child_frames[WD].dtype, buffer=child.buf)

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 0.5
    # Blue color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness = 2

    while True:
        #get frame
        frame = child_frames[WD]
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
        do_nms(boxes, 0.5)

        # get the details of the detected objects
        v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

        # summarize what we found
        for i in range(len(v_boxes)):

            #print(v_labels[i], v_scores[i])
            box = v_boxes[i]
            # get coordinates
            frame = cv2.rectangle(frame, (box.xmin, box.ymin), (box.xmax, box.ymax), color, thickness)
            # Using cv2.putText() method
            frame = cv2.putText(frame, v_labels[i] + " (" + str(int(v_scores[i])) + "%)", (box.xmin, box.ymin-3), font, fontScale, color, thickness-1, cv2.LINE_AA)

            frame = cv2.putText(frame, "WD {0:n}".format(WD), (10, 20), font, fontScale, color,
                                thickness - 1, cv2.LINE_AA)


        child_output[:] = frame[:]



if __name__ == "__main__":

    workerNum = 5

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    shape_array = []
    for i in range(workerNum):
        ret, frame = cap.read()
        shape_array.append(frame)
    shape_array = np.array(shape_array)

    #shared_frames = np.expand_dims(frame, 0)
    parent2 = shared_memory.SharedMemory(name='videoFrames2', create=True, size=shape_array.nbytes)
    shared_frames = np.ndarray(shape_array.shape, dtype=frame.dtype, buffer=parent2.buf)
    shared_frames[:] = shape_array[:]

    # Form a shared array and a lock, to protect access to shared memory.
   # lock = multiprocessing.Lock()
    parent = shared_memory.SharedMemory(name='videoFrames', create=True, size=frame.nbytes)
    output = np.ndarray(frame.shape, dtype=frame.dtype, buffer=parent.buf)
    output[:] = frame[:]

    workers = []
    for i in range(workerNum):
        workers.append(Process(target=WDProcess, args=((shared_frames,i),)))
        workers[-1].daemon = True
        workers[-1].start()
        time.sleep(.3)

    while True:
        time1= time.time()
        for i in range(workerNum):
            ret, frame = cap.read()
            shared_frames[i] = frame
            cv2.imshow("Web cam input", output)
            cv2.waitKey(1)



