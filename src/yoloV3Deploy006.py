import cv2
import numpy as np
from keras.models import load_model
from keras_preprocessing.image import img_to_array
from src.yolo3_one_file_to_detect_them_all import decode_netout, correct_yolo_boxes, do_nms
import tensorflow as tf
import time
from multiprocessing import Process, Array, shared_memory
import ctypes
import sys


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
    image = cv2.resize(image, shape, interpolation=cv2.INTER_AREA)
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
    configuration = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
    session = tf.compat.v1.Session(config=configuration)
    # load yolov3 model
    model = load_model('yolov3.h5')
    # define the expected input shape for the model
    #input_w, input_h = 416, 416
    input_w, input_h = 100, 100
    # define the anchors
    anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
    # define the probability threshold for detected objects
    class_threshold = 0.6
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

    #list of image names
    images = ["knife1.jpg", "pizza1.jpg", "pizza2.jpg", "zebra.jpg"]

    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0



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
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # fontScale
            fontScale = 0.5
            # Blue color in BGR
            color = (0, 0, 255)
            # Line thickness of 2 px
            thickness = 2

            #print(v_labels[i], v_scores[i])
            box = v_boxes[i]
            # get coordinates
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            # Using cv2.putText() method
            #origin of text
            org = (x1, y1-3)
            frame = cv2.putText(frame, v_labels[i] + " (" + str(int(v_scores[i])) + "%)", org, font, fontScale, color, thickness-1, cv2.LINE_AA)



            new_frame_time = time.time()
            # Calculating the fps
            # fps will be number of frame processed in given time frame
            # since their will be most of time error of 0.001 second
            # we will be subtracting it to get more accurate result
            fps = 1 / (new_frame_time - prev_frame_time) if new_frame_time - prev_frame_time != 0 else 0
            prev_frame_time = new_frame_time

            # converting the fps to string so that we can display it on frame
            # by using putText function
            #fps = str(fps)

            #display the fps
            #frame = cv2.putText(frame, "{0:.1f}".format(fps), (10, 20), font, fontScale, color,
                                #thickness - 1, cv2.LINE_AA)

            frame = cv2.putText(frame, "WD {0:n}".format(WD), (10, 20), font, fontScale, color,
                                thickness - 1, cv2.LINE_AA)

        #x, y, z = frame.shape

        child_output = np.ndarray(frame.shape, dtype=frame.dtype, buffer=child.buf)
        child_output[:] = frame[:]
        #output = output.reshape((x, y, z))
        #output = frame[:]
        #cv2.imshow("Web cam input 2", output)
        #cv2.waitKey(1)



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

    while True:

        for i in range(workerNum):
            ret, frame = cap.read()
            shared_frames[i] = frame

        cv2.imshow("Web cam input", output)
        cv2.waitKey(1)
