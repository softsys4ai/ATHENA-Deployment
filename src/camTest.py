import cv2
import tensorflow as tf


vid = cv2.VideoCapture(0)
devices = tf.config.experimental.list_physical_devices()
print(devices)



while True:
	_, img = vid.read()
	cv2.imshow('output', img)
	
	if cv2.waitKey(1) == ord('q'):
		break
