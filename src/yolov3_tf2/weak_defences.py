import tensorflow as tf
#from yolov3_tf2.dataset import transform_images
import skimage
import numpy as np
from absl import logging
import cv2

class WeakDefence(object):
    def __init__(self, model, trans_configs, size):
        self._model = model
        self._trans_configs = trans_configs
        self._size = size
        #tf.config.run_functions_eagerly(False)


    def transformation(self, x):
        if self._trans_configs == 'clean':
            x = x / 255
            return x
        elif self._trans_configs == 'gaussian':
            x = skimage.util.random_noise(x, mode='gaussian', seed=None, clip=True)
            return x
        elif self._trans_configs == 'salt':
            x = skimage.util.random_noise(x, mode='salt', seed=None, amount=0.05)
            return x
        elif self._trans_configs == 'pepper':
            x = skimage.util.random_noise(x, mode='pepper', seed=None, amount=0.05)
            return x
        elif self._trans_configs == 'poisson':
            x = skimage.util.random_noise(x, mode='poisson', seed=None, clip=True)
            return x
        elif self._trans_configs == 'flip_both':
            x = np.flip(x, axis=1)
            x = np.flip(x, axis=0)
            x = x / 255
            return x
        elif self._trans_configs == 'compress_png_8':
            encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 8]
            result, x = cv2.imencode('.png', x, encode_param)
            logging.info(result)
            x = cv2.imdecode(buf=x, flags=1)
            x = x / 255
            return x
        else:# TODO: clean is returned twice. should else throw an error?
            logging.info("no transformation selected")
            x = x / 255
            return x


    def get_image(self, x):
        x = self.transformation(x)
        return x


    def predict(self, x):
        """
                Perform prediction for a input.
                :param x: image.
                :type x: `np.ndarray`
                :return: tuple of prediction information of format `(boxes, scores, classes, nums)`.
                :rtype: `tuple`
                boxes, scores, classes, nums are all np.ndarray
        """
        x = self.transformation(x)
        x = tf.expand_dims(x, 0)
        x = tf.image.resize(x, (self._size, self._size))

        return self._model.predict(x)
