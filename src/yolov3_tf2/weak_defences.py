import tensorflow as tf



class WeakDefence(object):
    def __init__(self, model, trans_configs):
        self._model = model
        self._trans_configs = trans_configs
        tf.config.run_functions_eagerly(False)

    def transformation(self, x, trans_args):
        if trans_args == 'clean':
            return x
        elif trans_args == 'gaussian':
            return x
        elif trans_args == 'salt':
            return x
        elif trans_args == 'pepper':
            return x
        elif trans_args == 'flip':
            return x
        elif trans_args == 'mirror':
            return x

    def predict(self, x, **kwargs):
        """
                Perform prediction for a input.
                :param x: image.
                :type x: `tensorflow.python.framework.ops.EagerTensor`
                :return: tuple of prediction information of format `(boxes, scores, classes, nums)`.
                :rtype: `tuple`
                boxes, scores, classes, nums are all np.ndarray
        """

        x = self.transformation(self, x, self._trans_configs)


        return self._model.predict(x)
