from absl import logging
import numpy as np
import tensorflow as tf
import cv2
import math

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]


def load_darknet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    if tiny:
        layers = YOLOV3_TINY_LAYER_LIST
    else:
        layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.get_input_shape_at(0)[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]

    wh = np.flip(img.shape[0:2])
#class_names[int(classes[i])] == "person" and
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img

def draw_outputs_bbox_deltas(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]

    wh = np.flip(img.shape[0:2])
    box_coordinates = []
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        box_coordinates.append([x1y1, x2y2])

#class_names[int(classes[i])] == "person" and
    for i in range(nums):
        deltaFlag = True
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        for j in range(i+1, nums):
            x1, y1 = x1y1
            x2, y2 = x2y2
            x3, y3 = box_coordinates[j][0]
            x4, y4 = box_coordinates[j][1]
            size = math.sqrt((pow((y1 - y2), 2) + pow((x1 - x2), 2)))
            top_delta = math.sqrt((pow((y1 - y3), 2) + pow((x1 - x3), 2)))
            bottom_delta = math.sqrt((pow((y2 - y4), 2) + pow((x2 - x4), 2)))
            delta = (top_delta + bottom_delta) * size
            #tf.print(top_delta, bottom_delta, size)
            #delta += sum(abs(list(bottom_delta)))
            #delta /= size

            tf.print(class_names[int(classes[i])] + " vs " + class_names[int(classes[j])], delta)
            if delta < 0.5:
                tf.print(class_names[int(classes[i])] + " vs " + class_names[int(classes[j])], 'lost ' + i)
                deltaFlag = False
                break

        if deltaFlag:
            img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
            img = cv2.putText(img, '{} {:.4f}'.format(
                class_names[int(classes[i])], objectness[i]),
                x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    return img


def majority_voting(outputs, img_size, delta_limit):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]

    #tf.print(boxes))
    #box_coordinates = tf.map_fn(fn=lambda box: tf.math.multiply(box, img_size), elems=boxes)
    #tf.print(box_coordinates.shape, boxes.shape)
    #tf.print(boxes.shape, objectness.shape, classes.shape, nums.shape)
    all_object_boxes, all_object_objectness, all_object_classes, object_num = [], [], [], 0
    for i in range(nums):
        object_boxes, object_objectness, object_classes = [], [], []
        x1, y1, x2, y2 = np.array(boxes[i]) * img_size
        for j in range(i+1, nums):

            x3, y3, x4, y4 = np.array(boxes[j]) * img_size

            #size = math.sqrt((pow((y1 - y2), 2) + pow((x1 - x2), 2)))
            top_delta = math.sqrt((pow((y1 - y3), 2) + pow((x1 - x3), 2)))
            bottom_delta = math.sqrt((pow((y2 - y4), 2) + pow((x2 - x4), 2)))
            delta = (top_delta + bottom_delta) # * size

            if delta < delta_limit:
                object_boxes.append(boxes[j])
                object_objectness.append(objectness[j])
                object_classes.append(classes[j])

                #object.append([boxes[j], objectness[j], classes[j]])
                np.delete(boxes, j)
                np.delete(objectness, j)
                np.delete(classes, j)
                nums -= 1
                j -= 1

        object_boxes.append(boxes[i])
        object_objectness.append(objectness[i])
        object_classes.append(classes[i])

        #object_boxes.append(boxes[i])#TODO: delete this
        #object_objectness.append(objectness[i])#TODO: delete this
        #object_classes.append(classes[i])#TODO: delete this
        all_object_boxes.append(object_boxes)
        all_object_objectness.append(object_objectness)
        all_object_classes.append(object_classes)
        object_num += 1
        #object.append([boxes[i], objectness[i], classes[i]])
        #object.append([boxes[i], objectness[i], classes[i]])
        #objects.append(object)


    #boxes, objectness, classes, nums = [], [], [], len(objects)
    all_object_classes = tf.ragged.stack(all_object_classes, axis=0) #tf.ragged.constant(all_object_classes) #tf.RaggedTensor.from_tensor(all_object_classes)
    all_object_boxes = tf.ragged.stack(all_object_boxes, axis=0) #tf.convert_to_tensor(all_object_boxes, dtype=tf.float32)
    all_object_objectness = tf.ragged.stack(all_object_objectness, axis=0) #tf.convert_to_tensor(all_object_objectness, dtype=tf.float32)
    object_num = tf.convert_to_tensor(object_num, dtype=tf.int32)

    def get_decided_classes(x):
        cls, _, count = tf.unique_with_counts(x)
        return cls[tf.math.argmax(count)]

    decided_classes = tf.map_fn(fn=get_decided_classes, elems=all_object_classes, fn_output_signature=tf.int64)
    decided_boxes = tf.map_fn(fn=lambda object: object[-1], elems=all_object_boxes, fn_output_signature=tf.float32) #TODO: average the cordinates for more accurate boxes
    decided_objectness = tf.map_fn(fn=lambda object: object[-1], elems=all_object_objectness, fn_output_signature=tf.float32) #TODO: average the objectness for more accurate objectness
    decided_boxes = tf.expand_dims(decided_boxes, axis=0)
    decided_objectness = tf.expand_dims(decided_objectness, axis=0)
    decided_classes = tf.expand_dims(decided_classes, axis=0)
    object_num = tf.expand_dims(object_num, axis=0)
    return (decided_boxes, decided_objectness, decided_classes, object_num)

    #for i in range(len(objects)):
    #    #tf.print('this', np.shape(objects), np.shape(objects[0]), np.shape(objects[0][0]))
    #    object = objects[i]
    #    #tf.print(type(object))
    #    #tf.print("second", object[0][2])
    #    object = tf.convert_to_tensor(object)
    #    god = tf.map_fn(fn=lambda vote: vote[2], elems=object)
    #    #logging.info(type(god))
    #    for j in range(len(object)):
    #        object[j][0]
    #        pass

    #sizes = tf.map_fn(fn=lambda x1, y1, x2, y2: tf.math.sqrt(tf.math.square(x1 - x2) + tf.math.square(y1 - y1)), elems=box_coordinates)

    #TensorShape([1, 100, 4]) TensorShape([1, 100]) TensorShape([1, 100]) TensorShape([1])





def draw_labels(x, y, class_names):
    img = x.numpy()
    boxes, classes = tf.split(y, (4, 1), axis=-1)
    classes = classes[..., 0]
    wh = np.flip(img.shape[0:2])
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, class_names[classes[i]],
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          1, (0, 0, 255), 2)
    return img


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)
