#!/usr/bin/env python3
"""YOLO class module"""

import tensorflow.keras as K


class Yolo(object):
    """Uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Constructor
            :param self: obj

            :param model_path: path to where a Darknet Keras model is stored

            :param classes_path: path to where the list of class names used
                for the Darknet model, listed in order of index, can be found

            :param class_t: float representing the box score threshold for the
                initial filtering step

            :param nms_t: float representing the IOU threshold for non-max
                suppression
            :param anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
                containing all of the anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [class_n.strip() for class_n in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
