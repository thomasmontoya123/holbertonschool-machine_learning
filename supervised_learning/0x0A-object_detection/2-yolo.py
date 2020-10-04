#!/usr/bin/env python3
"""YOLO class module"""

import tensorflow.keras as K
import numpy as np


def sigmoid(x):
    """ Calculates sigmoid"""
    return 1 / (1 + np.exp(-x))


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

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet Outputs

            :param outputs: list of numpy.ndarrays containing the predictions
                from the Darknet model for a single image.
                Each output will have the shape
                (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)

            :param image_size: numpy.ndarray containing the image’s original
                size [image_height, image_width]

            :returns : (boxes, box_confidences, box_class_probs)
                boxes: a list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 4) containing the
                processed boundary boxes for each output, respectively
                box_confidences: a list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 1) containing the box
                confidences for each output, respectively
                box_class_probs: a list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, classes)
                containing the box’s class probabilities for each output,
        """
        image_h, image_w = image_size
        input_w = self.model.input.shape[1].value
        input_h = self.model.input.shape[2].value

        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            boxes.append(output[..., 0:4])
            box_confidences.append(sigmoid(output[..., 4, np.newaxis]))
            box_class_probs.append(sigmoid(output[..., 5:]))

        for i, box in enumerate(boxes):
            grid_h, grid_w, anchor_b, _ = box.shape

            c = np.zeros((grid_h, grid_w, anchor_b))

            index_x = np.arange(grid_w)
            index_y = np.arange(grid_h)

            index_x = index_x.reshape(1, grid_w, 1)
            index_y = index_y.reshape(grid_h, 1, 1)

            cx = c + index_x
            cy = c + index_y

            tx = (box[..., 0])
            ty = (box[..., 1])
            tw = (box[..., 2])
            th = (box[..., 3])

            bx = (sigmoid(tx) + cx) / grid_w
            by = (sigmoid(ty) + cy) / grid_h

            ph = self.anchors[i, :, 1]
            pw = self.anchors[i, :, 0]

            bh = (ph * np.exp(th)) / input_h
            bw = (pw * np.exp(tw)) / input_w

            x1 = bx - bw / 2
            x2 = x1 + bw
            y1 = by - bh / 2
            y2 = y1 + bh

            box[..., 0] = x1 * image_w
            box[..., 1] = y1 * image_h
            box[..., 2] = x2 * image_w
            box[..., 3] = y2 * image_h

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes

            :param boxes: list of numpy.ndarrays of shape
                grid_height, grid_width, anchor_boxes, 4) containing the
                processed boundary boxes for each output, respectively

            :param box_confidences: list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 1) containing the
                processed box confidences for each output, respectively

            :param box_class_probs: box_class_probs: a list of numpy.ndarrays
                of shape (grid_height, grid_width, anchor_boxes, classes)
                containing the processed
                box class probabilities for each output, respectively

            :returns: tuple of (filtered_boxes, box_classes, box_scores)
        """
        scores = []
        for confidence, class_prob in zip(box_confidences, box_class_probs):
            scores.append(confidence * class_prob)

        score_max = [score.argmax(axis=3) for score in scores]
        score_max = [score.reshape(-1) for score in score_max]
        box_scores = np.concatenate(score_max)
        min_prob_idx = np.where(box_scores < self.class_t)
        box_scores = np.delete(box_scores, min_prob_idx)

        class_max = [box.argmax(axis=3) for box in scores]
        class_max = [box.reshape(-1) for box in class_max]
        box_classes = np.concatenate(class_max)
        box_classes = np.delete(box_classes, min_prob_idx)

        boxes_list = [box.reshape(-1, 4) for box in boxes]
        boxes = np.concatenate(boxes_list, axis=0)
        filtered_boxes = np.delete(boxes, min_prob_idx, axis=0)

        return filtered_boxes, box_classes, box_scores
