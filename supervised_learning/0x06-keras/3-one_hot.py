#!/usr/bin/env python3
"""One-hot with keras module"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    converts a label vector into a one-hot matrix
        :param labels: labels
        :param classes: number of classes
    """
    return K.utils.to_categorical(y=labels,
                                  num_classes=classes)
