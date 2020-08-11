#!/usr/bin/env python3
"""one-hot module"""

import numpy as np


def one_hot_encode(Y, classes):
    """
        converts a numeric label vector into a one-hot matrix.
            Parameters
            ----------
            Y : numpy.ndarray
                shape (m,) containing numeric class labels
            classes : int
                maximum number of classes found in Y
    """
    shape = (classes, Y.max() + 1)
    one_hot = np.zeros(shape)
    rows = np.arange(classes)
    one_hot[Y, rows] = 1
    return one_hot
