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
    if type(Y) != np.ndarray or len(Y) < 1:
        return None
    if (type(classes) != int or classes < 1 or classes < np.amax(Y)):
        return None

    m = len(Y)
    one_hot = np.zeros((classes, m))
    rows = np.arange(m)
    one_hot[Y, rows] = 1

    return one_hot
