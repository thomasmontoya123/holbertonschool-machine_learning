#!/usr/bin/env python3
"""one-hot decode module"""

import numpy as np


def one_hot_decode(one_hot):
    """
        converts a one-hot matrix into a vector of labels
            Parameters
            ----------
            one_hot : numpy.ndarray
                shape (classes, m) hot encoded
    """
    if type(one_hot) != np.ndarray or one_hot.ndim != 2:
        return None

    return np.argmax(one_hot, axis=0)
