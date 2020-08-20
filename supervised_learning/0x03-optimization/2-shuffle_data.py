#!/usr/bin/env python3
"""Normalization module"""

import numpy as np


def shuffle_data(X, Y):
    """
        Normalizes (standardizes) a matrix
                Parameters
                ----------
                X : numpy.ndarray
                    shape (m, nx) to normalize
                        m is the number of data points
                        nx is the number of features in X
                Y : numpy.ndarray
                    shape (m, ny) to normalize
                        m is the number of data points
                        ny is the number of features in Y
    """
    indices = np.random.permutation(X.shape[0])
    return X[indices], Y[indices]
