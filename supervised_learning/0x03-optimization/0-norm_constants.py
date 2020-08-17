#!/usr/bin/env python3
"""Constant normalization module"""

import numpy as np


def normalization_constants(X):
    """
        calculates the normalization (standardization)
        constants of a matrix
            Parameters
            ----------
            X : numpy.ndarray
                shape (m, nx) to normalize
                    m is the number of data points
                    nx is the number of features.
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
