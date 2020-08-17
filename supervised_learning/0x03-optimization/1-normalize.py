#!/usr/bin/env python3
"""Normalization module"""

import numpy as np


def normalize(X, m, s):
    """
        Normalizes (standardizes) a matrix
            Parameters
            ----------
            X : numpy.ndarray
                shape (d, nx) to normalize
                    d is the number of data points
                    nx is the number of features.
            m : numpy.ndarray
                shape (nx,) that contains the mean of all features of X
            s : numpy.ndarray
                shape (nx,) that contains the standard deviation of all
                features of X
    """
    return (X - m) / s
