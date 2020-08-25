#!/usr/bin/env python3
"""Precision module """

import numpy as np


def precision(confusion):
    """
    calculates the precision for each class in a confusion matrix:
        :param confusion: confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column indices
            represent the predicted labels

        :return: numpy.ndarray of shape (classes,) containing the precision
            of each class
    """
    return np.diagonal(confusion) / np.sum(confusion, axis=0)
