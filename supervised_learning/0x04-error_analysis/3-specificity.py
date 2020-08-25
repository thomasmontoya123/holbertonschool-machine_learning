#!/usr/bin/env python3
"""Specificity module"""

import numpy as np


def specificity(confusion):
    """
    calculates the specificity for each class in a confusion matrix
        :param confusion: confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column indices
            represent the predicted labels

        :return: numpy.ndarray of shape (classes,) containing the specificity
            of each class
    """
    true_positive = np.diag(confusion)
    false_positve = np.sum(confusion, axis=0) - true_positive

    false_negative = np.sum(confusion, axis=1) - true_positive
    true_negative = np.sum(confusion) - (false_positve +
                                         false_negative + true_positive)

    return true_negative / (false_positve + true_negative)
