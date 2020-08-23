#!/usr/bin/env python3
"""Batch normalization module"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
       normalizes an unactivated output of a neural
       network using batch normalization
            Parameters
            ----------
            Z : numpy.ndarray
                shape (m, n) that should be normalized

            gamma : numpy.ndarray
                shape (1, n) containins the scales used for batch normalization

            beta : numpy.ndarray
                shape (1, n) containins the offsets used for
                batch normalization

            epsilon : float
                small number used to avoid division by zero
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    normalized_data = (Z - mean) / ((variance + epsilon) ** (1 / 2))
    return gamma * normalized_data + beta
