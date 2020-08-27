#!/usr/bin/env python3
"""Gradient descent with l2 regularization module"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    updates the weights and biases of a neural network using gradient
    descent with L2 regularization

        :param Y: one-hot numpy.ndarray of shape (classes, m)
            that contains the correct labels for the data

        :param weights: dictionary of the weights and biases
            of the neural network

        :param cache: dictionary of the outputs of each layer
            of the neural network

        :param alpha: learning rate
        :param lambtha: regularization parameter
        :param L: number of layers of the network
    """
    m = Y.shape[1]
    weights_c = weights.copy()

    for i in range(L, 0, -1):
        A = cache["A{}".format(i)]
        if i == L:
            dz = A - Y
        else:
            dz = np.matmul(weights_c["W{}".format(i + 1)].T, dz) \
                 * A * (1 - A)

        dw = np.matmul(dz, cache["A{}".format(i - 1)].T) / m
        dw_l2 = dw + (lambtha / m) * weights_c['W{}'.format(i)]
        db = np.sum(dz, axis=1, keepdims=True) / m

        w = weights_c["W{}".format(i)]
        b = weights_c["b{}".format(i)]

        weights["W{}".format(i)] = w - alpha * dw_l2
        weights["b{}".format(i)] = b - alpha * db
