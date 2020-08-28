#!/usr/bin/env python3
"""Dropout gradient descent module"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    updates the weights of a neural network with Dropout regularization
    using gradient descent
        :param Y: one-hot numpy.ndarray of shape (classes, m) that contains
            the correct labels for the data
        :param weights: dictionary of the weights and biases
            of the neural network
        :param cache: dictionary of the outputs and dropout masks
            of each layer of the neural network
        :param alpha: learning rate
        :param keep_prob: probability that a node will be kept
        :param L: number of layers of the network
    """
    m = Y.shape[1]
    weights_c = weights.copy()

    for i in range(L, 0, -1):
        A = cache["A{}".format(i)]
        if i == L:
            dz = A - Y
        else:
            dz = np.matmul(weights_c["W{}".format(i + 1)].T, dz) * (1 - A ** 2)
            dz *= cache["D{}".format(i)] / keep_prob

        dw = np.matmul(dz, cache["A{}".format(i - 1)].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        w = weights_c["W{}".format(i)]
        b = weights_c["b{}".format(i)]

        weights["W{}".format(i)] = w - alpha * dw
        weights["b{}".format(i)] = b - alpha * db
