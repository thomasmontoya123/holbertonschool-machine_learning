#!/usr/bin/env python3
"""dropout forward propagation module"""

import numpy as np


def softmax(z):
    """ softmax activation fuction """
    exponentiated = np.exp(z)
    probability = exponentiated / np.sum(exponentiated, axis=0, keepdims=True)
    return probability


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    conducts forward propagation using Dropout
        :param X: numpy.ndarray of shape (nx, m) containing
            the input data for the network
        :param weights: dictionary of the weights and biases
            of the neural network
        :param L: number of layers in the network
        :param keep_prob: probability that a node will be kept
    """
    cache = {'A0': X}

    for i in range(L):
        w = weights['W{}'.format(i + 1)]
        b = weights['b{}'.format(i + 1)]
        A_key = 'A{}'.format(i + 1)
        dropout_key = 'D{}'.format(i + 1)

        values = np.matmul(w, cache['A{}'.format(i)]) + b
        dropout = np.random.binomial(1, keep_prob, size=values.shape)

        if i == L - 1:
            A = softmax(values)
            cache[A_key] = A

        else:
            A = np.tanh(values)
            cache[dropout_key] = dropout
            cache[A_key] = (A * cache[dropout_key]) / keep_prob

    return cache
