#!/usr/bin/env python3
"""l2 regularization module"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    calculates the cost of a neural network with L2 regularization
        :param cost: cost of the network without L2 regularization
        :param lambtha: regularization parameter
        :param weights: dictionary of the weights and biases
            (numpy.ndarrays) of the neural network

        :param L: number of layers in the neural network
        :param m: number of data points used
        :return : cost of the network accounting for L2 regularization
    """
    norm = 0
    for i in range(L):
        W = weights["W{}".format(i + 1)]
        norm += np.linalg.norm(W)

    L2_cost = cost + (lambtha / (2 * m)) * norm
    return L2_cost
