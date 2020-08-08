#!/usr/bin/env python3
"""DeepNeuralNetwork module"""

import numpy as np


class DeepNeuralNetwork(object):
    """defines a deep neural network"""

    def __init__(self, nx, layers):
        """
            Constructor
            ...

            Attributes
            ----------
            nx : int
                number of input features to the neuron
            layers : list
                represents the number of nodes in each
                layer of the network
        """

        if type(nx) != int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        elif type(layers) != list:
            raise TypeError("layers must be a list of positive integers")

        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i, n in enumerate(layers):
            if type(n) != int or n < 1:
                raise TypeError('layers must be a list of positive integers')
            self.weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
            if i == 0:
                self.weights['W1'] = np.random.randn(n, nx) * np.sqrt(2 / nx)
            else:
                j = layers[i - 1]
                ws = np.random.randn(n, j) * np.sqrt(2 / j)
                self.weights["W{}".format(i + 1)] = ws
