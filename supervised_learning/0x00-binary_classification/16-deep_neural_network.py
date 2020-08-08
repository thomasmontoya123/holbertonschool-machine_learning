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
        elif type(layers) != list or not layers:
            raise TypeError("layers must be a list of positive integers")

        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            if type(i) != int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))

            if i == 0:
                sqrt_0 = np.sqrt(2 / nx)
                self.weights['W1'] = np.random.randn(layers[i], nx) * sqrt_0
            else:
                sqrt = np.sqrt(2 / layers[i - 1])
                formula = np.random.randn(layers[i], layers[i - 1]) * sqrt
                self.weights["W{}".format(i + 1)] = formula
