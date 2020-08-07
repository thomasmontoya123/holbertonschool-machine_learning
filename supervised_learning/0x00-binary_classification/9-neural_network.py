#!/usr/bin/env python3
"""Neural network module"""

import numpy as np


class NeuralNetwork(object):
    """Defines a neural network with one hidden layer"""

    def __init__(self, nx, nodes):
        """
            Constructor
            ...

            Attributes
            ----------
            nx : int
                number of input features to the neuron
            nodes : int
                number of nodes found in the hidden layer
        """

        if type(nx) != int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        elif type(nodes) != int:
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(0, 1, (nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(0, 1, (1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Weight1 Getter"""
        return self.__W1

    @property
    def b1(self):
        """bias1 getter"""
        return self.__b1

    @property
    def A1(self):
        """Prediction1 getter"""
        return self.__A1

    @property
    def W2(self):
        """Weight2 Getter"""
        return self.__W2

    @property
    def b2(self):
        """bias2 getter"""
        return self.__b2

    @property
    def A2(self):
        """Prediction2 getter"""
        return self.__A2
