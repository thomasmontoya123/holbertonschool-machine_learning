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

        self.W1 = np.random.normal(0, 1, (nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(0, 1, (1, nodes))
        self.b2 = 0
        self.A2 = 0
