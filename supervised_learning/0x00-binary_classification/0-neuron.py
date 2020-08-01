#!/usr/bin/env python3
"""Neuron module"""

import numpy as np


class Neuron(object):
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """
                Constructor
                ...

                Attributes
                ----------
                nx : int
                    number of input features to the neuron
                W : numpy.array
                    The weights vector for the neuron
                b : int
                    The bias for the neuron
                A : int
                    The activated output of the neuron (prediction)

        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")

        elif nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.normal(0, 1, (1, nx))
        self.b = 0
        self.A = 0
