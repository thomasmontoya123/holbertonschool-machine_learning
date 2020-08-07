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

        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Weight Getter"""
        return self.__W

    @property
    def b(self):
        """bias getter"""
        return self.__b

    @property
    def A(self):
        """Prediction getter"""
        return self.__A

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neuron.
                Parameters
                ----------
                X : numpy.ndarray.
                    shape (nx, m) that contains the input data

        """
        values = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1 + np.exp(-values))  # Sigmoid
        return self.__A
