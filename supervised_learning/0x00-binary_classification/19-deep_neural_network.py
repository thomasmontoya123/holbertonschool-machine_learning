#!/usr/bin/env python3
"""DeepNeuralNetwork module"""

import numpy as np


def sigmoid(z):
    """ Activation function used to map any real value between 0 and 1 """
    return 1 / (1 + np.exp(-z))


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
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if type(i) != int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))

            if i == 0:
                sqrt_0 = np.sqrt(2 / nx)
                self.__weights['W1'] = np.random.randn(layers[i], nx) * sqrt_0
            else:
                sqrt = np.sqrt(2 / layers[i - 1])
                formula = np.random.randn(layers[i], layers[i - 1]) * sqrt
                self.__weights["W{}".format(i + 1)] = formula

    @property
    def L(self):
        """L getter"""
        return self.__L

    @property
    def cache(self):
        """cache getter"""
        return self.__cache

    @property
    def weights(self):
        """weights getter"""
        return self.__weights

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neural network
                Parameters
                ----------
                X : numpy.ndarray
                    shape (nx, m) that contains the input data
        """
        self.__cache['A0'] = X

        values_1 = np.matmul(self.weights['W1'], X) + self.weights['b1']
        A = sigmoid(values_1)
        self.__cache['A1'] = A

        for i in range(1, self.__L):
            w = self.__weights['W{}'.format(i + 1)]
            b = self.__weights['b{}'.format(i + 1)]
            values = np.matmul(w, A) + b
            A = sigmoid(values)
            self.__cache['A{}'.format(i + 1)] = A

        return A, self.cache

    def cost(self, Y, A):
        """
            Calculates the cost of the model using logistic regression
                Parameters
                ----------
                Y : numpy.ndarray
                    shape (1, m) Contains the correct labels
                    for the input data
                A : numpy.ndarray
                    shape (1, m) containing the activated output
                    (Predictions) of the neuron for each example
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A))) / m
        return cost
