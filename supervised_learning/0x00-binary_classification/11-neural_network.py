#!/usr/bin/env python3
"""Neural network module"""

import numpy as np


def sigmoid(z):
    """ Activation function used to map any real value between 0 and 1 """
    return 1 / (1 + np.exp(-z))


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

    def forward_prop(self, X):
        """
            Calculates the forward propagation of the neural network
                Parameters
                ----------
                X : numpy.ndarray
                    shape (nx, m) that contains the input data
        """
        values_1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = sigmoid(values_1)
        values_2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = sigmoid(values_2)

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
            Calculates the forward propagation of the neuron.
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
