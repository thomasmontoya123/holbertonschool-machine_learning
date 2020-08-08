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

    def evaluate(self, X, Y):
        """
                Evaluates the neural network’s predictions
                Parameters
                ----------
                X : numpy.ndarray
                    shape (nx, m) contains the input data

                Y : numpy.ndarray
                    shape (1, m) Contains the correct labels
                    for the input data
        """
        threshold = 0.5
        A2 = self.forward_prop(X)
        A2 = np.where(self.__A2 >= threshold, 1, 0)
        cost = self.cost(Y, self.__A2)
        return A2, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
            Calculates one pass of gradient descent on the neuron
                Parameters
                ----------
                X : numpy.ndarray
                    shape (nx, m) contains the input data

                Y : numpy.ndarray
                    shape (1, m) Contains the correct labels
                    for the input data

                A1 : numpy.ndarray
                    is the output of the hidden layer
                    shape (1, m) Contains predictions

                A2 : numpy.ndarray
                    is the predicted output

                alpha : float
                    learning rate
        """
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = np.matmul(A1, dZ2.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.matmul(X, dZ1.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.__W1 = self.__W1 - (alpha * dW1).T
        self.__b1 = self.__b1 - alpha * db1
        self.__W2 = self.__W2 - (alpha * dW2).T
        self.__b2 = self.__b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
            Trains the neuron
                Parameters
                ----------
                X : numpy.ndarray
                    shape (nx, m) contains the input data

                Y : numpy.ndarray
                    shape (1, m) Contains the correct labels
                    for the input data

                iterations : int
                    is the number of iterations to train over

                alpha : float
                    learning rate
        """

        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        elif iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        elif type(alpha) != float:
            raise TypeError("alpha must be a float")
        elif alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

        return self.evaluate(X, Y)
