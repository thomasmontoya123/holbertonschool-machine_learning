#!/usr/bin/env python3
"""Neuron module"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    """ Activation function used to map any real value between 0 and 1 """
    return 1 / (1 + np.exp(-z))


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
                X : numpy.ndarray
                    shape (nx, m) that contains the input data
        """
        values = np.matmul(self.__W, X) + self.__b
        self.__A = sigmoid(values)
        return self.__A

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

    def evaluate(self, X, Y):
        """
                Evaluates the neuron’s predictions
                Parameters
                ----------
                X : numpy.ndarray
                    shape (nx, m) contains the input data

                Y : numpy.ndarray
                    shape (1, m) Contains the correct labels
                    for the input data
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        threshold = 0.5
        A = np.where(A >= threshold, 1, 0)
        return A, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
            Calculates one pass of gradient descent on the neuron
                Parameters
                ----------
                X : numpy.ndarray
                    shape (nx, m) contains the input data

                Y : numpy.ndarray
                    shape (1, m) Contains the correct labels
                    for the input data

                A : numpy.ndarray
                    shape (1, m) Contains predictions

                alpha : float
                    learning rate
        """
        m = Y.shape[1]
        dz = A - Y
        db = np.sum(dz) / m
        dw = np.matmul(X, dz.T) / m
        self.__W = self.__W - (alpha * dw).T
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
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

                verbose : bool
                    defines whether or not to print information
                    about the training

                graph : bool
                    defines whether or not to graph information
                    about the training

                step : int
                    How many iterations show data
        """

        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        elif iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        elif type(alpha) != float:
            raise TypeError("alpha must be a float")
        elif alpha <= 0:
            raise ValueError("alpha must be positive")
        elif verbose or graph:
            if type(step) != int:
                raise TypeError("step must be an integer")
            elif step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        iterations_steps = []
        costs_steps = []
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            if i % step == 0 or i == iterations:
                if verbose:
                    print("Cost after {} iterations: {}".
                          format(i, self.cost(Y, self.__A)))
                if graph:
                    iterations_steps.append(i)
                    costs_steps.append(self.cost(Y, self.__A))

        if graph:
            plt.plot(iterations_steps, costs_steps, 'b-')
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.show()
        return self.evaluate(X, Y)
