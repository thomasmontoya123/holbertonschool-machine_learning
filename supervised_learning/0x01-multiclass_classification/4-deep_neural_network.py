#!/usr/bin/env python3
"""DeepNeuralNetwork module"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


def sigmoid(z):
    """ Activation function used to map any real value between 0 and 1 """
    return 1 / (1 + np.exp(-z))


def softmax(z):
    """ softmax activation fuction """
    exponentiated = np.exp(z)
    probability = exponentiated / np.sum(exponentiated, axis=0, keepdims=True)
    return probability


class DeepNeuralNetwork(object):
    """defines a deep neural network"""

    def __init__(self, nx, layers, activation='sig'):
        """
            Constructor
            ...

            Attributes:
            ----------
            nx : int
                number of input features to the neuron
            layers : list
                represents the number of nodes in each
                layer of the network
            activation : str
                represents the type of activation function
                used in the hidden layers
        """
        activation_functions = ['sig', 'tanh']
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        elif type(layers) != list or not layers:
            raise TypeError("layers must be a list of positive integers")
        elif activation not in activation_functions:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.nx = nx
        self.layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

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

    @property
    def activation(self):
        """activation function getter"""
        return self.__activation

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
        if self.__activation == 'sig':
            A = sigmoid(values_1)
        else:
            A = np.tanh(values_1)
        self.__cache['A1'] = A

        for i in range(1, self.__L):
            w = self.__weights['W{}'.format(i + 1)]
            b = self.__weights['b{}'.format(i + 1)]
            values = np.matmul(w, A) + b
            if i == self.__L - 1:
                A = softmax(values)
            else:
                if self.__activation == 'sig':
                    A = sigmoid(values)
                else:
                    A = np.tanh(values)
            self.__cache['A{}'.format(i + 1)] = A

        return A, self.cache

    def cost(self, Y, A):
        """
            Calculates the cost of the model using logistic regression
                Parameters
                ----------
                Y : one-hot numpy.ndarray
                    shape (classes, m) Contains the correct labels
                    for the input data
                A : numpy.ndarray
                    shape (1, m) containing the activated output
                    (Predictions) of the neuron for each example
        """
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """
                Evaluates the neural network’s predictions
                Parameters
                ----------
                X : numpy.ndarray
                    shape (nx, m) contains the input data

                Y : one-hot numpy.ndarray
                    shape (classes, m) Contains the correct labels
                    for the input data
        """
        A = self.forward_prop(X)[0]
        cost = self.cost(Y, A)
        A_max = np.amax(A, axis=0)
        A = np.where(A == A_max, 1, 0)
        return A, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
            Calculates one pass of gradient descent on the neuron
                Parameters
                ----------

                Y : numpy.ndarray
                    shape (1, m) Contains the correct labels
                    for the input data

                cache : dict
                    contains all the intermediary values of the network

                alpha : float
                    learning rate
        """
        m = Y.shape[1]
        weights_c = self.weights.copy()

        for i in range(self.__L, 0, -1):
            A = cache["A{}".format(i)]
            if i == self.L:
                dz = A - Y
            else:
                if self.__activation == 'sig':
                    dz = np.matmul(weights_c["W{}".format(i + 1)].T, dz) \
                         * A * (1 - A)
                else:
                    dz = np.matmul(weights_c["W{}".format(i + 1)].T, dz)\
                         * (1 - (A ** 2))

            dw = np.matmul(dz, cache["A{}".format(i - 1)].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            w = self.__weights["W{}".format(i)]
            b = self.__weights["b{}".format(i)]

            self.__weights["W{}".format(i)] = w - alpha * dw
            self.__weights["b{}".format(i)] = b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
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
            self.gradient_descent(Y, self.cache, alpha)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, self.__cache['A{}'.format(self.__L)])
                if verbose:
                    print("Cost after {} iterations: {}".
                          format(i, cost))
                if graph:
                    iterations_steps.append(i)
                    costs_steps.append(cost)
        if verbose:
            A, cost = self.evaluate(X, Y)
            print('Cost after {} iterations: {}'.format(iterations, cost))
        if graph:
            plt.plot(iterations_steps, costs_steps, 'b-')
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """
            Saves the instance object to a file in pickle format
                Parameters
                ----------
                filename : str
                    file to which the object should be saved
        """
        if filename[-4:] != ".pkl":
            filename += ".pkl"

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
            Loads a pickled DeepNeuralNetwork object
                Parameters
                ----------
                filename : str
                    file to which the object should be saved
        """

        try:
            with open(filename, 'rb') as f:
                loaded_obj = pickle.load(f)
                return loaded_obj
        except FileNotFoundError:
            return None
