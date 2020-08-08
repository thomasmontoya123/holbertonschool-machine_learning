#!/usr/bin/env python3

import numpy as np

Deep = __import__('17-deep_neural_network').DeepNeuralNetwork

lib_train = np.load('/Users/thomas/PycharmProjects/holbertonschool-machine_learning/supervised_learning/0x00'
                    '-binary_classification/data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X.shape[0], [5, 3, 1])
print(deep.cache)
print(deep.weights)
print(deep.L)
deep.L = 10
print(deep.L)