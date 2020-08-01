#!/usr/bin/env python3

import numpy as np

Neuron = __import__('3-neuron').Neuron

lib_train = np.load('/Users/thomas/PycharmProjects/holbertonschool-machine_learning/supervised_learning/0x00-binary_classification/data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
A = neuron.forward_prop(X)
cost = neuron.cost(Y, A)

print(cost)