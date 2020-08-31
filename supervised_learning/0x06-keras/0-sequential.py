#!/usr/bin/env python3
"""Keras sequential module"""

from tensorflow import keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library
        :param nx: number of input features to the network
        :param layers: list containing the number of nodes
            in each layer of the network
        :param activations: list containing the activation functions used
            for each layer of the network
        :param lambtha: L2 regularization parameter
        :param keep_prob: probability that a node will be kept for dropout
    """
    model = keras.Sequential()
    L2 = keras.regularizers.l2(l=lambtha)

    model.add(keras.layers.Dense(units=layers[0],
                                 activation=activations[0],
                                 kernel_regularizer=L2,
                                 input_shape=(nx,)))

    for i in range(1, len(layers)):
        model.add(keras.layers.Dropout(rate=1 - keep_prob))

        model.add(keras.layers.Dense(units=layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=L2))

    return model
