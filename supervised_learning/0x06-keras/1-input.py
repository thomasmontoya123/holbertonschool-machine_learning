#!/usr/bin/env python3
"""Keras input module"""

import tensorflow.keras as K


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
    x = K.Input(shape=(nx,))
    L2 = K.regularizers.L1L2(l2=lambtha)

    y = K.layers.Dense(units=layers[0],
                       activation=activations[0],
                       kernel_regularizer=L2,
                       input_shape=(nx,))(x)

    for i in range(1, len(layers)):
        y = K.layers.Dropout(rate=1 - keep_prob)(y)
        y = K.layers.Dense(units=layers[i],
                           activation=activations[i],
                           kernel_regularizer=L2, )(y)

    model = K.Model(x, y)
    return model
