#!/usr/bin/env python3
"""forward propagation module"""

import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
        creates the forward propagation graph for the neural network:
            Parameters
            ----------
            x : tf.placeholder
                input data
            layer_sizes : list
                containis the number of nodes in each
                layer of the network
            activations : list
                Contains the activation functions for each
                layer of the network
    """
    y_pred = create_layer(x, layer_sizes[0], activations[0])

    for i in range(1, len(layer_sizes)):
        y_pred = create_layer(y_pred, layer_sizes[i], activations[i])

    return y_pred
