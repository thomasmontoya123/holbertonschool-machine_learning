#!/usr/bin/env python3
"""Layer module"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
        Creates a layer
            Parameters
            ----------
            prev : tensor
                output of the previous layer
            n : int
                number of nodes in the layer to create
            activation : function
                activation function that the layer should use
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    model = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init, name='layer')

    return model(prev)
