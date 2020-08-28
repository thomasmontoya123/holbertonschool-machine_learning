#!/usr/bin/env python3
"""Droput layer with tf module"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    creates a layer of a neural network using dropout
        :param prev: tensor containing the output of the previous layer
        :param n: number of nodes the new layer should contain
        :param activation: activation function that should be used on the layer
        :param keep_prob: probability that a node will be kept
    """
    initializer = tf.contrib.layers \
        .variance_scaling_initializer(mode="FAN_AVG")

    model = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=initializer,
                            name='layer')

    dropout = tf.layers.Dropout(rate=keep_prob)
    return dropout(model(prev))
