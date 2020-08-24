#!/usr/bin/env python3
"""Batch normalization with tf module"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
       creates a batch normalization layer for a neural network in tensorflow
            Parameters
            ----------
            prev : activated output of the previous layer

            n :  number of nodes in the layer to be created

            activation : activation function that should be
                used on the output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    epsilon = 1e-8
    model = tf.layers.Dense(units=n,
                            activation=None,
                            kernel_initializer=init,
                            name='layer')

    mean, variance = tf.nn.moments(model(prev), axes=0, keep_dims=True)

    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)

    normaliced = tf.nn.batch_normalization(model(prev),
                                           mean=mean,
                                           variance=variance,
                                           offset=beta,
                                           scale=gamma,
                                           variance_epsilon=epsilon)

    return activation(normaliced)
