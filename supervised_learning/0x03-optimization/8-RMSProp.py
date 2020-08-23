#!/usr/bin/env python3
"""RMSProp with tf module"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
       creates the training operation for a neural network
       in tensorflow using the RMSProp optimization algorithm
            Parameters
            ----------
            loss : tf
                loss of the network

            alpha : float
                learning rate

            beta2 : float
                RMSProp weight

            epsilon : float
                small number to avoid division by zero
    """
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2,
                                          epsilon=epsilon)
    return optimizer.minimize(loss)
