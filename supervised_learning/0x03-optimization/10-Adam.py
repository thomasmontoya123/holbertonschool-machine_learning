#!/usr/bin/env python3
"""Adam with tf module"""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
       creates the training operation for a neural network
       in tensorflow using the Adam optimization algorithm
            Parameters
            ----------

            loss : loss of the network

            alpha : float
                learning rate

            beta1 : float
                weight used for the first moment

            beta2 : float
                weight used for the second moment

            epsilon : float
                small number to avoid division by zero

    """
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon)

    return optimizer.minimize(loss)
