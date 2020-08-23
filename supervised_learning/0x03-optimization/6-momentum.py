#!/usr/bin/env python3
"""Momentum with tf module"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
        creates the training operation for a neural
        network in tensorflow using the gradient descent
        with momentum optimization algorithm
            Parameters
            ----------
            loss : tf
                loss of the network

            alpha : float
                learning rate

            beta1 : float
                momentum weight
    """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)
