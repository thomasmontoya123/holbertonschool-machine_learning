#!/usr/bin/env python3
"""Train op module"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
        creates the training operation for the network
            Parameters
            ----------
            loss : tf.tensor
                loss of the network’s prediction
            alpha : float
                learning rate
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)

    return train
