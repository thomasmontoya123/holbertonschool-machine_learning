#!/usr/bin/env python3
"""loss module"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
        calculates the softmax cross-entropy loss of a prediction
            Parameters
            ----------
            y : tf.placeholder
                labels of the input data
            y_pred : tf.tensor
                network’s predictions
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
