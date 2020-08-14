#!/usr/bin/env python3
"""Accuracy module"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
        calculates the accuracy of a prediction
            Parameters
            ----------
            y : tf.placeholder
                labels of the input data
            y_pred : tf.tensor
                network’s predictions
    """
    prediction_y = tf.argmax(y, axis=1)
    prediction_y_pred = tf.argmax(y_pred, axis=1)
    equality = tf.equal(prediction_y, prediction_y_pred)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    sess = tf.Session()
    return accuracy
