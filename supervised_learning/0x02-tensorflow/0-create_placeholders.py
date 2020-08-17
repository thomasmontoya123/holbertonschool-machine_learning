#!/usr/bin/env python3
"""Placeholder module"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """
        returns two placeholders, x and y, for the neural network
            Parameters
            ----------
            nx : int
                number of feature columns in our data
            classes : int
                number of classes in our classifier
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name="x")
    y = tf.placeholder(tf.float32, shape=[None, classes], name="y")

    return x, y
