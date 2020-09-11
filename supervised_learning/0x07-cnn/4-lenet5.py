#!/usr/bin/env python3
"""Lenet-5 cnn module"""

import tensorflow as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow

        :param x: tf.placeholder of shape (m, 28, 28, 1)
        containing the input images for the network

        :param y: tf.placeholder of shape (m, 10)
        containing the one-hot labels for the network

        :returns Returns: a tensor for the softmax activated output
        a training operation that utilizes Adam optimization
        (with default hyperparameters)
        a tensor for the loss of the netowrk
        a tensor for the accuracy of the network
    """
    init = tf.contrib.layers.variance_scaling_initializer()

    C1 = tf.layers.Conv2D(filters=6,
                          kernel_size=(5, 5),
                          padding='same',
                          activation=tf.nn.relu,
                          kernel_initializer=init)(x)

    S2 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))(C1)

    C3 = tf.layers.Conv2D(filters=16,
                          kernel_size=(5, 5),
                          activation=tf.nn.relu,
                          kernel_initializer=init)(S2)

    S4 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))(C3)

    S4 = tf.layers.Flatten()(S4)

    F5 = tf.layers.Dense(units=120,
                         activation=tf.nn.relu,
                         kernel_initializer=init)(S4)

    F6 = tf.layers.Dense(units=84,
                         activation=tf.nn.relu,
                         kernel_initializer=init)(F5)

    out = tf.layers.Dense(units=10,
                          kernel_initializer=init)(F6)

    y_hat = tf.nn.softmax(out)
    y_hat_t = tf.argmax(y_hat, 1)
    y_t = tf.argmax(y, 1)
    loss = tf.losses.softmax_cross_entropy(y, out)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    comparison = tf.equal(y_hat_t, y_t)
    acc = tf.reduce_mean(tf.cast(comparison, tf.float32))

    return y_hat, train_op, loss, acc
