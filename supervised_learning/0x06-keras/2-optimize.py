#!/usr/bin/env python3
"""Optimize module"""

from tensorflow import keras


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a keras model with categorical
    crossentropy loss and accuracy metrics:
    :param network: model to optimize
    :param alpha: learning rate
    :param beta1: first Adam optimization parameter
    :param beta2: Second Adam optimization parameter
    """
    optimizer = keras.optimizers.Adam(lr=alpha,
                                      beta_1=beta1,
                                      beta_2=beta2)

    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
