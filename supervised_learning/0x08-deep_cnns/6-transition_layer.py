#!/usr/bin/env python3
"""Transition layer module"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in Densely
    Connected Convolutional Networks

        :param X: output from the previous layer

        :param nb_filters: integer representing the
            number of filters in X

        :param compression: compression factor for
        the transition layer

        :returns: The output of the transition layer
        and the number of filters within the output, respectively
    """
    He = K.initializers.he_normal()
    nb_filters = int(compression * nb_filters)

    Bn = K.layers.BatchNormalization()(X)
    relu = K.layers.Activation('relu')(Bn)

    C = K.layers.Conv2D(filters=nb_filters,
                        kernel_size=(1, 1),
                        padding='same',
                        kernel_initializer=He)(relu)

    X = K.layers.AveragePooling2D(pool_size=(2, 2),
                                  padding='same')(C)

    return X, nb_filters
