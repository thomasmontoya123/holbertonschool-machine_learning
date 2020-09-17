#!/usr/bin/env python3
"""Dense block module"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    builds a dense block as described in Densely Connected
    Convolutional Networks

        :param X: output from the previous layer

        :param nb_filters: integer representing the number
            of filters in X

        :param growth_rate: growth rate for the dense block

        :param layers: number of layers in the dense block
    """
    He = K.initializers.he_normal()

    for i in range(layers):
        Bn0 = K.layers.BatchNormalization()(X)
        relu0 = K.layers.Activation('relu')(Bn0)
        C0 = K.layers.Conv2D(filters=growth_rate * 4,
                             kernel_size=(1, 1),
                             padding='same',
                             kernel_initializer=He)(relu0)

        Bn1 = K.layers.BatchNormalization()(C0)
        relu1 = K.layers.Activation('relu')(Bn1)
        C1 = K.layers.Conv2D(filters=growth_rate,
                             kernel_size=(3, 3),
                             padding='same',
                             kernel_initializer=He)(relu1)

        nb_filters += growth_rate
        X = K.layers.concatenate([X, C1])

    return X, nb_filters
