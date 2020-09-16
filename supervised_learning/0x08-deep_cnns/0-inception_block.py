#!/usr/bin/env python3
"""Inception Block module"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in Going Deeper
    with Convolutions (2014)

        :param A_prev: output from the previous layer

        :param filters: tuple or list containing F1, F3R, F3,F5R, F5, FPP

        :return concatenated output of the inception block
    """
    He = K.initializers.he_normal(seed=None)
    F1, F3R, F3, F5R, F5, FPP = filters

    left_conv = K.layers.Conv2D(filters=F1,
                                kernel_size=(1, 1),
                                padding='same',
                                activation='relu',
                                kernel_initializer=He)(A_prev)

    bottom_l = K.layers.Conv2D(filters=F3R,
                               kernel_size=(1, 1),
                               padding='same',
                               activation='relu',
                               kernel_initializer=He)(A_prev)

    top_l = K.layers.Conv2D(filters=F3,
                            kernel_size=(3, 3),
                            padding='same',
                            activation='relu',
                            kernel_initializer=He)(bottom_l)

    bottom_m = K.layers.Conv2D(filters=F5R,
                               kernel_size=(1, 1),
                               padding='same',
                               activation='relu',
                               kernel_initializer=He)(A_prev)

    top_m = K.layers.Conv2D(filters=F5,
                            kernel_size=(5, 5),
                            padding='same',
                            activation='relu',
                            kernel_initializer=He)(bottom_m)

    bottom_r = K.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=(1, 1),
                                  padding='same')(A_prev)

    top_r = K.layers.Conv2D(filters=FPP,
                            kernel_size=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_initializer=He)(bottom_r)

    f_concat = K.layers.concatenate([left_conv, top_l, top_m, top_r])

    return f_concat
