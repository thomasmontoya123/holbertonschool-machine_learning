#!/usr/bin/env python3
"""Inception Network  module"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in Deep Residual
    Learning for Image Recognition (2015)

        :param A_prev: output from the previous layer

        :param filters: tuple or list containing F11, F3, F12

        :returns activated output of the identity block
    """
    He = K.initializers.he_normal()
    F11, F3, F12 = filters

    C0 = K.layers.Conv2D(filters=F11,
                         kernel_size=(1, 1),
                         padding='same',
                         kernel_initializer=He)(A_prev)

    batch_norm0 = K.layers.BatchNormalization(axis=3)(C0)
    relu0 = K.layers.Activation('relu')(batch_norm0)

    C1 = K.layers.Conv2D(filters=F3,
                         kernel_size=(3, 3),
                         padding='same',
                         kernel_initializer=He)(relu0)

    batch_norm1 = K.layers.BatchNormalization(axis=3)(C1)
    relu1 = K.layers.Activation('relu')(batch_norm1)

    C2 = K.layers.Conv2D(filters=F12,
                         kernel_size=(1, 1),
                         padding='same',
                         kernel_initializer=He)(relu1)

    batch_norm2 = K.layers.BatchNormalization(axis=3)(C2)
    input_add_conv = K.layers.Add()([batch_norm2, A_prev])
    output = K.layers.Activation('relu')(input_add_conv)

    return output
