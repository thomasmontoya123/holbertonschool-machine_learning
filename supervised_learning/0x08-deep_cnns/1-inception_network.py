#!/usr/bin/env python3
"""Inception Network  module"""

import tensorflow.keras as K

inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the inception network as described
    in Going Deeper with Convolutions (2014)

        :returns the keras model
    """
    He = K.initializers.he_normal()

    input = K.Input(shape=(224, 224, 3))

    C0 = K.layers.Conv2D(filters=64,
                         kernel_size=(7, 7),
                         strides=(2, 2),
                         padding='same',
                         activation='relu',
                         kernel_initializer=He)(input)

    P0 = K.layers.MaxPool2D(pool_size=(3, 3),
                            strides=(2, 2),
                            padding='same')(C0)

    C1 = K.layers.Conv2D(filters=64,
                         kernel_size=(1, 1),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         kernel_initializer=He)(P0)

    C2 = K.layers.Conv2D(filters=192,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         kernel_initializer=He)(C1)

    P1 = K.layers.MaxPool2D(pool_size=(3, 3),
                            strides=(2, 2),
                            padding='same')(C2)

    inception_3a = inception_block(P1, [64, 96, 128, 16, 32, 32])
    inception_3b = inception_block(inception_3a, [128, 128, 192, 32, 96, 64])

    P2 = K.layers.MaxPool2D(pool_size=(3, 3),
                            strides=(2, 2),
                            padding='same')(inception_3b)

    inception_4a = inception_block(P2, [192, 96, 208, 16, 48, 64])
    inception_4b = inception_block(inception_4a, [160, 112, 224, 24, 64, 64])
    inception_4c = inception_block(inception_4b, [128, 128, 256, 24, 64, 64])
    inception_4d = inception_block(inception_4c, [112, 144, 288, 32, 64, 64])
    inception_4c = inception_block(inception_4d, [256, 160, 320, 32, 128, 128])

    P3 = K.layers.MaxPool2D(pool_size=(3, 3),
                            strides=(2, 2),
                            padding='same')(inception_4c)

    inception_5a = inception_block(P3, [256, 160, 320, 32, 128, 128])
    inception_5b = inception_block(inception_5a, [384, 192, 384, 48, 128, 128])

    P4_avg = K.layers.AveragePooling2D(pool_size=(7, 7),
                                       padding='same')(inception_5b)

    drop = K.layers.Dropout(rate=0.4)(P4_avg)

    output = K.layers.Dense(units=1000,
                            activation='softmax',
                            kernel_initializer=He)(drop)

    cnn = K.models.Model(inputs=input, outputs=output)

    return cnn
