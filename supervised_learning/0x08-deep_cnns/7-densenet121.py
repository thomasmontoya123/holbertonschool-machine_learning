#!/usr/bin/env python3
"""DenseNet 121 module"""

import tensorflow.keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described
    in Densely Connected Convolutional Networks

        :param growth_rate: growth rate

        :param compression: compression factor

        :returns: keras model
    """
    He = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))
    nb_filters = growth_rate * 2

    Bn0 = K.layers.BatchNormalization(axis=3)(X)
    relu0 = K.layers.Activation('relu')(Bn0)

    C0 = K.layers.Conv2D(filters=nb_filters,
                         kernel_size=(7, 7),
                         strides=(2, 2),
                         padding='same',
                         kernel_initializer=He)(relu0)

    P0 = K.layers.MaxPool2D(pool_size=(3, 3),
                            strides=(2, 2),
                            padding='same')(C0)

    block, nb_filters = dense_block(P0, nb_filters, growth_rate, 6)

    for i in [12, 24, 16]:
        block, nb_filters = transition_layer(block, nb_filters, compression)
        block, nb_filters = dense_block(block, nb_filters, growth_rate, i)

    P1_avg = K.layers.AveragePooling2D(pool_size=(7, 7),
                                       padding='same')(block)

    output = K.layers.Dense(units=1000,
                            activation='softmax',
                            kernel_initializer=He)(P1_avg)

    densenet = K.models.Model(inputs=X, outputs=output)

    return densenet
