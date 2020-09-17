#!/usr/bin/env python3
"""ResNet-50 module"""

import tensorflow.keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015)

        :returns keras model
    """
    He = K.initializers.he_normal()
    inputs = K.Input(shape=(224, 224, 3))

    C0 = K.layers.Conv2D(filters=64,
                         kernel_size=(7, 7),
                         strides=(2, 2),
                         padding='same',
                         kernel_initializer=He)(inputs)

    Bn0 = K.layers.BatchNormalization(axis=3)(C0)
    relu0 = K.layers.Activation('relu')(Bn0)

    P0 = K.layers.MaxPool2D(pool_size=(3, 3),
                            strides=(2, 2),
                            padding='same')(relu0)

    Pb_Ib0 = projection_block(P0, [64, 64, 256], 1)
    for i in range(2):
        Pb_Ib0 = identity_block(Pb_Ib0, [64, 64, 256])

    Pb_Ib1 = projection_block(Pb_Ib0, [128, 128, 512])
    for i in range(3):
        Pb_Ib1 = identity_block(Pb_Ib1, [128, 128, 512])

    Pb_Ib2 = projection_block(Pb_Ib1, [256, 256, 1024])
    for i in range(5):
        Pb_Ib2 = identity_block(Pb_Ib2, [256, 256, 1024])

    Pb_Ib3 = projection_block(Pb_Ib2, [512, 512, 2048])
    for i in range(2):
        Pb_Ib3 = identity_block(Pb_Ib3, [512, 512, 2048])

    P1_avg = K.layers.AveragePooling2D(pool_size=(7, 7),
                                       padding='same')(Pb_Ib3)

    out = K.layers.Dense(units=1000,
                         activation='softmax',
                         kernel_initializer=He)(P1_avg)

    resnet = K.models.Model(inputs=inputs, outputs=out)

    return resnet
