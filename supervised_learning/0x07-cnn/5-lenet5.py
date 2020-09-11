#!/usr/bin/env python3
""" LeNet-5 (Keras)  module """

import tensorflow.keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using keras

        :param X: K.Input of shape (m, 28, 28, 1)
            containing the input images for the network
        
    """

    init = K.initializers.he_normal()
    optimizer = K.optimizers.Adam()

    C1 = K.layers.Conv2D(filters=6,
                         kernel_size=(5, 5),
                         padding='same',
                         activation='relu',
                         kernel_initializer=init)(X)

    S2 = K.layers.MaxPool2D(pool_size=(2, 2),
                            strides=(2, 2))(C1)

    C3 = K.layers.Conv2D(filters=16,
                         kernel_size=(5, 5),
                         activation='relu',
                         kernel_initializer=init)(S2)

    S4 = K.layers.MaxPool2D(pool_size=(2, 2),
                            strides=(2, 2))(C3)

    S4 = K.layers.Flatten()(S4)

    F5 = K.layers.Dense(units=120,
                        activation='relu',
                        kernel_initializer=init)(S4)

    F6 = K.layers.Dense(units=84,
                        activation='relu',
                        kernel_initializer=init)(F5)

    out = K.layers.Dense(units=10,
                         activation='softmax',
                         kernel_initializer=init)(F6)

    model = K.Model(inputs=X, outputs=out)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
