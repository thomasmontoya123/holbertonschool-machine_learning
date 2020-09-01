#!/usr/bin/env python3
"""Train module"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent
        :param network: model to train
        :param data: numpy.ndarray of shape (m, nx) containing the input data
        :param labels: one-hot numpy.ndarray of shape (m, classes)
            containing the labels of data
        :param batch_size: size of the batch used for
            mini-batch gradient descent
        :param epochs: number of passes through data for mini-batch
            gradient descent
        :param verbose: boolean that determines if output should be printed
            during training
        :param shuffle: boolean that determines whether to shuffle the
            batches every epoch
        :param validation_data : data to validate the model with
        :param early_stopping: boolean that indicates whether
            early stopping should be used
        :param patience: patience used for early stopping
    """
    callbacks = []
    if early_stopping and validation_data:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience)

        callbacks.append(early_stop)

    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
