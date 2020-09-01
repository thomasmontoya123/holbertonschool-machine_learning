#!/usr/bin/env python3
"""Train module"""

def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
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
    """
    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle)
