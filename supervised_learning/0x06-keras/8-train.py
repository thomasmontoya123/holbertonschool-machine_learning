#!/usr/bin/env python3
"""Train module"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None, verbose=True,
                shuffle=False):
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
        :param learning_rate_decay boolean that indicates whether
            learning rate decay should be used
        :param alpha: initial learning rate
        :param save_best: boolean indicating whether to save the model
            after each epoch if it is the best
        :param filepath where the model should be saved
    """
    callbacks = []

    # Scheduler encapsulation
    def lr_scheduler(epoch):
        """ updates the learning rate using inverse time decay"""
        return alpha / (1 + decay_rate * epoch)

    if save_best:
        checkpoint = K.callbacks.ModelCheckpoint(filepath,
                                                 monitor='val_loss',
                                                 save_weights_only=save_best,
                                                 mode='min')

        callbacks.append(checkpoint)

    if validation_data and learning_rate_decay:
        decay_ck = K.callbacks.LearningRateScheduler(schedule=lr_scheduler,
                                                     verbose=True)

        callbacks.append(decay_ck)

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
