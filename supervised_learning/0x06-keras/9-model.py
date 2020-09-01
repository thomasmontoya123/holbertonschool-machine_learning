#!/usr/bin/env python3
"""Load and save model module"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    saves an entire model
    :param network: model to save
    :param filename: path of the file that the model should be saved to
    """
    network.save(filename)


def load_model(filename):
    """
     loads an entire model
    :param filename: path of the file that the
        model should be loaded from
    """
    return K.models.load_model(filename)
