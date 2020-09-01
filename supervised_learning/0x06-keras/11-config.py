#!/usr/bin/env python3
"""Load and save model config module"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    saves a model’s configuration in JSON format
    :param network: model whose configuration should be saved
    :param filename: path of the file that the configuration should be saved to
    """
    with open(filename, 'w') as f:
        f.write(network.to_json())


def load_config(filename):
    """
    loads a model with a specific configuration
    :param filename: path of the file containing
        the model’s configuration in JSON format
    """
    with open(filename, 'r') as f:
        return K.models.model_from_json(f.read())
