#!/usr/bin/env python
"""L2 Regularization Cost module"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    calculates the cost of a neural network with L2 regularization
        :param cost: tensor containing the cost of the network without
            L2 regularization

        :return: tensor containing the cost of the network accounting
            for L2 regularization
    """
    return cost + tf.losses.get_regularization_losses()
