#!/usr/bin/env python3
"""RMSProp module"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
       creates the training operation for a neural network
       in tensorflow using the RMSProp optimization algorithm
            Parameters
            ----------
            loss : tf
                loss of the network

            alpha : float
                learning rate

            beta2 : float
                RMSProp weight

            epsilon : float
                small number to avoid division by zero

            var : numpy.ndarray
                contains the variable to be updated

            grad : numpy.ndarray
                contains the gradient of var

            s : is the previous second moment of var
    """
    s = (s * beta2) + ((1 - beta2) * (grad ** 2))
    var = var - ((alpha * grad) / (s ** (1 / 2) + epsilon))
    return var, s
