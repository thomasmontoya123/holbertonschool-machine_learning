#!/usr/bin/env python3
"""Momentum module"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
        Updates a variable using the gradient descent with
        momentum optimization algorithm
            Parameters
            ----------
            alpha : float
                learning rate

            beta1 : float
                momentum weight

            var : numpy.ndarray
                contains the variable to be updated

            grad : numpy.ndarray
                containing the gradient of var

            v : previous first moment of var
    """
    v = v * beta1 + ((1 - beta1) * grad)
    var = var - (alpha * v)
    return var, v
