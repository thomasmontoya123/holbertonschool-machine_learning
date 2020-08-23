#!/usr/bin/env python3
"""Adam module"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
       updates a variable in place using the Adam optimization algorithm
            Parameters
            ----------

            alpha : float
                learning rate

            beta1 : float
                weight used for the first moment

            beta2 : float
                weight used for the second moment

            epsilon : float
                small number to avoid division by zero

            var : numpy.ndarray
                contains the variable to be updated

            grad : numpy.ndarray
                contains the gradient of var

            v : previous first moment of var

            s : is the previous second moment of var

            t :  time step used for bias correction
    """

    v = (v * beta1) + ((1 - beta1) * grad)
    v_hat = v / (1 - (beta1 ** t))
    s = (s * beta2) + ((1 - beta2) * (grad ** 2))
    s_hat = s / (1 - (beta2 ** t))
    var = var - ((alpha * v_hat) / (s_hat ** (1 / 2) + epsilon))
    return var, v, s
