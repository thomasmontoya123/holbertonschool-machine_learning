#!/usr/bin/env python3
"""Moving average module"""

import numpy as np


def moving_average(data, beta):
    """
       calculates the weighted moving average of a data set
            Parameters
            ----------
            data : list
                data to calculate the moving average of

            beta : float
                weight used for the moving average
    """
    mov = []
    vt = 0
    for i, x in enumerate(data):
        vt = beta * vt + (1 - beta) * x
        bias = 1.0 - (beta ** (i + 1))
        mov.append(vt / bias)
    return mov
