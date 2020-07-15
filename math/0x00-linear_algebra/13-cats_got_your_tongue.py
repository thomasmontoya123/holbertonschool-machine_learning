#!/usr/bin/env python3
"""Matrix concat module using np"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis"""
    result = np.concatenate((mat1, mat2), axis=axis)
    return result
