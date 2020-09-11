#!/usr/bin/env python3
"""Cnn pooling forward propagation module"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs forward propagation over a pooling layer of a neural network

        :param A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer

        :param kernel_shape: tuple of (kh, kw)
        containing the size of the kernel for the pooling

        :param stride: tuple of (sh, sw)
        containing the strides for the pooling

        :param mode: string containing either max or avg, indicating
        whether to perform maximum or average pooling, respectively
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    conv_h = int((h_prev - kh) / sh + 1)
    conv_w = int((w_prev - kw) / sw + 1)

    output = np.zeros((m, conv_h, conv_w, c_prev))

    if mode == 'max':
        pool_mode = np.max

    elif mode == 'avg':
        pool_mode = np.average

    for x in range(conv_h):
        for y in range(conv_w):
            i = x * sh
            j = y * sw
            output[:, x, y, :] = pool_mode(A_prev[:, i: i + kh,
                                           j: j + kw],
                                           axis=(1, 2))

    return output
