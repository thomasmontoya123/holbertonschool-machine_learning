#!/usr/bin/env python3
"""Cnn forward propagation module"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    performs forward propagation over a convolutional layer of a neural network
        :param A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
        :param W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
        containing the kernels for the convolution
        :param b: numpy.ndarray of shape (1, 1, 1, c_new)
        containing the biases applied to the convolution
        :param activation: activation function applied to the convolution
        :param padding: string that is either same or valid
        :param stride: tuple of (sh, sw) containing the strides
        for the convolution
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    ph = 0
    pw = 0

    if padding == 'same':
        ph = int(((h_prev - 1) * sh - h_prev + kh) / 2)
        pw = int(((w_prev - 1) * sw - w_prev + kw) / 2)

    in_padded = np.pad(array=A_prev,
                       pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                       mode='constant')

    conv_h = int((h_prev + 2 * ph - kh) / sh + 1)
    conv_w = int((w_prev + 2 * pw - kw) / sw + 1)

    output = np.zeros((m, conv_h, conv_w, c_new))

    for x in range(conv_h):
        for y in range(conv_w):
            for z in range(c_new):
                i = y * sh
                j = x * sw
                output[:, y, x, z] = (W[:, :, :, z] *
                                      in_padded[:, i: i + kh,
                                      j: j + kw,
                                      :]).sum(axis=(1, 2, 3))

                output[:, y, x, z] = activation(output[:, y, x, z] +
                                                b[0, 0, 0, z])

    return output
