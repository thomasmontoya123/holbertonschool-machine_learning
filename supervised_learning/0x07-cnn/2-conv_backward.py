#!/usr/bin/env python3
"""Conv back propagation module"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    performs back propagation over a convolutional
    layer of a neural network

        :param dZ: numpy.ndarray of shape (m, h_new, w_new, c_new)
        containing the partial derivatives with respect to the
        unactivated output of the convolutional layer

        :param A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer

        :param W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
        containing the kernels for the convolution

        :param b: numpy.ndarray of shape (1, 1, 1, c_new) containing
        the biases applied to the convolution

        :param padding: string that is either same or valid

        :param stride: tuple of (sh, sw) containing the strides
        for the convolution

        :return partial derivatives with respect to the previous
        layer (dA_prev), the kernels (dW), and the biases (db),
        respectively
    """
    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    ph = 0
    pw = 0

    if padding == 'same':
        ph = int(((h_prev - 1) * sh - h_prev + kh) / 2) + 1
        pw = int(((w_prev - 1) * sw - w_prev + kw) / 2) + 1

    in_padded = np.pad(array=A_prev,
                       pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                       mode='constant')

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    dA = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)

    for m in range(m):
        for h in range(h_new):
            for w in range(w_new):
                i = h * sh
                j = w * sw
                for c in range(c_new):
                    dA[m, i: i + kh, j: j + kw, :] +=\
                        dZ[m, h, w, c] * W[:, :, :, c]

                    dW[:, :, :, c] +=\
                        A_prev[m, i: i + kh, j: j + kw, :] * dZ[m, h, w, c]

    dA = dA[:, ph:dA.shape[1] - ph, pw:dA.shape[2] - pw, :]
    return dA, dW, db
