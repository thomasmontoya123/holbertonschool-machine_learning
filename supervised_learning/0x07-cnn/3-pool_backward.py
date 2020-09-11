#!/usr/bin/env python3
"""Pooling Back Propagation module"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network

        :param dA: numpy.ndarray of shape (m, h_new, w_new, c_new)
        containing the partial derivatives with respect to the output
        of the pooling layer

        :param A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c)
        containing the output of the previous layer

        :param kernel_shape: tuple of (kh, kw) containing the size of
        the kernel for the pooling

        :param stride: tuple of (sh, sw) containing the strides for the pooling

        :param mode: string containing either max or avg, indicating
        whether to perform maximum or average pooling, respectively

        :returns partial derivatives with respect to the previous
        layer (dA_prev)
    """
    m, h_new, w_new, c_new = dA.shape
    _, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for m in range(m):
        for h in range(h_new):
            for w in range(w_new):
                i = h * sh
                j = w * sw
                for c in range(c_new):
                    frame = A_prev[m, i: i + kh, j: j + kw, c]

                    if mode == 'max':
                        kernel = np.zeros(kernel_shape)
                        frame_max = np.amax(frame)
                        np.copyto(kernel, 1, where=(frame == frame_max))
                        dA_prev[m, i: i + kh, j: j + kw, c] +=\
                            kernel * dA[m, h, w, c]

                    elif mode == 'avg':
                        avg = dA[m, h, w, c] / kh / kw
                        kernel = np.ones(kernel_shape)
                        dA_prev[m, i: i + kh, j: j + kw, c] += avg * kernel

    return dA_prev
