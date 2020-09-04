#!/usr/bin/env python3
"""Pooling module"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    performs pooling on images
        :param images: numpy.ndarray with shape (m, h, w, c)
        containing multiple images
        :param kernel_shape: tuple of (kh, kw)
        Containing the kernel shape for the pooling
        :param stride: tuple of (sh, sw)
        :param mode: indicates the type of pooling
    """
    m, input_h, input_w, input_c = images.shape
    kernel_h, kernel_w = kernel_shape
    stride_h, stride_w = stride

    conv_h = int((input_h - kernel_h) / stride_h + 1)
    conv_w = int((input_w - kernel_w) / stride_w + 1)

    output = np.zeros((m, conv_h, conv_w, input_c))

    if mode == 'max':
        pool_mode = np.max

    elif mode == 'avg':
        pool_mode = np.average

    for x in range(conv_h):
        for y in range(conv_w):
            i = x * stride_h
            j = y * stride_w
            output[:, x, y, :] = pool_mode(images[:, i: i + kernel_h,
                                                  j: j + kernel_w],
                                           axis=(1, 2))

    return output
