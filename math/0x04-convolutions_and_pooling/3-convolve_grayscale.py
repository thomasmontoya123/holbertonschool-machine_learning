#!/usr/bin/env python3
"""Convolve with Stride module"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    performs a convolution on grayscale images with custom padding
        :param images: numpy.ndarray with shape (m, h, w)
            containing multiple grayscale images
        :param kernel: numpy.ndarray with shape (kh, kw)
            containing the kernel for the convolution
        :param padding: either a tuple of (ph, pw), ‘same’, or ‘valid
        :param stride: tuple of (sh, sw)
    """
    m = images.shape[0]
    input_w, input_h = images.shape[2], images.shape[1]
    kernel_w, kernel_h = kernel.shape[1], kernel.shape[0]
    stride_h, stride_w = stride

    if padding == 'same':
        ph = int(((input_h - 1) * stride_h - input_h + kernel_h) / 2) + 1
        pw = int(((input_w - 1) * stride_w - input_w + kernel_w) / 2) + 1

    elif type(padding) == tuple:
        ph, pw = padding

    else:
        ph = 0
        pw = 0

    in_padded = np.pad(array=images,
                       pad_width=((0, 0), (ph, ph), (pw, pw)),
                       mode='constant')

    conv_h = int((input_h + 2 * ph - kernel_h) / stride_h + 1)
    conv_w = int((input_w + 2 * pw - kernel_w) / stride_w + 1)

    output = np.zeros((m, conv_h, conv_w))

    for x in range(conv_w):
        for y in range(conv_h):
            i = y * stride_h
            j = x * stride_w
            output[:, y, x] = (kernel *
                               in_padded[:, i: i + kernel_h,
                                         j: j + kernel_w]).sum(axis=(1, 2))

    return output
