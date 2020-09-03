#!/usr/bin/env python3
"""Convolve with Stride module"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    performs a convolution on grayscale images with custom padding
        :param images: numpy.ndarray with shape (m, h, w, c)
            containing multiple images
        :param kernel: numpy.ndarray with shape (kh, kw, c)
            containing the kernel for the convolution
        :param padding: either a tuple of (ph, pw), ‘same’, or ‘valid
        :param stride: tuple of (sh, sw)
    """
    m, input_w, input_h, input_ch = images.shape
    kernel_w, kernel_h, kernel_ch = kernel.shape
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
                       pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
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
                                         j: j + kernel_w,
                                         :]).sum(axis=(1, 2, 3))

    return output
