#!/usr/bin/env python3
"""Convolve with padding module"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    performs a convolution on grayscale images with custom padding
        :param images: numpy.ndarray with shape (m, h, w)
            containing multiple grayscale images
        :param kernel: numpy.ndarray with shape (kh, kw)
            containing the kernel for the convolution
        :param padding is a tuple of (ph, pw)
    """
    m = images.shape[0]
    input_w, input_h = images.shape[2], images.shape[1]
    kernel_w, kernel_h = kernel.shape[1], kernel.shape[0]

    ph = padding[0]
    pw = padding[1]

    in_padded = np.pad(array=images,
                       pad_width=((0, 0), (ph, ph), (pw, pw)),
                       mode='constant')

    pad_h = int(input_h + 2 * ph - kernel_h + 1)
    pad_w = int(input_w + 2 * pw - kernel_w + 1)

    output = np.zeros((m, pad_h, pad_w))

    for x in range(pad_w):
        for y in range(pad_h):
            output[:, y, x] = \
                (kernel * in_padded[:,
                                    y: y + kernel_h,
                                    x: x + kernel_w]).sum(axis=(1, 2))

    return output
