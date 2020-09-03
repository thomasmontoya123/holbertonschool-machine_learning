#!/usr/bin/env python3
"""Grayscale convolve with padding module"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    performs a valid convolution on grayscale images
        :param images: numpy.ndarray with shape (m, h, w)
            containing multiple grayscale images
        :param kernel: numpy.ndarray with shape (kh, kw)
            containing the kernel for the convolution
    """
    m = images.shape[0]
    input_w, input_h = images.shape[2], images.shape[1]
    kernel_w, kernel_h = kernel.shape[1], kernel.shape[0]

    pad_w = int(kernel_w - 1 / 2)
    pad_h = int(kernel_w - 1 / 2)

    in_padded = np.pad(array=images,
                       pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                       mode='constant')

    output = np.zeros((m, input_h, input_w))

    for y in range(input_h):
        for x in range(input_w):
            output[:, y, x] = \
                (kernel * in_padded[:,
                                    y: y + kernel_h,
                                    x: x + kernel_w]).sum(axis=(1, 2))

    return output
