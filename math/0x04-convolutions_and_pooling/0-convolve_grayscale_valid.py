#!/usr/bin/env python3
"""Grayscale convolve module"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
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

    output_w = int(input_w - kernel_w + 1)
    output_h = int(input_h - kernel_h + 1)
    output = np.zeros((m, output_h, output_w))

    for x in range(output_w):
        for y in range(output_h):
            output[:, y, x] =\
                (kernel * images[:,
                                 y: y + kernel_h,
                                 x: x + kernel_w]).sum(axis=(1, 2))

    return output
