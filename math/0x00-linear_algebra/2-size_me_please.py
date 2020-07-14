#!/usr/bin/env python3
"""Matrix shape calculation module"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix """

    if not matrix:
        return []

    shape = [len(matrix)]
    while type(matrix[0]) == type(matrix):
        shape.append(len(matrix[0]))
        matrix = matrix[0]

    return shape
