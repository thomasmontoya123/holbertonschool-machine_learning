#!/usr/bin/env python3
"""Matrix transpose module"""


def matrix_transpose(matrix):
    """Transpose a matrix"""

    transposed = []

    for i in range(len(matrix[0])):
        transposed_row = []
        for j in range(len(matrix)):
            transposed_row.append(matrix[j][i])
        transposed.append(transposed_row)

    return transposed
