#!/usr/bin/env python3
"""Matrix sum module"""


def add_matrices2D(mat1, mat2):
    """Adds two matrices element-wise"""

    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    res_matrix = []

    for i in range(len(mat1)):
        res_row = []
        for j in range(len(mat1[0])):
            res_row.append(mat1[i][j] + mat2[i][j])
        res_matrix.append(res_row)

    return res_matrix
