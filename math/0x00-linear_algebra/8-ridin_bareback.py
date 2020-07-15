#!/usr/bin/env python3
"""Matrix multiplication module"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication"""

    '''Rule of inner dimension'''
    if len(mat1[0]) != len(mat2):
        return None

    result_matrix = []

    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            partial_result = 0
            for k in range(len(mat1[0])):
                element_mult = mat1[i][k] * mat2[k][j]
                partial_result += element_mult
            row.append(partial_result)
        result_matrix.append(row)

    return result_matrix

