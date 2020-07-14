#!/usr/bin/env python3
"""Matrix concat module """


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis"""

    concat_matrix = []

    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        concat_matrix = [*mat1] + [*mat2]
        return concat_matrix

    elif axis == 1 and len(mat1) == len(mat2):
        for i in range(len(mat1)):
            concat_matrix.append([*mat1[i]] + [*mat2[i]])
        return concat_matrix

    return None
