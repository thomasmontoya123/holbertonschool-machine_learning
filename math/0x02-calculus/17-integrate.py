#!/usr/bin/env python3
"""Integrale module"""


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial"""
    if type(poly) != list or len(poly) == 0:
        return None

    result = [C]
    for i in range(len(poly)):
        if type(poly[i]) not in [int, float]:
            return None
        coef = poly[i] / (i + 1)
        if coef % 1 == 0:
            coef = int(coef)
        result.append(coef)

    index = len(poly) - 1
    while result[index] == 0:
        result.pop(index)
        index -= 1

    return result
