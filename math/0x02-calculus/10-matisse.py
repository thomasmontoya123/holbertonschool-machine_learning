#!/usr/bin/env python3
"""Polinomial derivative module"""


def poly_derivative(poly):
    """calculates the derivative of a polynomial"""
    if type(poly) != list or len(poly) == 0 or type(poly[0]) not in [int, float]:
        return None

    if len(poly) == 1:
        return [0]

    result = [poly[i] * i for i in range(1, len(poly))]
    return result
