#!/usr/bin/env python3
"""Integrale module"""


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial"""
    if type(poly) != list or len(poly) == 0 \
            or type(poly[0]) not in [int, float] \
            or type(C) not in [int, float]:
        return None

    result = [poly[i] / (i + 1) for i in range(len(poly))]
    cleaned = [int(i) if i % 1 == 0 else i for i in result]
    cleaned.insert(0, C)

    index = len(poly) - 1
    while cleaned[index] == 0:
        cleaned.pop(index)
        index -= 1

    return cleaned
