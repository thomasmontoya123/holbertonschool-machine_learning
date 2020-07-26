#!/usr/bin/env python3
"""Sigma module"""


def summation_i_squared(n):
    """Calculates summation i**2 1 to n"""
    if n <= 0 or type(n) != int:
        return None
    return n * (n + 1) * ((2 * n) + 1) // 6
