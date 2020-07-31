#!/usr/bin/env python3
"""Binomial module"""


class Binomial(object):
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Constructor"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            elif (p <= 0) or (p >= 1):
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.n = int(n)
                self.p = float(p)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")

            else:
                mean = float(sum(data) / len(data))
                values = [(x - mean) ** 2 for x in data]
                variance = sum(values) / len(data)
                p = 1 - variance / mean
                if ((mean / p) - (mean // p)) >= 0.5:
                    self.n = 1 + int(mean / p)
                else:
                    self.n = int(mean / p)
                self.p = float(mean / self.n)
