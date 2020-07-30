#!/usr/bin/env python3
"""Exponential module"""


class Exponential(object):
    """Represents an exponential distribution"""
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """Constructor"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")

            else:
                self.lambtha = 1 / (sum(data) / len(data))
                self.lambtha = float(self.lambtha)

    def pdf(self, x):
        """Calculates the value of the PDF for a given time period"""
        if x < 0:
            return 0
        else:
            pdf = self.lambtha * self.e ** (- self.lambtha * x)
            return pdf

    def cdf(self, x):
        """Calculates the value of the CDF for a given time period"""
        if x < 0:
            return 0

        cdf = 1 - self.e ** (- self.lambtha * x)
        return cdf
