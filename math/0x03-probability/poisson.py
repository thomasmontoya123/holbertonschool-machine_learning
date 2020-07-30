#!/usr/bin/env python3
"""Poisson module"""


class Poisson(object):
    """Represents a poisson distribution"""
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
                self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""
        if k < 0:
            return 0

        k = int(k)
        fact = 1
        for i in range(1, k + 1):
            fact = fact * i

        pmf = ((self.e ** (- self.lambtha)) * (self.lambtha ** k)) / fact
        return pmf

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of successes"""
        if k < 0:
            return 0
        k = int(k)
        summ = 0
        for i in range(k + 1):
            factorial = 1
            for x in range(1, i + 1):
                factorial *= x
            summ += (self.lambtha ** i) / factorial
        poisson_cdf = (self.e ** (-self.lambtha)) * summ
        return poisson_cdf
