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

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""
        if k < 0:
            return 0
        k = int(k)

        factorial_k = 1
        factorial_n = 1
        factorial_n_k = 1

        for x in range(1, k + 1):
            factorial_k *= x

        for x in range(1, self.n + 1):
            factorial_n *= x

        for x in range(1, self.n - k + 1):
            factorial_n_k *= x

        pmf = (factorial_n / (factorial_k * factorial_n_k)) * \
            self.p ** k * (1 - self.p) ** (self.n - k)

        return pmf

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of successes"""
        if k < 0:
            return 0
        k = int(k)
        cdf = 0
        for x in range(k + 1):
            cdf += self.pmf(x)

        return cdf
