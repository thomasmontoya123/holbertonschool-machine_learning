#!/usr/bin/env python3
"""Normal module"""


class Normal(object):
    """Represents a normal distribution"""
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """Constructor"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.stddev = float(stddev)
                self.mean = float(mean)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")

            else:
                self.mean = float(sum(data) / len(data))
                values = [(x - self.mean) ** 2 for x in data]
                self.stddev = (sum(values) / len(data)) ** 0.5

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        pdf = 1 / (self.stddev * (2 * self.pi) ** 0.5) * \
            self.e ** (- (x - self.mean)**2 / (2 * self.stddev**2))
        return pdf

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        val_erf = (x - self.mean) / (self.stddev * 2 ** 0.5)

        erf = erf = (2 / 3.1415926536 ** 0.5) * \
            (val_erf - (val_erf ** 3) / 3 + (val_erf ** 5) / 10 -
                (val_erf ** 7) / 42 + (val_erf ** 9) / 216)

        cdf = (1 / 2) * (1 + erf)
        return cdf
