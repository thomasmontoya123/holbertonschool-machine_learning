#!/usr/bin/env python3
"""Normal module"""


class Normal(object):
    """Represents a normal distribution"""

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
