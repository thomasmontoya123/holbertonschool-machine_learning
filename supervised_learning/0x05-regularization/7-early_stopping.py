#!/usr/bin/env python3
"""Early stopping module"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    determines if you should stop gradient descent early
        :param cost: current validation cost of the neural network
        :param opt_cost: lowest recorded validation cost of the neural network
        :param threshold: threshold used for early stopping
        :param patience: patience count used for early stopping
        :param count: count of how long the threshold has not been met
        :returns : boolean of whether the network should be stopped early,
            followed by the updated count
    """
    if threshold < opt_cost - cost:
        return False, 0

    else:
        count += 1
        if patience > count:
            return False, count
        else:
            return True, count
