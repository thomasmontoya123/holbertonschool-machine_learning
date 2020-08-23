#!/usr/bin/env python3
"""Learning rate decay module"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
       updates the learning rate using inverse time decay
            Parameters
            ----------
            alpha : float
                learning rate

            decay_rate : weight used to determine the rate
                        at which alpha will decay

            decay_step : int
                number of passes of gradient descent that
                should occur before alpha is decayed further
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))
