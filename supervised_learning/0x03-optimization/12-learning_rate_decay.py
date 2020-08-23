#!/usr/bin/env python3
"""Learning rate decay with tf module"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
       creates a learning rate decay operation in tensorflow using
       inverse time decay
            Parameters
            ----------
            alpha : float
                learning rate

            decay_rate : weight used to determine the rate
                        at which alpha will decay

            global_step : number of passes of gradient descent
                        that have elapsed

            decay_step : int
                number of passes of gradient descent that
                should occur before alpha is decayed further
    """
    return tf.train.inverse_time_decay(learning_rate=alpha,
                                       global_step=global_step,
                                       decay_steps=decay_step,
                                       decay_rate=decay_rate,
                                       staircase=True)
