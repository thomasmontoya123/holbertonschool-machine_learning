#!/usr/bin/env python3
"""Contains the train function"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """
        evaluates the output of a neural network
            Parameters
            ----------
            X : numpy.ndarray
                containing the input data to evaluate
            Y : numpy.ndarray
                containing the one-hot labels for X
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sess, save_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        prediction_ = sess.run(y_pred, feed_dict={x: X, y: Y})

        accuracy = tf.get_collection("accuracy")[0]
        accuracy_ = sess.run(accuracy, feed_dict={x: X, y: Y})

        loss = tf.get_collection("loss")[0]
        cost_ = sess.run(loss, feed_dict={x: X, y: Y})

    return prediction_, accuracy_, cost_
