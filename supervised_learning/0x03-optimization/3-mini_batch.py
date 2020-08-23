#!/usr/bin/env python3
"""mini batch  module"""

import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def model_loader(path):
    """
        model load
            Parameters
            ----------
            path : str
                model to load
    """
    sess = tf.Session()

    saver = tf.train.import_meta_graph(path + ".meta")
    saver.restore(sess, path)
    x = tf.get_collection("x")[0]
    y = tf.get_collection("y")[0]
    accuracy = tf.get_collection("accuracy")[0]
    loss = tf.get_collection("loss")[0]
    train_op = tf.get_collection("train_op")[0]
    return x, y, accuracy, loss, train_op, saver, sess


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
        Trains a loaded neural network model using mini-batch gradient descent:
            Parameters
            ----------
            X_train : numpy.ndarray
                shape (m, 784) containing the training data
                m is the number of data points
                784 is the number of input features

            Y_train : numpy.ndarray
                one-hot numpy.ndarray
                shape (m, 10) containing the training labels

            X_valid : numpy.ndarray
                shape (m, 784) containing the validation data

            Y_valid : one-hot numpy.ndarray
                shape (m, 10) containing the validation labels

            batch_size : int
                number of data points in a batch

            epochs : int
                number of times the training should pass through
                the whole dataset

            load_path : str
                path from which to load the model

            save_path : str
                path to where the model should be saved after training
    """
    x, y, accuracy, loss, train_op, saver, sess = model_loader(load_path)

    m = X_train.shape[0]
    batch_len = m // batch_size
    if m % batch_size != 0:
        batch_len += 1

    with sess:
        for epoch in range(epochs + 1):
            train_cost, train_accuracy = sess.run([loss, accuracy],
                                                  feed_dict={x: X_train,
                                                             y: Y_train})

            val_cost, val_accuracy = sess.run([loss, accuracy],
                                              feed_dict={x: X_valid,
                                                         y: Y_valid})

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(val_cost))
            print("\tValidation Accuracy: {}".format(val_accuracy))

            if epoch < epochs:
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

                for i in range(batch_len):
                    start = i * batch_size
                    if i * batch_size < m:
                        stop = i * batch_size + batch_size
                    else:
                        stop = m
                    X_batch = X_shuffled[start:stop]
                    Y_batch = Y_shuffled[start:stop]
                    sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                    if i != 0 and (i + 1) % 100 == 0:
                        batch_loss = sess.run(loss, feed_dict={x: X_batch,
                                                               y: Y_batch})

                        batch_accuracy = sess.run(accuracy,
                                                  feed_dict={x: X_batch,
                                                             y: Y_batch})

                        print("\tStep {}:".format(i + 1))
                        print("\t\tCost: {}".format(batch_loss))
                        print("\t\tAccuracy: {}".format(batch_accuracy))

        save_path = saver.save(sess, save_path)
        return save_path
