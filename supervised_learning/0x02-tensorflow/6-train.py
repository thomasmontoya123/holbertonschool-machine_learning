#!/usr/bin/env python3
"""Train module"""

import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
        builds, trains, and saves a neural network classifier
            Parameters
            ----------
            X_train : numpy.ndarray
                contains the training input data
            Y_train : numpy.ndarray
                contains the training labels
            X_valid : numpy.ndarray
                contains the validation input data
            Y_valid : numpy.ndarray
                contains the validation labels
            layer_sizes : list
                contains the number of nodes in each
                layer of the network
            activations : list
                containins the activation functions for
                each layer of the network
            alpha : float
                learning rate
            iterations : int
                number of iterations to train over
            save_path : str
                designates where to save the model
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    init_op = tf.global_variables_initializer()
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(iterations + 1):
            train_cost, train_accuracy = sess.run([loss, accuracy],
                                                  feed_dict={x: X_train,
                                                             y: Y_train})

            val_cost, val_accuracy = sess.run([loss, accuracy],
                                              feed_dict={x: X_valid,
                                                         y: Y_valid})

            sess.run(train_op, feed_dict={x: X_train, y: Y_train})

            if i % 100 == 0 or i == iterations:
                print('After {} iterations:'.format(i))
                print('\tTraining Cost: {}'.format(train_cost))
                print('\tTraining Accuracy: {}'.format(train_accuracy))
                print('\tValidation Cost: {}'.format(val_cost))
                print('\tValidation Accuracy: {}'.format(val_accuracy))

        path = saver.save(sess, save_path)
    return path
