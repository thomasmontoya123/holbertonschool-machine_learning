#!/usr/bin/env python3
"""all together module """

import numpy as np
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
       creates the training operation for a neural network
       in tensorflow using the Adam optimization algorithm
            Parameters
            ----------

            loss : loss of the network

            alpha : float
                learning rate

            beta1 : float
                weight used for the first moment

            beta2 : float
                weight used for the second moment

            epsilon : float
                small number to avoid division by zero

    """
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon)

    return optimizer.minimize(loss)


def shuffle_data(X, Y):
    """
        Normalizes (standardizes) a matrix
                Parameters
                ----------
                X : numpy.ndarray
                    shape (m, nx) to normalize
                        m is the number of data points
                        nx is the number of features in X
                Y : numpy.ndarray
                    shape (m, ny) to normalize
                        m is the number of data points
                        ny is the number of features in Y
    """
    indices = np.random.permutation(X.shape[0])
    return X[indices], Y[indices]


def calculate_loss(y, y_pred):
    """
        calculates the softmax cross-entropy loss of a prediction
            Parameters
            ----------
            y : tf.placeholder
                labels of the input data
            y_pred : tf.tensor
                network’s predictions
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)


def create_layer(prev, n, activation):
    """
        Creates a layer
            Parameters
            ----------
            prev : tensor
                output of the previous layer
            n : int
                number of nodes in the layer to create
            activation : function
                activation function that the layer should use
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    model = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init, name='layer')

    return model(prev)


def forward_prop(x, layer_sizes=[], activations=[]):
    """
        creates the forward propagation graph for the neural network:
            Parameters
            ----------
            x : tf.placeholder
                input data
            layer_sizes : list
                containis the number of nodes in each
                layer of the network
            activations : list
                Contains the activation functions for each
                layer of the network
    """
    y_pred = create_layer(x, layer_sizes[0], activations[0])

    for i in range(1, len(layer_sizes)):
        y_pred = create_layer(y_pred, layer_sizes[i], activations[i])

    return y_pred


def create_batch_norm_layer(prev, n, activation):
    """
       creates a batch normalization layer for a neural network in tensorflow
            Parameters
            ----------
            prev : activated output of the previous layer

            n :  number of nodes in the layer to be created

            activation : activation function that should be
                used on the output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    epsilon = 1e-8
    model = tf.layers.Dense(units=n,
                            activation=None,
                            kernel_initializer=init,
                            name='layer')

    mean, variance = tf.nn.moments(model(prev), axes=0, keep_dims=True)

    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)

    normaliced = tf.nn.batch_normalization(model(prev),
                                           mean=mean,
                                           variance=variance,
                                           offset=beta,
                                           scale=gamma,
                                           variance_epsilon=epsilon)

    return activation(normaliced)


def create_placeholders(nx, classes):
    """
        returns two placeholders, x and y, for the neural network
            Parameters
            ----------
            nx : int
                number of feature columns in our data
            classes : int
                number of classes in our classifier
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name="x")
    y = tf.placeholder(tf.float32, shape=[None, classes], name="y")

    return x, y


def calculate_accuracy(y, y_pred):
    """
        calculates the accuracy of a prediction
            Parameters
            ----------
            y : tf.placeholder
                labels of the input data
            y_pred : tf.tensor
                network’s predictions
    """
    prediction_y = tf.argmax(y, axis=1)
    prediction_y_pred = tf.argmax(y_pred, axis=1)
    equality = tf.equal(prediction_y, prediction_y_pred)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy


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


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999,
          epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in tensorflow using
    Adam optimization, mini-batch gradient descent, learning rate decay,
    and batch normalization

        :param Data_train: tuple containing the training inputs and training
        labels, respectively

        :param Data_valid: tuple containing the validation inputs and
        validation labels, respectively

        :param layers: list containing the number of nodes in each layer
        of the network

        :param activations: list containing the activation functions used
        for each layer of the network

        :param alpha: learning rate
        :param beta1: weight for the first moment of Adam Optimization
        :param beta2: weight for the second moment of Adam Optimization
        :param epsilon: small number used to avoid division by zero

        :param decay_rate: decay rate for inverse time decay of the
        learning rate (the corresponding decay step should be 1

        :param batch_size: number of data points that should be in a mini-batch
        :param epochs: number of times the training should pass
        through the whole dataset

        :param save_path: path where the model should be saved to

    :return path where the model was saved
    """
    X_train = Data_train[0]
    Y_train = Data_train[1]
    X_valid = Data_valid[0]
    Y_valid = Data_valid[1]

    m = X_train.shape[0]
    batch_len = m // batch_size
    if m % batch_size != 0:
        batch_len += 1

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    global_step = tf.Variable(0, trainable=False)
    alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)

    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

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

                sess.run(global_step.assign(epoch))
                sess.run(alpha)

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
