#!/usr/bin/env python3
"""Transfer learning app module"""

import tensorflow.keras as K
import tensorflow as tf
import datetime


def preprocess_data(X, Y):
    """
    Pre-processes the data for the model

        :param X: numpy.ndarray of shape (m, 32, 32, 3)
            containing the CIFAR 10 data, where m is the
            number of data points

        :param Y: numpy.ndarray of shape (m,) containing
            the CIFAR 10 labels for X

        :returns: X_p, Y_p
    """
    X_p = K.applications.densenet.preprocess_input(X)

    # encode to one-hot
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    CALLBACKS = []
    MODEL_PATH = 'cifar10.h5'
    optimizer = K.optimizers.Adam()

    # load cifar 10
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    # pre-procces data
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # input tensor
    inputs = K.Input(shape=(32, 32, 3))

    # upscale layer resize with pad to avoid distortion
    upscale = \
        K.layers.Lambda(lambda x: tf.image.resize_image_with_pad(x,
                                                                 160,
                                                                 160))(inputs)

    # load base model
    base_model = K.applications.DenseNet121(include_top=False,
                                            weights='imagenet',
                                            input_tensor=upscale,
                                            input_shape=(160, 160, 3),
                                            pooling='max')

    # freeze layers to avoid destroying any of the information they
    # contain during future training rounds
    # for i in base_model.layers[:200]:
    #  i.trainable = False
    # base_model.trainable = False

    # add top layers
    out = base_model.output
    out = K.layers.Flatten()(out)
    out = K.layers.BatchNormalization()(out)
    out = K.layers.Dense(256, activation='relu')(out)
    out = K.layers.Dropout(0.3)(out)
    out = K.layers.BatchNormalization()(out)
    out = K.layers.Dense(128, activation='relu')(out)
    out = K.layers.Dropout(0.3)(out)
    out = K.layers.BatchNormalization()(out)
    out = K.layers.Dense(64, activation='relu')(out)
    out = K.layers.Dropout(0.3)(out)
    out = K.layers.Dense(10, activation='softmax')(out)

    # callbacks
    CALLBACKS.append(K.callbacks.ModelCheckpoint(filepath=MODEL_PATH,
                                                 monitor='val_acc',
                                                 save_best_only=True))

    CALLBACKS.append(K.callbacks.EarlyStopping(monitor='val_acc',
                                               verbose=1,
                                               patience=5))

    CALLBACKS.append(K.callbacks.TensorBoard(log_dir='logs'))

    # model compile
    model = K.models.Model(inputs=inputs, outputs=out)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # train
    model.fit(x=x_train,
              y=y_train,
              batch_size=64,
              epochs=20,
              callbacks=CALLBACKS,
              validation_data=(x_test, y_test))
