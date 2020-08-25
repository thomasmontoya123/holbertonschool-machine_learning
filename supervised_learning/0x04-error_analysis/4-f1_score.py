#!/usr/bin/env python3
"""F1 score module"""

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    calculates the F1 score of a confusion matrix
        :param confusion: confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels

        :return: numpy.ndarray of shape (classes,) containing the F1 score
            of each class
    """
    precision_result = precision(confusion)
    sensitivity_result = sensitivity(confusion)

    return 2 * (precision_result * sensitivity_result) /\
        (precision_result + sensitivity_result)
