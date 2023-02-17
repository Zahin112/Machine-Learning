"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""
import numpy as np


def accuracy(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    # tp = np.sum(y_pred == y_true)
    # print(tp)
    return np.sum(y_pred == y_true)/ len(y_true)


def precision_score(tp, fp):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    return tp/(tp+fp)


def recall_score(tp, fn):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    return tp/(tp+fn)


def f1_score(prec, recall):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    return 2*(prec * recall)/(prec+recall)
