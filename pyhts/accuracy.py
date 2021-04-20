import numpy as np


def mase(history, y_true, y_pred, m):
    q = np.abs(history[m:] - history[:(len(history) - m)]).mean()
    e = np.abs(y_true - y_pred).mean()
    return e/q


def mse(history, y_true, y_pred, m):
    return np.mean(np.square(y_true-y_pred))