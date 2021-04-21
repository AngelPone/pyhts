import numpy as np


def mase(history, y_true, y_pred, m):
    q = np.abs(history[m:] - history[:(len(history) - m)]).mean()
    e = np.abs(y_true - y_pred).mean()
    return e/q


def mse(history, y_true, y_pred, m):
    return np.mean(np.square(y_true-y_pred))


def mape(history, y_true, y_pred, m):
    return np.mean(np.abs((y_true-y_pred)/y_true))


def rmse(history, y_true, y_pred, m):
    return np.sqrt(mse(history, y_true, y_pred, m))


def mae(history, y_true, y_pred, m):
    return np.mean(np.abs(y_true - y_pred))


def smape(history, y_true, y_pred, m):
    return 2 * np.mean((y_true-y_pred)/(np.abs(y_true)+np.abs(y_pred)))


def rmsse(history, y_true, y_pred, m):
    q = np.square((history[m:] - history[:(len(history) - m)]).mean())
    e = np.square((y_true-y_pred).mean())
    return np.sqrt(e/q)