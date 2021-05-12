import numpy as np


def mase(y_true, y_pred, history, m):
    q = np.abs(history[m:] - history[:(len(history) - m)]).mean()
    e = np.abs(y_true - y_pred).mean()
    return e/q


def mse(y_true, y_pred, history, m):
    return np.mean(np.square(y_true-y_pred))


def mape(y_true, y_pred, history, m):
    return np.mean(np.abs((y_true-y_pred)/y_true))


def rmse(y_true, y_pred, history, m):
    return np.sqrt(mse(y_true, y_pred, history, m))


def mae(y_true, y_pred, history, m):
    return np.mean(np.abs(y_true - y_pred))


def smape(y_true, y_pred, history, m):
    return 2 * np.mean((y_true-y_pred)/(np.abs(y_true)+np.abs(y_pred)))


def rmsse(y_true, y_pred, history, m):
    q = np.square((history[m:] - history[:(len(history) - m)]).mean())
    e = np.square((y_true-y_pred).mean())
    return np.sqrt(e/q)

