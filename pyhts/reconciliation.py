import numpy as np
import scipy.linalg as lg

from copy import deepcopy


def lamb_estimate(x):
    T = x.shape[0]
    covm = x.T.dot(x)/T
    xs = x / np.sqrt(np.diag(covm))
    corm = xs.T.dot(xs)/T
    np.fill_diagonal(corm, 0)
    d = np.sum(np.square(corm))
    xs2 = np.square(xs)
    v = 1/(T*(T-1))*(xs2.T.dot(xs2) - 1/T*np.square(xs.T.dot(xs)))
    np.fill_diagonal(v, 0)
    lamb = np.max(np.min([np.sum(v)/d, 1]), 0)
    return lamb


def wls(hts, base_forecast, weight_matrix=None):

    if not weight_matrix:
        weight_matrix = np.identity(base_forecast.shape[1])
    S = hts.constraints.toarray()
    G = np.linalg.inv(S.T.dot(weight_matrix).dot(S)).dot(S.T).dot(weight_matrix)
    reconciled_y = G.dot(base_forecast.T)
    reconciled_hts = deepcopy(hts)
    reconciled_hts.bts = reconciled_y.T

    return reconciled_hts


def min_trace(hts, base_forecast, weighting_method="shrink"):
    y = hts.aggregate_ts()
    T = y.shape[0]

    error = y - base_forecast[:T, :]
    W = error.T.dot(error)/T
    if weighting_method == "var":
        weight_matrix = np.diag(np.diagonal(W))
    elif weighting_method == "cov":
        weight_matrix = W
        if not np.all(np.linalg.eigvals(weight_matrix)>0):
            raise ValueError("Sample method needs covariance matrix to be positive definite")
    elif weighting_method == "shrink":
        lamb = lamb_estimate(error)
        weight_matrix = lamb * np.diag(np.diag(W)) + (1 - lamb) * W
    else:
        raise NotImplementedError("this min_trace method has not been implemented")
    S = hts.constraints.toarray()
    w_inv = np.linalg.inv(weight_matrix)
    G = lg.inv(S.T.dot(w_inv).dot(S)).dot(S.T).dot(w_inv)
    reconciled_y = G.dot(base_forecast[T:, :].T)
    reconciled_hts = deepcopy(hts)
    reconciled_hts.bts = reconciled_y.T
    return reconciled_hts