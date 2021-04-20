import numpy as np
import scipy.linalg as lg

from typing import Union
from pyhts.hts import Hts


def lamb_estimate(x: np.ndarray) -> float:
    """estimate :math`\\lambda` used in :ref:`shrinkage` estimator of mint method

    :param x: in-sample 1-step-ahead forecast error
    :return: :math`\\lambda`
    """
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


def wls(hts: Hts,
        base_forecast: np.ndarray,
        method: str = "ols",
        weighting: Union[str, np.ndarray, None] = None,
        constraint: bool = False,
        constraint_level: int = 0) -> Hts:
    """function for forecast reconciliation.

    :param hts: history time series.
    :param base_forecast: base forecasts.
    :param method: method used for forecast reconciliation, e.g. ols, wls, mint.
    :param weighting:
        method for calculating weight matrix used in reconciliation method, e.g. covariance matrix in mint
        or wls, for details, refer to :doc:`/tutorials/reconciliation`
    :param constraint: If some levels are constrained to be unchangeable when reconciling base forecasts.
    :param constraint_level: Which level is constrained to be unchangeable when reconciling base forecasts.
    :return: reconciled forecasts
    """
    y = hts.aggregate_ts()
    T = y.shape[0]
    S = hts.constraints.toarray()
    n = S.shape[0]
    m = S.shape[1]
    if method == "mint":
        out_sample_fcasts = base_forecast[T:, :]
    else:
        out_sample_fcasts = base_forecast

    if method == "mint":
        in_sample_fcasts = base_forecast[:T, :]
        error = y - in_sample_fcasts
        W = error.T.dot(error) / T
        if weighting == "variance":
            weight_matrix = np.diag(np.diagonal(W))
        elif weighting == "sample":
            weight_matrix = W
            if not np.all(np.linalg.eigvals(weight_matrix) > 0):
                raise ValueError("Sample method needs covariance matrix to be positive definite")
        elif weighting == "shrinkage":
            lamb = lamb_estimate(error)
            weight_matrix = lamb * np.diag(np.diag(W)) + (1 - lamb) * W
        else:
            raise NotImplementedError("this min_trace method has not been implemented")

    elif method == "ols":
        weight_matrix = np.identity(n)
    else:
        if isinstance(weighting, np.ndarray):
            weight_matrix = np.linalg.inv(weighting)
        elif weighting == "structural":
            weight_matrix = np.diag(S.dot(np.array([1]*m)))
        else:
            raise ValueError("this wls weights is not supported now.")

    w_inv = np.linalg.inv(weight_matrix)
    if constraint:
        reconciled_y = constrained(hts, w_inv, out_sample_fcasts, constraint_level)
    else:
        G = lg.inv(S.T.dot(w_inv).dot(S)).dot(S.T).dot(w_inv)
        reconciled_y = G.dot(out_sample_fcasts.T)
    reconciled_hts = Hts(hts.constraints, reconciled_y.T, hts.node_level, hts.m)
    return reconciled_hts


def constrained(hts: Hts, weights: np.ndarray, base_forecast: np.ndarray, constraint_level: int):
    """function to compute constrained reconciled forecasts

    :param hts: history Hts
    :param weights: weight matrix used in unconstrained reconciliation method.
    :param base_forecast: base forecasts.
    :param constraint_level:
    :return: Which level is constrained to be unchangeable when reconciling base forecasts.
    """
    if constraint_level == np.max(hts.node_level):
        return base_forecast
    a = hts.node_level > constraint_level
    x = hts.constraints[a, :].toarray()
    q = hts.constraints[hts.node_level == constraint_level].T.toarray()
    weights = weights[np.ix_(a, a)]
    base = base_forecast.T[a]
    c = base_forecast.T[hts.node_level == constraint_level]
    x_tr_x = np.linalg.inv(x.T.dot(weights).dot(x))
    # calculate ols result
    beta_hat = x_tr_x.dot(x.T).dot(weights).dot(base)
    # calculate cls result
    reconciled_bts = beta_hat - x_tr_x.dot(q).dot(np.linalg.inv(
        q.T.dot(x_tr_x).dot(q)
    )).dot(q.T.dot(beta_hat) - c)
    return reconciled_bts

