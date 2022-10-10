import numpy as np
from typing import *
import scipy.linalg as lg
from pyhts._hierarchy import *

__all__ = ["mint"]


def _lamb_estimate(x: np.ndarray) -> float:
    """Estimate :math`\\lambda` used in :ref:`shrinkage` estimator of mint method.

    :param x: in-sample 1-step-ahead forecast error.
    :return: :math`\\lambda`.
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


def mint(hierarchy: Union[Hierarchy, TemporalHierarchy],
         error: np.ndarray = None,
         method: str = "ols",
         weighting: Union[str, np.ndarray, None] = None,
         immutable_set: Optional[List[int]] = None) -> np.ndarray:
    """Function for forecast reconciliation.

    :param hierarchy: historical time series.
    :param error: in-sample error with shape (n, T)
    :param method: method used for forecast reconciliation, i.e., ols, wls, and mint.
    :param weighting:
        method for the weight matrix used in forecast reconciliation, i.e., covariance matrix in mint
        or wls.
    :param immutable_set: the subset of time series to be unchanged during reconciliation.
    :return: reconciled forecasts.
    """
    S = hierarchy.s_mat
    n = S.shape[0]
    m = S.shape[1]
    if method == "mint":
        T = error.shape[1]
        W = error.dot(error.T) / T
        if weighting == "variance":
            weight_matrix = np.diag(np.diagonal(W))
        elif weighting == "sample":
            weight_matrix = W
            if not np.all(np.linalg.eigvals(weight_matrix) > 0):
                raise ValueError("Sample method needs covariance matrix to be positive definite.")
        elif weighting == "shrinkage":
            lamb = _lamb_estimate(error.T)
            weight_matrix = lamb * np.diag(np.diag(W)) + (1 - lamb) * W
        else:
            raise NotImplementedError("This min_trace method has not been implemented.")

    elif method == "ols":
        weight_matrix = np.identity(n)
    else:
        if isinstance(weighting, np.ndarray):
            weight_matrix = np.linalg.inv(weighting)
        elif weighting == "structural":
            weight_matrix = np.diag(S.dot(np.array([1]*m)))
        else:
            raise ValueError("This wls weighting method is not supported for now.")
    G = compute_g_mat(hierarchy, weight_matrix, immutable_set)
    return G


def _construct_u_mat(hierarchy: Hierarchy, immutable_set: Optional[List[int]] = None):
    """construct U matrix used in solution.

    :param immutable_set:
    :return:
    """
    s_mat = hierarchy.s_mat
    n, m = s_mat.shape
    u1 = np.identity(n - m)
    u2 = 0-s_mat[:(n-m), :].astype('int32')
    u_mat = np.concatenate([u1, u2], axis=1)
    if immutable_set:
        u_up = np.identity(n)[immutable_set]
        return np.concatenate([u_up, u_mat], axis=0).T
    return u_mat.T



def compute_g_mat(hierarchy: Hierarchy, weight_matrix,
                  immutable_set: Optional[Iterable[int]] = None):
    """Compute G matrix given the weight_matrix.

    :param hierarchy:
    :param weight_matrix:
    :param immutable_set: the subset of time series to be unchanged during reconciliation
    :return:
    """
    n, m = hierarchy.s_mat.shape

    if immutable_set:
        immutable_set = list(immutable_set)
        k = len(immutable_set)
        assert k <= m, f"The number of immutable series can not be bigger than the number of bottom-level series {m}."
    u = _construct_u_mat(hierarchy, immutable_set=immutable_set)
    J = np.concatenate([np.zeros([m, n-m]), np.identity(m)], axis=1)
    v = np.zeros([n - m, n])
    if immutable_set:
        v = np.concatenate([np.identity(n)[immutable_set], v])
    target = u.T.dot(weight_matrix).dot(u)
    x, lower = lg.cho_factor(target)
    inv_dot = lg.cho_solve((x, lower), (u.T-v))
    return J - J.dot(weight_matrix).dot(u).dot(inv_dot)
