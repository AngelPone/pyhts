import numpy as np

from pyhts.hts import Hts
from copy import deepcopy


def wls(hts: Hts, base_forecast, weight_matrix=None):
    if not weight_matrix:
        weight_matrix = np.identity(base_forecast.shape[1])
    S = hts.constraints
    G = np.linalg.inv(S.T.dot(weight_matrix).dot(S)).dot(S.T).dot(weight_matrix)
    reconciled_y = G.dot(base_forecast.T)
    reconciled_hts = deepcopy(hts)
    reconciled_hts.bts = reconciled_y.T
    return reconciled_hts


def min_trace(hts: Hts, base_forecast, f_fun=None,weighting_method="shrinkage"):
    return 0