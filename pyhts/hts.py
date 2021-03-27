from __future__ import annotations
import rpy2.robjects as robjects
import pyhts.reconciliation as fr
from pyhts.accuracy import *
from pandas import DataFrame

from typing import List, Union, Optional
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
from scipy.sparse import csr_matrix

forecast = importr("forecast")


def _nodes2constraints(nodes: List):
    n = sum(nodes[-1])
    m = sum(map(sum, nodes)) + 1
    node_level = n * [len(nodes)]
    nodes.append([1]*n)
    s = np.zeros([m, n])
    c_row_start = m - n
    s[c_row_start:, :] = np.identity(n)
    bts_count = nodes[-1]
    for level_idx in range(len(nodes)-2, -1, -1):

        c_cum = 0
        c_x = 0
        level = nodes[level_idx]
        c_row_start = c_row_start - len(level)
        new_bts_count = []
        c_row = c_row_start
        for node_idx in range(len(nodes[level_idx])):
            n_x = c_x+level[node_idx]
            new_bts_count.append(sum(bts_count[c_x:n_x]))
            n_cum = c_cum + new_bts_count[-1]
            s[c_row, c_cum:n_cum] = 1
            c_cum = n_cum
            c_row += 1
            c_x = n_x
            node_level.insert(0, level_idx)
        bts_count = new_bts_count
    return csr_matrix(s), np.array(node_level)


def _constraints_from_chars(names: List, chars: List):
    total_pos = 0
    import pandas as pd
    df = pd.DataFrame()
    df["0"] = [1]*len(names)
    for index in range(len(chars)):
        a = list(map(lambda x: x[:(total_pos+chars[index])], names))
        df[f"{index+1}"] = a
        total_pos += chars[index]

    constraints = pd.get_dummies(df).values.T

    return csr_matrix(constraints), np.array(list(map(lambda x: int(x.split("_")[0]), pd.get_dummies(df).columns)))


# TODO: 调整bts以及预测中用到的base foracast的维度，避免过多无用的转置
class Hts:
    """
    Class for hierarchical time series, can be constructed from  cross-sectional hierarchical time series
    like hierarchical or grouped time series and temporal hierarchies [1]_.

    """
    def __init__(self, constraints: Union[csr_matrix, np.ndarray], bts: np.ndarray, node_level: np.ndarray, m: int = 1):
        if isinstance(constraints, np.ndarray):
            constraints = csr_matrix(constraints)
        self.constraints = constraints
        self.bts = bts
        self.node_level = node_level
        self.m = m

    @classmethod
    def from_hts(cls, bts: Union[np.ndarray, DataFrame],
                 m: int,
                 characters: Optional[List[int]] = None,
                 nodes: Optional[List[List]] = None) -> Hts:
        """Construct hts from cross-sectional hierarchical time series.

        :param bts: :math:`T\\times m` bottom level series.
        :param m: frequency of time series
        :param characters:
            length of characters of each level,
            for example, "ABC" 'A' represents 'A' series of  first level,  'B' represents 'B' series under node 'A',
            'C' represents 'C' series under node 'AB'. So the value of parameter is [1,1,1]
        :param nodes: List of list to demonstrate the hierarchy, see details.
        :return: Hts object.
        """
        if isinstance(bts, DataFrame):
            names = bts.columns
            bts = bts.values
        elif isinstance(bts, np.ndarray):
            bts = bts
            names = None
        else:
            raise TypeError("bts must be numpy.ndarray or pandas.DataFrame")
        if nodes is not None:
            constraints, node_level = _nodes2constraints(nodes)
        elif characters is not None:
            constraints, node_level = _constraints_from_chars(list(names), characters)
        else:
            constraints, node_level = _nodes2constraints([[bts.shape[1]]])
        hts = cls(constraints, bts, node_level, m)
        return hts

    def aggregate_ts(self, levels: Union[int, List[int]] = None) -> np.ndarray:
        """aggregate bottom-levels time series.

        :param levels: which levels you want.
        :return: upper-level time series.
        """
        if isinstance(levels, int):
            s = self.constraints[np.where(self.node_level == levels)]
            return s.dot(self.bts.T).T
        if isinstance(levels, list):
            s = self.constraints[np.isin(self.node_level, levels)]
            return s.dot(self.bts.T).T
        return self.constraints.dot(self.bts.T).T

    # TODO: 优化结构
    def forecast(self,
                 h,
                 base_method="arima",
                 hf_method="comb",
                 weights_method="ols",
                 weights=None,
                 variance="shrink",
                 parallel=False,
                 constraint=False,
                 constrain_level=0
                 ):
        """
        :param h: forecast horzion
        :param base_method: method for generate base forecast
        :param hf_method: method for hierarchical forecast: comb, bu, td, mo
        :param weights_method: ols, wls, mint
        :param weights: nseries, custom_matrix
        :param variance: cov, var, shrink
        :param parallel: Bool
        :param constraint: Bool, if constrained
        :param constrain_level: int, if constrained, which level
        :return: Hts: reconciled forecast
        """
        keep_fitted = False
        if hf_method == "comb":
            # generate base forecast
            if weights_method == "wls":
                keep_fitted = True
            base_forecast = self.generate_base_forecast(base_method, h, keep_fitted)

            # reconcile base forecasts
            if weights_method == "ols":
                if constraint:
                    reconciled_y = fr.constrained_wls(self, base_forecast, constrain_level)
                else:
                    reconciled_y = fr.wls(self, base_forecast)
            elif weights_method == "wls":
                if weights == "nseries":
                    weight_matrix = np.diag(self.constraints.dot(np.array([1] * self.bts.shape[1])))
                    reconciled_y = fr.wls(self, base_forecast, weight_matrix)
                elif isinstance(weights, np.ndarray):
                    reconciled_y = fr.wls(self, base_forecast, weights)
                else:
                    raise ValueError("weights for wls method is not supported")
            elif weights_method == "mint":
                reconciled_y = fr.min_trace(self, base_forecast, variance)
            else:
                raise ValueError("this comination method is not supported")
        else:
            raise NotImplementedError("this method is not implemented")

        return reconciled_y

    def generate_base_forecast(self, method="arima", h=1, keep_fitted=False):
        ts = robjects.r['ts']

        y = self.constraints.dot(self.bts.T)
        if keep_fitted:
            f_casts = np.zeros([h+y.shape[1], y.shape[0]])
        else:
            f_casts = np.zeros([h, y.shape[0]])
        if method == "arima":
            auto_arima = forecast.auto_arima
            for i in range(y.shape[0]):
                series = ts(FloatVector(y[i, :]), frequency=self.m)
                model = forecast.forecast(auto_arima(series), h=12)
                if keep_fitted:
                    f_casts[:y.shape[1], i] = np.array(model.rx2["fitted"])
                    f_casts[y.shape[1]:, i] = np.array(model.rx2["mean"])
                else:
                    f_casts[:, i] = np.array(model.rx2["mean"])
            return f_casts

    # TODO: 修正结构方便base forecast的比较
    def accuracy(self, y_true, y_pred, levels=0):
        # MASE
        agg_ts = self.aggregate_ts(levels=levels)
        agg_true = y_true.aggregate_ts(levels=levels)
        agg_pred = y_pred.aggregate_ts(levels=levels)
        mases = np.array(list(map(lambda x,y: mase(*x, y), zip(agg_ts.T, agg_true.T, agg_pred.T), [12]*agg_ts.shape[1])))
        return mases