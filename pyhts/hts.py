from __future__ import annotations
import rpy2.robjects as robjects
from pyhts.accuracy import *
from pandas import DataFrame

from typing import List, Union, Optional, Callable
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
from scipy.sparse import csr_matrix

forecast = importr("forecast")


def _nodes2constraints(nodes: List[List[int]]) -> [csr_matrix, np.ndarray]:
    """construct constraints from nodes

    :param nodes: nodes that demonstrate hierarchy
    :return: constraints
    """
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


def _constraints_from_chars(names: List, chars: List) -> [csr_matrix, np.ndarray]:
    """construct constraints from columns and characters.

    :param names: names that represents location of series in hierarchy.
    :param chars: characters.
    :return: constraints
    """
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
    like hierarchical or grouped time series and temporal hierarchies :ref:`[1]<references>`.

    """
    def __init__(self, constraints: Union[csr_matrix, np.ndarray], bts: np.ndarray, node_level: np.ndarray, m: int = 1):
        """initialize a Hts object according to attributes directly. Do not use it unless you are familiar with meanings
        of parameters.

        :param constraints: constraints, also called "summing matrix" in some context.
        :param bts: bottom level time series.
        :param node_level: which level every nodes belong to.
        :param m: frequency of time series data.
        """
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
                 h: int,
                 base_method: Union[str, Callable] = "arima",
                 hf_method: str = "comb",
                 weights_method: str = "ols",
                 weights: Optional[np.ndarray] = None,
                 variance: str = "shrink",
                 parallel: bool = False,
                 constraint: bool = False,
                 constrain_level: int = 0
                 ) -> Hts:
        """forecast Hts using specific reconciliation methods and base methods.

        :param h: forecasting horizon.
        :param base_method:
            method for generate base forecast, arima, ets, or custom function.
            arima and ets are implemented using `forecast` package in R.
            If custom forecast function is specified and using "mint" method, the function
            should return the in-sample forecasts for estimating the covariance matrix, e.g. the
            base forecast returned should be :math:`(T+h) \\times n`.
        :param hf_method: method for hierarchical forecasting, "comb", "bu", "td", "mo"
        :param weights_method:
            "ols", "wls", "mint"(e.g Minimum Trace :ref:`MinT<mint>` ), weights method used for "comb"(e.g. optimal combination)
            reconciliation method, if you choose "wls", you should specify `weights`, or the result is same as ols. If
            you choose "mint", you should specify `variance` parameter.
        :param weights:
            weighting matrix used for "wls" method, can be "nseries" or custom_matrix, this custom matrix should be
            :math:`n\\times n`
        :param variance:
            "cov", "var", "shrink", variance estimation method used for `mint` method.
        :param parallel: If parallel, not supported for now.
        :param constraint: If some levels are constrained to be unchangeable when reconciling base forecasts.
        :param constrain_level: Which level is constrained to be unchangeable when reconciling base forecasts.
        :return: Hts: reconciled forecast.
        """
        keep_fitted = False
        import pyhts.reconciliation as fr
        if hf_method == "comb":
            # generate base forecast
            if weights_method == "mint":
                keep_fitted = True
            base_forecast = self.generate_base_forecast(base_method, h, keep_fitted)

            # reconcile base forecasts
            if weights_method == "ols":
                reconciled_y = fr.wls(self, base_forecast, method="ols")
            elif weights_method == "wls":
                if weights == "nseries":
                    reconciled_y = fr.wls(self, base_forecast, method="wls", weighting="nseries", constraint=constraint,
                                          constraint_level=constrain_level)
                elif isinstance(weights, np.ndarray):
                    reconciled_y = fr.wls(self, base_forecast, method="wls", weighting=weights,
                                          constraint=constraint, constraint_level=constrain_level)
                else:
                    raise ValueError("weights for wls method is not supported")
            elif weights_method == "mint":
                reconciled_y = fr.wls(self, base_forecast, method="mint", weighting=variance,
                                      constraint=constraint, constraint_level=constrain_level)
            else:
                raise ValueError("this comination method is not supported")
        else:
            raise NotImplementedError("this method is not implemented")

        return reconciled_y

    # base forecast 也变成Hts对象，方便精度比较
    def generate_base_forecast(self, method: str = "arima", h: int = 1, keep_fitted: bool = False)-> np.ndarray:
        """generate base forecasts by `forecast` in R with rpy2.

        :param method: base forecast method.
        :param h: forecasting horizons.
        :param keep_fitted: if keep in-sample fitted value, useful when mint method is specified.
        :return: base forecast
        """
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
    def accuracy(self, y_true: Hts, y_pred: Hts, levels: int = 0) -> Union[float, np.ndarray]:
        """calculate forecast accuracy, mase is supported only for now.

        :param y_true: real observations.
        :param y_pred: forecasts.
        :param levels: which level.
        :return: forecast accuracy.
        """
        # MASE
        agg_ts = self.aggregate_ts(levels=levels)
        agg_true = y_true.aggregate_ts(levels=levels)
        agg_pred = y_pred.aggregate_ts(levels=levels)
        mases = np.array(list(map(lambda x,y: mase(*x, y), zip(agg_ts.T, agg_true.T, agg_pred.T), [12]*agg_ts.shape[1])))
        return mases