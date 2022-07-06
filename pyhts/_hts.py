from __future__ import annotations
import numpy as np
from pyhts import _accuracy
from pandas import DataFrame
from copy import copy

from typing import List, Union, Optional
from scipy.sparse import csr_matrix
from pyhts._forecaster import BaseForecaster, AutoArimaForecaster

__all__ = ["Hts"]

def _nodes2constraints(nodes: List[List[int]]) -> [csr_matrix, np.ndarray]:
    """Construct constraints from nodes.

    :param nodes: nodes that demonstrate the hierarchy.
    :return: constraints.
    """
    nodes = copy(nodes)
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

    :param names: names that represent the location of series in the hierarchy.
    :param chars: characters.
    :return: constraints.
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


class Hts:
    """
    Class for hierarchical time series, which can be constructed from  cross-sectional hierarchical time series
    like hierarchical or grouped time series and temporal hierarchies :ref:`[1]<references>`.

    """
    def __init__(self, constraints: Union[csr_matrix, np.ndarray], bts: np.ndarray, node_level: np.ndarray, m: int = 1):
        """Initialize a Hts object according to attributes directly. Do not use it unless you are familiar with the
        parameters.

        :param constraints: constraints, also called "summing matrix" in some contexts.
        :param bts: bottom-level time series.
        :param node_level: which level each node belongs to.
        :param m: frequency of time series.
        """
        if isinstance(constraints, np.ndarray):
            constraints = csr_matrix(constraints)
        self.constraints = constraints
        self.bts = bts
        self.node_level = node_level
        self.m = m
        self.base_forecast = None
        self.keep_fitted = False

    def aggregate_ts(self, levels: Union[int, List[int]] = None) -> np.ndarray:
        """Aggregate bottom-level time series.

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

    def forecast(self,
                 h: int,
                 base_method: Union[str, None] = "arima",
                 base_forecaster: Union[List, None, BaseForecaster] = None,
                 hf_method: str = "comb",
                 comb_method: str = "ols",
                 weights: Optional[np.ndarray, str] = None,
                 parallel: bool = False,
                 constraint: bool = False,
                 constrain_level: int = 0,
                 keep_base_forecast: bool = True
                 ) -> Hts:
        """Forecast Hts using specific reconciliation methods and base forecasting methods.

        :param h: forecasting horizon.
        :param base_method:
            method for generate base forecasts, arima, ets, or custom function.
            arima and ets are implemented using `forecast` package in R.
            If custom forecast function is specified and you are using "mint" method, the function
            returns the in-sample forecasts for estimating the covariance matrix, i.e., the
            base forecast returned should be :math:`(T+h) \\times n`.
        :param base_forecaster:
            list for the base forecaster of each level. If you want different base forecasting methods
            for different levels, just pass a list of base forecasters, see :doc:forecaster.
        :param hf_method: method for hierarchical forecasting, "comb", "bu", "td", "mo".
        :param comb_method:
            "ols", "wls", "mint"(i.e., Minimum Trace :ref:`[2]<mint>` ), weighting method used for "comb"(i.e., optimal
            combination) reconciliation method. If you choose "wls", you should specify `weights`, or the result is the same
            as ols. If you choose "mint", you should specify the `variance` parameter.
        :param weights:
            weighting matrix used for `wls` combination method and variance used for `mint` combination method.
            For `wls`, it can be "structural" or custom_matrix, which is an
            :math:`n\\times n` symmetric matrix.
            For `mint`, it can be "sample", "variance" or "shrinkage". Please refer to :doc:`/tutorials/reconciliation`.
        :param parallel: not supported for now.
        :param constraint: Ture if some levels are constrained to be unchangeable when reconciling the base forecasts. False otherwise.
        :param constrain_level: Which level is constrained to be unchangeable when reconciling the base forecasts.
        :param keep_base_forecast:
            if keep keep_base_forecast, if True, attribute `Hts.base_forecast` is set. if false, it will be None
        :return: Hts: reconciled forecast.
        """
        import pyhts._reconciliation as fr
        if hf_method == "comb":
            # generate base forecast
            if comb_method == "mint":
                self.keep_fitted = True
            if base_method == "arima":
                base_forecaster = [AutoArimaForecaster(self.m)]*int(max(self.node_level)+1)
            elif base_method == "ets":
                base_forecaster = [EtsForecaster(self.m)]*int(max(self.node_level)+1)
            elif base_method is not None:
                raise ValueError("This base forecasting method is not supported for now.")
            else:
                if base_forecaster is None:
                    raise ValueError("base_method of base_forecaster is not specified.")
                elif not isinstance(base_forecaster, List):
                    base_forecaster = [base_forecaster] * int(max(self.node_level)+1)
                else:
                    assert len(base_forecaster) == int(max(self.node_level)+1)
            base_forecast = self.generate_base_forecast(base_forecaster, h, self.keep_fitted)
            # reconcile base forecasts
            if comb_method == "ols":
                reconciled_y = fr.wls(self, base_forecast, method="ols")
            elif comb_method == "wls":
                if weights == "structural":
                    reconciled_y = fr.wls(self, base_forecast, method="wls", weighting="structural",
                                          constraint=constraint, constraint_level=constrain_level)
                elif isinstance(weights, np.ndarray):
                    reconciled_y = fr.wls(self, base_forecast, method="wls", weighting=weights,
                                          constraint=constraint, constraint_level=constrain_level)
                else:
                    raise ValueError("This weighting method for wls is not supported.")
            elif comb_method == "mint":
                reconciled_y = fr.wls(self, base_forecast, method="mint", weighting=weights,
                                      constraint=constraint, constraint_level=constrain_level)
            else:
                raise ValueError("This comination method is not supported.")
        else:
            raise NotImplementedError("This method is not implemented.")
        if keep_base_forecast:
            self.base_forecast = base_forecast
        return reconciled_y

    def generate_base_forecast(self, method: List[BaseForecaster], h: int = 1, keep_fitted: bool = False) -> np.ndarray:
        """Generate base forecasts by `forecast` in R with rpy2.

        :param method: base forecasting method.
        :param h: forecasting horizon.
        :param keep_fitted: True if you want to keep in-sample fitted value. This parameter is useful when mint method is used.
        :return: base forecasts
        """
        k = int(max(self.node_level)+1)
        length = self.bts.shape[0]
        n = len(self.node_level)
        if keep_fitted:
            f_casts = np.zeros([h + length, n])
        else:
            f_casts = np.zeros([h, n])
        j = 0
        for i in range(k):
            aggts = self.aggregate_ts(levels=i)
            forecaster = method[i]
            for ts in range(aggts.shape[1]):
                f_casts[:, j] = forecaster.forecast(hist=aggts[:, ts], h=h, keep_fitted=keep_fitted)
                j += 1
        return f_casts

    @classmethod
    def from_temporal_hierarchy(cls, ts: np.ndarray, m: int, aggregate_lens: List[int]):
        """Construct Hts from a temporal hierarchy.

        :param ts: a time series.
        :param m: frequency of the time series.
        :param aggregate_lens:
            length of bottom level time series used for aggregation. For example, for monthly data,
            3 means quarterly data, 6 means half-annual data, and 12 means annual time series.  The time
            series should include 1 at least.
        :return: Hts object.
        """
        aggregate_lens.sort()

        smatrix = np.zeros([sum([int(m / k) for k in aggregate_lens]), m])
        node_level = []
        if len(ts) % m != 0:
            Warning("The length of the historical time series is not multiple of m. Some observations at very beginning "
                    "will be cut out.")
            ts = ts[len(ts) % m:]

        if aggregate_lens[0] != 1:
            raise ValueError("Please always include 1 in the aggregate_lens.")

        aggregate_lens.reverse()
        index = 0
        for i in range(len(aggregate_lens)):
            k = aggregate_lens[i]
            if (m % k) != 0:
                raise ValueError("Aggregate length should be a factor of m.")
            mk = m // k
            for j in range(mk):
                smatrix[index, j * k:(j + 1) * k] = 1
                index += 1
                node_level.append(i)
        constraints = csr_matrix(smatrix)
        bts = ts.reshape([len(ts) // m, m])
        node_level = np.array(node_level)
        hts = Hts(constraints, bts, node_level, m)
        return hts

    @classmethod
    def from_hts(cls, bts: Union[np.ndarray, DataFrame],
                 m: int,
                 characters: Optional[List[int]] = None,
                 nodes: Optional[List[List]] = None) -> Hts:
        """Construct hts from cross-sectional hierarchical time series.

        :param bts: :math:`T\\times m` bottom level series.
        :param m: frequency of time series
        :param characters:
            length of characters of each level.
            For example, in "ABC", 'A' represents 'A' series of  first level,  'B' represents 'B' series under node 'A',
            and 'C' represents 'C' series under node 'AB'. So the value of parameter is [1,1,1].
        :param nodes: List of lists to demonstrate the hierarchy, see details.
        :return: Hts object.
        """
        if isinstance(bts, DataFrame):
            names = bts.columns
            bts = bts.values
        elif isinstance(bts, np.ndarray):
            bts = bts
            names = None
        else:
            raise TypeError("bts must be numpy.ndarray or pandas.DataFrame.")
        if nodes is not None:
            constraints, node_level = _nodes2constraints(nodes)
        elif characters is not None:
            constraints, node_level = _constraints_from_chars(list(names), characters)
        else:
            constraints, node_level = _nodes2constraints([[bts.shape[1]]])
        hts = cls(constraints, bts, node_level, m)
        return hts

    def accuracy(self, y_true: Hts, y_pred: Hts, levels: Union[int, None, List] = None, measure: List[str] = None) -> Union[float, DataFrame]:
        """Calculate the forecast accuracy. mase and mse are supported only for now.

        :param y_true: real observations.
        :param y_pred: forecasts.
        :param levels: which levels.
        :param measure:
            mase and mse are supported for now. e.g., ['mase'], ['mse', 'mase'].
            if None, mase is calculated.
        :return: forecast accuracy.
        """
        if measure is None:
            measure = ['mase']
        agg_ts = self.aggregate_ts(levels=levels)
        agg_true = y_true.aggregate_ts(levels=levels)
        agg_pred = y_pred.aggregate_ts(levels=levels)
        accs = DataFrame()
        for me in measure:
            try:
                accs[me] = np.array(list(map(lambda x: getattr(accuracy, me)(*x),
                                             zip(agg_ts.T, agg_true.T, agg_pred.T, [self.m] * agg_ts.shape[1]))))
            except AttributeError:
                print('This forecasting measure is not supported!')
        return accs

    def accuracy_base(self, y_true: Hts, levels: Union[int, None, List] = None, measure: List[str] = None) -> DataFrame:
        """Calculate forecast accuracy of base forecasts.

        :param y_true: real observations.
        :param levels: which levels.
        :param measure:
            mase and mse are supported for now. e.g., ['mase'], ['mse', 'mase'].
            if None, mase is calculated.
        :return: forecast accuracy of base forecasts.
        """
        agg_ts = self.aggregate_ts(levels=levels)
        agg_true = y_true.aggregate_ts(levels=levels)
        horizon = agg_true.shape[0]
        agg_pred = self.base_forecast[-horizon:, :]
        if measure is None:
            measure = ['mase']
        accs = DataFrame()
        for me in measure:
            try:
                accs[me] = np.array(list(map(lambda x: getattr(accuracy, me)(*x),
                                             zip(agg_ts.T, agg_true.T, agg_pred.T, [self.m] * agg_ts.shape[1]))))
            except AttributeError:
                print(f'Forecasting measure {me} is not supported!')
        return accs


if __name__ == '__main__':
    print(_nodes2constraints([[2], [2, 2]]))
