import numpy as np
import pandas as pd
from copy import copy
from typing import Union, List, Tuple, Optional, Iterable, Dict
# from pyhts import _accuracy
from itertools import combinations
from scipy.sparse import csr_array, identity, vstack, kron

__all__ = ["Hierarchy", "TemporalHierarchy"]

def get_all_combinations(lst):
    n = len(lst)
    for r in range(1, n):
        for comb in combinations(lst, r):
            yield comb


class Hierarchy:
    """Class for a hierarchy structure.

    **Attributes**

        .. py:attribute:: s_mat

            summing matrix.
        .. py:attribute:: period

            frequency of time series.
        .. py:attribute:: node_name

            name of each node in the pattern: level-name_attribute-anme
        .. py:attribute:: node_level

            level name of each node.
    """

    def __init__(self, s_mat, indices: pd.DataFrame):
        self.s_mat = s_mat
        self.indices = indices

        # cross-sectional numbers
        agg_periods = self.indices['agg_period'].unique().tolist()
        agg_periods.sort(reverse=True)
        self.frequencies = [max(agg_periods) // i for i in agg_periods]
        self.n = s_mat.shape[0] // sum(self.frequencies)
        self.m = s_mat.shape[1] // indices["agg_period"].max()


    @classmethod
    def new(cls, df: Optional[pd.DataFrame] = None,
            structures: Optional[List[str]] = None,
            excludes: Optional[List[Dict]] = None,
            includes: Optional[List[Dict]] = None,
            agg_periods: Optional[List[int]] = None) -> "Hierarchy":
        """Construct cross-sectional/temporal/cross-temporal hierarchy. If only agg_periods is specified, \
        a temporal hierarchy is constructed. If only structures is specified, a cross-sectional hierarchy is \
        constructed. If both are specified, a cross-temporal hierarchy is constructed.

        **Examples** TODO
            >>> import pandas as pd


        :param df: DataFrame contains keys for determining cross-section structural. Each row responsents a \
        cross-sectional time series in the bottom level. Each col represents a attribute of the time series. \
        The attributes are used to define the cross-sectional structure of the hierarchical time series.
        :param structures: Columns to use. Use all columns by default.
        :param excludes: middle levels excluded from the hierarchy, identified by attribute values. \
        For example, in a (Category, Subcategory, Product) sales Hierarchy, [{"Subcategory": "Fruit"}] means \
        excluding the "Fruit" time series, which is sum of all products in the "Fruit" subcategory.
        :param includes: same structure as excludes, but only the specified series are included in the hierarchy.
        :param agg_periods: list of aggregation periods used to construct temporal hierarchy. \
        For example, [1, 2, 4] for a quarterly time series means aggregating the quarterly time series to \
        half-yearly and yearly levels. 
        :return: Hierarchy object.
        """

        if df is not None:
            if structures is not None:
                df = df[structures]
            df = df.drop_duplicates().reset_index(drop=True)
            columns = df.columns
            indices = df.copy()
            indptr = np.array([0])
            indices_s = np.zeros(0, dtype='int')
            data = np.zeros(0)
            current_row = 0
            existing_idxs = []
            for comb in get_all_combinations(columns):
                keys = df.groupby(list(comb))
                keys_dict = []
                for (key, idx) in keys.indices.items():
                    if len(idx) == 1:
                        continue
                    if str(idx) in existing_idxs:
                        continue
                    existing_idxs.append(str(idx))
                    indptr = np.append(indptr, indptr[-1] + len(idx))
                    current_row += 1
                    indices_s = np.concatenate([indices_s, idx])
                    keys_dict.append(key)
                tmp_df = pd.DataFrame(keys_dict, columns=list(comb))
                indices = pd.concat([tmp_df, indices], axis=0, ignore_index=True)
            s_mat = csr_array((np.array([1] * len(indices_s), dtype='int'), indices_s, indptr), shape=(current_row, df.shape[0]))
            s_mat = vstack([s_mat, identity(df.shape[0], dtype='int')])
            if agg_periods is None:
                indices["agg_period"] = 1
        if agg_periods is not None:
            assert len(agg_periods) > 1, "agg_periods should be a list of length > 1"
            assert 1 in agg_periods, "agg_periods should contain 1"
            agg_periods = list(set(agg_periods))
            agg_periods.sort(reverse=True)
            s_mat_temporal = None
            for agg_period in agg_periods:
                assert agg_periods[0] % agg_period == 0, f"agg_period should be a factor of max agg_periods, \
                    {agg_periods[0]} % {agg_period} != 0"
                
                    
                s_mat_tmp = np.zeros((agg_periods[0] // agg_period, agg_periods[0]), dtype='int')
                for i in range(agg_periods[0] // agg_period):
                    s_mat_tmp[i, i * agg_period:(i + 1) * agg_period] = 1
                if s_mat_temporal is None:
                    s_mat_temporal = s_mat_tmp
                else:
                    s_mat_temporal = np.concatenate([s_mat_temporal, s_mat_tmp], axis=0)
            s_mat_temporal = csr_array(s_mat_temporal)
            indices_temporal = pd.DataFrame({"agg_period": agg_periods, "key_cartisian": 1})
            
            # Cross-temporal hierarchy
            if df is not None:
                s_mat = kron(s_mat, s_mat_temporal)
                indices["key_cartisian"] = 1
                indices = indices.merge(indices_temporal, on="key_cartisian")
            # Temporal hierarchy
            else:
                s_mat = s_mat_temporal
                indices = indices_temporal
            indices.drop("key_cartisian", axis=1, inplace=True)

        return cls(s_mat, indices)

    @property
    def type(self):
        """Type of the hierarchy, either cross-sectional, temporal or cross-temporal."""
        if self.indices["agg_period"].max() == 1:
            return "cs"
        elif self.indices.shape[1] == 1:
            return "te"
        else:
            return "ct"
    
    def aggregate_ts(self, bts: np.ndarray, indices = None) -> Union[np.ndarray, List[Dict]]:
        """Aggregate bottom-level time series.

        :param bts: bottom-level time series, array-like of shape (T, m)
        :param indices: indices of the series to aggregate
        :return: y.
        """
        # cross-sectional hierarchy
        max_k = self.indices['agg_period'].max()
        if max_k == 1:
            return bts.dot(self.s_mat.toarray().T)
        # temporal hierarchy and cross-temporal hierarchy
        else: 
            if self.type == "te": 
                assert len(bts.shape) == 1, "temporal hierarchy can only be applied to univariate time series"
                bts = bts.reshape((-1, 1))
            time_window = bts.shape[0]
            if time_window % max_k != 0:
                T_ = time_window // max_k * max_k
                bts = bts[(bts.shape[0] - T_ + 1):,]
                Warning(f"the observations at the first {time_window - T_} timestamps are dropped")
            
            bts = np.concatenate([bts[:,i].reshape((-1, max_k)) for i in range(self.m)], axis=1)
            all_ts = bts.dot(self.s_mat.toarray().T)
            current_idx = 0
            output = {}
            for freq in self.frequencies:
                idx = [list(range(i * sum(self.frequencies) + current_idx, i * sum(self.frequencies) + current_idx + freq)) for i in range(self.n)]
                tmp = [all_ts[:, i] for i in idx]
                tmp = [i.reshape((-1,)) for i in tmp]
                if len(tmp) == 1:
                    tmp = tmp[0]
                else:
                    tmp = np.stack(tmp, axis=1)
                current_idx += freq
                # TODO: cross-sectional indices when cross-temporal hierarchy
                output[freq] = tmp
            return output

    def check_hierarchy(self, *hts):
        pass
        # for ts in hts:
        #     if ts.shape[1] == self.s_mat.shape[1]:
        #         return True
        #     else:
        #         return False

    def accuracy_base(self, real, pred, hist=None,
                      levels: Optional[Union[str, Iterable[str]]] = None,
                      measure: List[str] = None) -> pd.DataFrame:
        """Calculate the forecast accuracy of base forecasts.

        :param real: real future observations, array-like of shape (forecast_horizon, m)
        :param pred: forecast values, array-like of shape (forecast_horizon, n)
        :param levels: which levels. None means all levels.
        :param hist: historical time series.
        :param measure: list of measures, e.g., ['mase'], ['mse', 'mase'].
        :return: forecast accuracy of base forecasts.
        """
        pass
        # assert self.check_hierarchy(real), f"True observations should be of shape (h, m)"
        # assert real.shape[0] == pred.shape[0], \
        #     f" {real.shape} True observations have different lengths with {real.shape} forecasts"

        # if measure is None:
        #     measure = ['mase', 'mape', 'rmse']
        # if 'mase' in measure or 'smape' in measure or 'rmsse' in measure:
        #     assert hist is not None
        #     assert self.check_hierarchy(hist), "History observations should be of shape(T, m)"
        #     hist = self.aggregate_ts(hist, levels=levels).T
        # else:
        #     hist = [None] * self.s_mat.shape[0]
        # accs = pd.DataFrame()

        # agg_true = self.aggregate_ts(real, levels=levels).T

        # if levels is not None:
        #     pred = pred.T[np.isin(self.node_level, levels), ]
        # else:
        #     pred = pred.T

        # for me in measure:
        #     try:
        #         accs[me] = np.array([getattr(_accuracy, me)(agg_true[i], pred[i], hist[i], self.period)
        #                              for i in range(hist.shape[0])])
        #     except AttributeError:
        #         print(f'Forecasting measure {me} is not supported!')
        # accs.index = self.node_name[np.isin(self.node_level, levels)] if levels is not None else self.node_name
        # return accs

    def accuracy(self, real, pred, hist=None,
                 levels: Union[str, None, Iterable[str]] = None,
                 measure: List[str] = None):
        """Calculate the forecast accuracy.

        :param real: real future observations, array-like of shape (forecast_horizon, m)
        :param pred: forecast values, array-like of shape (forecast_horizon, m)
        :param levels: which levels, None means all levels.
        :param hist: historical time series.
        :param measure: list of measures, e.g., ['mase'], ['mse', 'mase'].
        :return: forecast accuracy of reconciled forecasts.
        """
        pass
        # assert self.check_hierarchy(real), f"True observations should be of shape (h, m)"
        # assert self.check_hierarchy(pred), f"Forecast values should be of shape (h, m)"
        # assert real.shape[0] == pred.shape[0], \
        #     f" {real.shape} True observations have different length with {real.shape} forecasts"
        # if measure is None:
        #     measure = ['mase', 'mape', 'rmse']
        # if 'mase' in measure or 'smape' in measure or 'rmsse' in measure:
        #     assert hist is not None
        # if hist is not None:
        #     assert self.check_hierarchy(hist), "History observations should be of shape(T, m)"
        #     hist = self.aggregate_ts(hist, levels=levels).T
        # else:
        #     hist = [None] * self.s_mat.shape[0]
        # agg_true = self.aggregate_ts(real, levels=levels).T
        # agg_pred = self.aggregate_ts(pred, levels=levels).T
        # accs = pd.DataFrame()
        # for me in measure:
        #     try:
        #         accs[me] = np.array([getattr(_accuracy, me)(agg_true[i], agg_pred[i], hist[i], self.period)
        #                     for i in range(agg_true.shape[0])])
        #     except AttributeError:
        #         print('This forecasting measure is not supported!')
        # accs.index = self.node_name[np.isin(self.node_level, levels)] if levels is not None else self.node_name
        # return accs


class TemporalHierarchy:
    """Class for temporal hierarchy, constructed by multiple temporal aggregations.

    **Attributes**

        .. py:attribute:: s_mat

        summing matrix

        .. py:attribute:: node_level

        array indicating the corresponding level name of each node

        .. py:attribute:: node_name

        array indicating the name of each node. It is corresponded with row of summing matrix.

        .. py:attribute:: level_name

        array indicating the name of each level
    """

    def __init__(self, s_mat, node_level, names, period, level_name):
        self.s_mat = s_mat
        self.period = period

        self.node_level = node_level
        self.node_name = names
        self.level_name = level_name

    @classmethod
    def new(cls,
            agg_periods: List[int],
            forecast_frequency: int) -> "TemporalHierarchy":
        """TemporalHierarchy constructor.

        :param agg_periods: periods of the aggregation levels, referring to how many periods in the bottom-level are \
        aggregated. To ensure a reasonable hierarchy, each element in :code:`agg_periods` should be a factor of the  \
        the max agg_period. For example, possible aggregation periods for monthly time series could be 2 (two \
        months ), 3 (a quarter), 4 (four months), 6(half year), 12 (a year).
        :param forecast_frequency: frequency of the bottom level series, corresponding to the aggregation level \
        :code:`1` in agg_periods
        """
        agg_periods.sort(reverse=True)
        period = agg_periods[0]

        for agg_period in agg_periods:
            assert period % agg_period == 0, f"agg_period should be a factor of period, {period} % {agg_period} != 0"

        if 1 not in agg_periods:
            agg_periods.append(1)
        s_matrix = np.concatenate([np.kron(np.eye(period // agg_period, dtype=int), np.tile(1, (1, agg_period)))
                                   for agg_period in agg_periods])
        level_names = [f'agg_{agg_period}' for agg_period in agg_periods]
        names = []
        for agg_period in agg_periods:
            names.extend([f'agg_{agg_period}_{i+1}' for i in range(period // agg_period)])
        node_level = []
        for agg_period in agg_periods:
            node_level.extend([f'agg_{agg_period}'] * (period // agg_period))

        return cls(s_matrix, node_level, names, forecast_frequency, level_names)

    def aggregate_ts(self, bts: np.ndarray, levels=None) -> dict:
        """aggregate time series

        :param bts: should be a univariate time series
        :param levels: which level to be aggregated, should be one of the level_name
        :return: a dict whose keys are level_name and value are temporally aggregated time series.
        """

        assert len(bts.shape) == 1, "the function can only be applied to univariate time series"

        if levels is not None:
            if not isinstance(levels, str):
                levels = list(levels)
            assert np.isin(levels, self.level_name).all(), "the levels should be in level_names"

        n, m = self.s_mat.shape
        k = len(bts) // m
        bts = bts[-(k*m):].reshape((k, m))
        ats = bts.dot(self.s_mat.T)
        ats_dict = self._temporal_array2dict(ats)

        if levels is not None:
            return {key: ats_dict[key] for key in levels}
        return ats_dict

    def accuracy(self, real: np.ndarray, pred: dict, hist: np.ndarray = None, measure=None):
        """function to compute forecast accuracy

        :param real: univariate time series at forecast periods
        :param pred: dict containing forecasts, either reconciled or base
        :param hist: univariate historical time series
        :param measure: measures
        :return:
        """

        if measure is None:
            measure = ['mase', 'mape', 'rmse']
        if 'mase' in measure or 'smape' in measure or 'rmsse' in measure:
            assert hist is not None

        agg_true = self.aggregate_ts(real)

        if hist is not None:
            hist = self.aggregate_ts(hist)

        accs = dict()
        for me in measure:
            try:
                accs[me] = np.array([getattr(_accuracy, me)(agg_true[key], pred[key], hist[key],
                                                            self.period // int(key.split('_')[1]) if self.period // int(key.split('_')[1]) > 1 else 1)
                                     for key in pred])
            except AttributeError:
                print('This forecasting measure is not supported!')
        output = pd.DataFrame(accs)
        output.index = list(pred.keys())
        return output

    def _temporal_dict2array(self, dt: Dict):
        freqs = [self.period // int(i.split('_')[1]) for i in self.level_name]
        tmp = []
        for i, series in enumerate(self.level_name):
            tmp.append(dt[series].reshape((-1, freqs[i])))
        return np.concatenate(tmp, axis=1)

    def _temporal_array2dict(self, array: np.ndarray):
        ats_dict = dict()
        for l in self.level_name:
            ats_dict[l] = array[:, np.isin(self.node_level, l)].reshape((-1,))
        return ats_dict


if __name__ == "__main__":
    df = pd.read_csv("pyhts/data/Tourism.csv", index_col=0)
    cross_sectional = Hierarchy.new(df, structures=["state", "region", "city"])
    temporal = Hierarchy.new(agg_periods=[1, 4, 12])
    cross_temporal = Hierarchy.new(df, structures=["state", "region", "city"], agg_periods=[1, 4, 12])

    bts = df.loc[:, [str(i) for i in range(240)]].values.T
    a = cross_sectional.aggregate_ts(bts)
    b = temporal.aggregate_ts(bts[:,0])
    c = cross_temporal.aggregate_ts(bts)