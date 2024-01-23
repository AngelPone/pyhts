import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict
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

    def __init__(self, s_cs, s_temp, indices_cs, indices_tmp):
        self.s_cs = s_cs
        self.s_temp = s_temp
        self.indices_cs = indices_cs
        self.indices_tmp = indices_tmp

        # cross-sectional numbers
        if s_cs is None:
            self.m = 1
            self.n = 1
        else:
            (self.n, self.m) = s_cs.shape
        if s_temp is None:
            self.K = 1
            self.k = 1
        else:
            (self.K, self.k) = s_temp.shape

    @property
    def s_mat(self):
        """ summing matrix """
        if self.indices_tmp is None:
            return self.s_cs
        elif self.indices_cs is None:
            return self.s_temp
        else:
            return kron(self.s_cs, self.s_temp)

    # TODO: support for excludes and includes
    @classmethod
    def new(cls, df: Optional[pd.DataFrame] = None,
            structures: Optional[List[str]] = None,
            # excludes: Optional[List[Dict]] = None,
            # includes: Optional[List[Dict]] = None,
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
        s_mat_cs = None
        s_te = None
        indices_cs = None
        indices_temp = None

        if df is not None:
            if structures is not None:
                df = df[structures]
            df = df.drop_duplicates().reset_index(drop=True)
            columns = df.columns
            indices = pd.DataFrame()
            indptr = np.array([0])
            indices_s = np.zeros(0, dtype='int')
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
                indices = pd.concat([indices, tmp_df], axis=0, ignore_index=True)
            s_mat = csr_array((np.array([1] * len(indices_s), dtype='int'), indices_s, indptr), 
                              shape=(current_row, df.shape[0]))
            total_s = csr_array(np.array([1] * df.shape[0], dtype='int'))
            s_mat_cs = vstack([total_s, s_mat, identity(df.shape[0], dtype='int')])
            indices_cs = pd.concat([pd.DataFrame({key: [np.nan] for key in columns}), indices, df.copy()], 
                                   axis=0, ignore_index=True)

        if agg_periods is not None:
            assert len(agg_periods) > 1, "agg_periods should be a list of length > 1"
            assert 1 in agg_periods, "agg_periods should contain 1"
            agg_periods = list(set(agg_periods))
            agg_periods.sort(reverse=True)
            for agg_period in agg_periods:
                assert agg_periods[0] % agg_period == 0, f"agg_period should be a factor of max agg_periods, \
                    {agg_periods[0]} % {agg_period} != 0"
                    
                s_mat_tmp = np.zeros((agg_periods[0] // agg_period, agg_periods[0]), dtype='int')
                for i in range(agg_periods[0] // agg_period):
                    s_mat_tmp[i, i * agg_period:(i + 1) * agg_period] = 1
                if s_te is None:
                    s_te = s_mat_tmp
                else:
                    s_te = np.concatenate([s_te, s_mat_tmp], axis=0)
            s_te = csr_array(s_te)
            indices_temp = []
            for agg_period in agg_periods:
                indices_temp.extend([agg_period] * (agg_periods[0] // agg_period))
        
        # if excludes is not None:
        #     excludes_cs = pd.DataFrame(excludes)
        #     if 'agg_period' in excludes.columns:
        #         excludes_cs = excludes[excludes['agg_period'].isna()]
        #     indices_cs['_idx'] = indices_cs.index
        #     assert excludes_cs.columns.isin(indices_cs.columns).all(), "excludes contains columns not in df"
        #     excludes_cs = excludes_cs.merge(pd.DataFrame(columns=indices_cs.columns)).merge(indices_cs, how="left", on=list(excludes_cs.columns))
        #     exclude_idx = excludes_cs['_idx'].values
        #     s_mat_cs = s_mat_cs[~exclude_idx,]
        #     indices_cs = indices_cs[~exclude_idx,].drop(columns="_idx")

        return cls(s_mat_cs, s_te, indices_cs, indices_temp)

    @property
    def type(self):
        """Type of the hierarchy, either cross-sectional, temporal or cross-temporal."""
        if self.indices_tmp is None:
            return "cs"
        elif self.indices_cs is None:
            return "te"
        else:
            return "ct"
    
    def _index_cs(self, indices: pd.DataFrame) -> np.ndarray:
        indices = indices.merge(pd.DataFrame(columns=self.indices_cs.columns), how="left")
        right = self.indices_cs.copy()
        right['_idx'] = right.index
        indices = indices.merge(right, how="left", on=list(self.indices_cs.columns))
        return indices["_idx"].dropna().values

    def _temporal_aggregate(self, bts: np.ndarray, agg_period: int) -> np.ndarray:
        if len(bts.shape) == 1:
            return bts.reshape((-1, agg_period)).sum(axis=1).reshape((-1, 1))
        if len(bts.shape) == 2:
            return bts.reshape((-1, agg_period, bts.shape[1])).sum(axis=1)

    def aggregate_ts(self, bts: np.ndarray, 
                     indices: [Dict] = None) -> Union[np.ndarray, List[Dict]]:
        """Aggregate bottom-level time series.

        :param bts: bottom-level time series, array-like of shape (T, m)
        :param indices: indices of the series to aggregate. For example: 
        [{"Category": "Fruit", "SubCategory": "Apple"}, {"Category": "Meat"}] will aggregate the series of \
        "Apple" subcategory and "Meat" category.
        [{"Category": "Fruit", "agg_period": 12}] will aggregate the series of "Fruit" category at yearly level.
        If agg_period is not specified, return most frequencist series, i.e., agg_period = 1
        :return: y.
        """
        output = {}
        if self.type == "te": 
            assert len(bts.shape) == 1, "temporal hierarchy can only be applied to univariate time series"
        if self.type != "cs":
            max_k = max(self.indices_tmp)
            time_window = bts.shape[0]
            if time_window % max_k != 0:
                T_ = time_window // max_k * max_k
                bts = bts[(bts.shape[0] - T_):,]
                Warning(f"the observations at the first {time_window - T_} timestamps are dropped")
        # filter series accorrding to indices
        if indices is not None:
            assert isinstance(indices, list), "indices should be a list of dict"
            assert all([isinstance(i, dict) for i in indices]), "each element in indices should be a dict"
            
            indices = pd.DataFrame(indices)
            # pure cross-sectional aggregation
            if self.type == "cs" or ("agg_period" not in indices.columns and self.type == "ct"):
                indices = self._index_cs(indices)
                assert len(indices) > 0, "no series found"
                return bts.dot(self.s_cs.toarray()[indices,].T)
            # pure temporal aggregation
            elif self.type == "te":
                assert "agg_period" in indices.columns, "agg_period should be specified"
                assert indices.shape[1] == 1, "non agg_period parameters are specified to aggregate a temporal hierarchy."
                assert (len(bts.shape) == 1) or (bts.shape[1] == 1), "temporal hierarchy can only be applied to univariate time series."
                agg_periods = indices['agg_period'].unique()
                return {i: self._temporal_aggregate(bts, i) for i in agg_periods}
            else:
                if "agg_period" in indices.columns:
                    indices = indices.fillna({"agg_period": 1})
                for agg_period in indices["agg_period"].unique():
                    assert agg_period in set(self.indices_tmp), f"agg_period {agg_period} not found"
                    cs = indices[indices["agg_period"] == agg_period].drop(columns="agg_period")
                    if cs.shape[1] > 0:
                        idx = self._index_cs(cs)
                        output[agg_period] = self._temporal_aggregate(bts.dot(self.s_cs.toarray()[idx,].T), 
                                                                      agg_period)
                    else:
                        bts = bts.sum(axis=1)
                        output[agg_period] = self._temporal_aggregate(bts, agg_period)
                    
                return output
        elif self.type == 'cs':
            return bts.dot(self.s_cs.toarray().T)
        # temporal hierarchy and cross-temporal hierarchy
        else:
            if self.type == "ct":
                bts = bts.dot(self.s_cs.toarray().T)
            for period in set(self.indices_tmp):
                output[period] = self._temporal_aggregate(bts, period)
            return output
