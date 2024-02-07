import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Iterable
from itertools import combinations
from scipy.sparse import block_diag, csr_matrix, identity, vstack, kron, diags, hstack
from _reconciliation import _lamb_estimate
import scipy.linalg as lg
from abc import ABC, abstractmethod

__all__ = ["Hierarchy"]


def get_all_combinations(lst):
    n = len(lst)
    for r in range(1, n):
        for comb in combinations(lst, r):
            yield comb


class Hierarchy(ABC):

    @property
    @abstractmethod
    def s_mat(self) -> csr_matrix:
        pass

    @property
    @abstractmethod
    def n(self) -> int:
        pass

    @property
    @abstractmethod
    def m(self) -> int:
        pass

    @abstractmethod
    def _check_input(
        self, input: Union[np.ndarray, Dict[int, np.ndarray]], type: str, message: str
    ):
        raise NotImplementedError

    def _check_reconcile(
        self,
        fcasts: Union[np.ndarray, Dict[int, np.ndarray]],
        method: str,
        cov_method: Optional[str],
        residuals: Optional[Union[np.ndarray, Dict[int, np.ndarray]]],
    ):
        self._check_input(fcasts, type="fcasts", message="fcasts")
        assert method in ["bu", "tdhp", "tdfp", "ols", "wls", "mint"]
        if method == "wls":
            assert cov_method in ["structural", "variance"]
            if method == "variance":
                assert residuals is not None
                self._check_input(residuals, "forecast", "residuals")
        if method == "mint":
            assert residuals is not None
            self._check_input(residuals, "forecast", "residuals")
            assert cov_method in ["shrinkage", "sample"]

    def _check_aggregate_ts(self, bts: np.ndarray):
        self._check_input(bts, type="observation", message="bts")

    def compute_W(
        self, residuals: Optional[np.ndarray], method: str, cov_method: Optional[str]
    ) -> Union[np.ndarray, csr_matrix]:
        if method == "ols":
            return csr_matrix(identity(self.n))
        elif method == "wls":
            if cov_method == "structural":
                return self.s_mat.dot(np.array([[1]] * self.n))
            else:
                assert residuals is not None
                return csr_matrix(diags(residuals.var(axis=0)))
        else:
            assert residuals is not None
            W = residuals.T.dot(residuals)
            if cov_method == "sample":
                return lg.inv(W)
            lamb = _lamb_estimate(residuals)
            W = lamb * np.diag(np.diag(W)) + (1 - lamb) * W
            return lg.inv(W)


class TemporalHierarchy(Hierarchy):
    def __init__(self, s_mat: csr_matrix, indices: List[int]) -> None:
        self._s_mat = s_mat
        self.indices = indices

    @property
    def s_mat(self) -> csr_matrix:
        return self._s_mat

    @property
    def n(self) -> int:
        max_indice = max(self.indices)
        return sum([max_indice // i for i in self.indices])

    @property
    def m(self) -> int:
        return max(self.indices)

    @classmethod
    def new(cls, agg_periods: Iterable[int]) -> "TemporalHierarchy":
        unique_periods = list(set(agg_periods))
        unique_periods.sort()
        assert 1 in unique_periods, "1 should be in agg_periods"
        assert (
            len(unique_periods) > 1
        ), "there is no need to construct temporal hierarchy"
        max_period = unique_periods[-1]
        rows: List[csr_matrix] = []
        for period in unique_periods:
            assert max_period % period == 0, f"{period} is not factor of {max_period}"
            blocks = [[1] * period] * (max_period // period)
            # call csr_matrix to be compatible with pyright
            rows.insert(0, csr_matrix(block_diag(blocks, format="csr", dtype=np.int8)))
        return cls(csr_matrix(hstack(rows)), unique_periods)

    def aggregate_ts(self, bts: np.ndarray) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        self._check_aggregate_ts(bts)
        output: Dict[int, np.ndarray] = {}
        for key in self.indices:
            output[key] = self._temporal_aggregate(bts, key)
        return output

    def _temporal_aggregate(self, bts: np.ndarray, agg_period: int) -> np.ndarray:
        return bts.reshape((-1, agg_period)).sum(axis=1).reshape((-1, 1))

    def _check_input(
        self, input: Union[np.ndarray, Dict[int, np.ndarray]], type: str, message: str
    ):
        if type == "observation":
            assert isinstance(input, np.ndarray)
            assert (
                len(input.shape) == 1
            ), "TemporalHierarchy can be only applied to univariate time series"
        if type == "forecast":
            assert isinstance(
                input, dict
            ), f"{message} should be dict for TemporalHierarchy"
            for key in self.indices:
                assert key in input.keys(), f"{key} not in {message}"

    def _input_to_mat(
        self, input: Optional[Dict[int, np.ndarray]]
    ) -> Optional[np.ndarray]:
        if input is None:
            return None
        series_list: List[np.ndarray] = []
        for key in self.indices:
            series_list.append(input[key].reshape((-1, max(self.indices) // key)))
        return np.concatenate(series_list, axis=1)

    def reconcile(
        self,
        fcasts: Dict[int, np.ndarray],
        method: str,
        cov_method: Optional[str],
        residuals: Optional[Dict[int, np.ndarray]],
    ) -> np.ndarray:
        self._check_reconcile(fcasts, method, cov_method, residuals)
        residuals_mat = self._input_to_mat(residuals)
        W = self.compute_W(residuals_mat, method, cov_method)


class CrossSectionalHierarchy(Hierarchy):
    """Class for a hierarchy structure.

    **Attributes**

    TODO
    """

    def __init__(
        self,
        s_cs: csr_matrix,
        s_temp: csr_matrix,
        indices_cs: Optional[pd.DataFrame],
        indices_tmp: Optional[List[int]],
    ):
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
        """summing matrix"""
        if self.indices_tmp is None:
            return self.s_cs
        elif self.indices_cs is None:
            return self.s_temp
        else:
            return kron(self.s_cs, self.s_temp)

    # TODO: support for excludes and includes
    @classmethod
    def new(
        cls,
        df: Optional[pd.DataFrame] = None,
        structures: Optional[List[str]] = None,
        # excludes: Optional[List[Dict]] = None,
        # includes: Optional[List[Dict]] = None,
        agg_periods: Optional[List[int]] = None,
    ) -> "Hierarchy":
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
        s_mat_cs: Optional[csr_matrix] = None
        s_te = None
        indices_cs = None
        indices_temp = None

        if df is not None:
            if structures is not None:
                df = df.loc[:, structures]
            df = df.drop_duplicates().reset_index(drop=True)
            columns = df.columns
            indices = pd.DataFrame()
            indptr = np.array([0])
            indices_s = np.zeros(0, dtype="int")
            current_row = 0
            existing_idxs = []
            for comb in get_all_combinations(columns):
                keys = df.groupby(list(comb))
                keys_dict = []
                for key, idx in keys.indices.items():
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
            s_mat = csr_matrix(
                (np.array([1] * len(indices_s), dtype="int"), indices_s, indptr),
                shape=(current_row, df.shape[0]),
            )
            total_s = csr_matrix(np.array([1] * df.shape[0], dtype="int"))
            s_mat_cs = csr_matrix(
                vstack([total_s, s_mat, csr_matrix(identity(df.shape[0], dtype="int"))])
            )
            indices_cs = pd.concat(
                [pd.DataFrame({key: [np.nan] for key in columns}), indices, df.copy()],
                axis=0,
                ignore_index=True,
            )

        if agg_periods is not None:
            assert len(agg_periods) > 1, "agg_periods should be a list of length > 1"
            assert 1 in agg_periods, "agg_periods should contain 1"
            agg_periods = list(set(agg_periods))
            agg_periods.sort(reverse=True)
            for agg_period in agg_periods:
                assert (
                    agg_periods[0] % agg_period == 0
                ), f"agg_period should be a factor of max agg_periods, \
                    {agg_periods[0]} % {agg_period} != 0"

                s_mat_tmp = np.zeros(
                    (agg_periods[0] // agg_period, agg_periods[0]), dtype="int"
                )
                for i in range(agg_periods[0] // agg_period):
                    s_mat_tmp[i, i * agg_period : (i + 1) * agg_period] = 1
                if s_te is None:
                    s_te = s_mat_tmp
                else:
                    s_te = np.concatenate([s_te, s_mat_tmp], axis=0)
            s_te = csr_matrix(s_te)
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
        if self.indices_tmp is None and self.indices_cs is not None:
            return "cs"
        elif self.indices_cs is None and self.indices_tmp is not None:
            return "te"
        else:
            assert self.indices_cs is not None and self.indices_tmp is not None
            return "ct"

    def _check_input(self, input, type="forecast", message="forecast"):
        assert input is not None, f"{message} should not be None"
        if self.type == "cs":
            assert isinstance(
                input, np.ndarray
            ), "input should be an array for cross-sectional hierarchy"
            assert len(input.shape) == 2, "array should be 2D"
            if type == "forecast":
                assert input.shape[1] == self.n, f"should have {self.n} forecasts"
            if type == "observation":
                assert (
                    input.shape[1] == self.m
                ), f"should have {self.m} bottom-level series"
            return
        else:
            assert self.indices_tmp is not None, "indices_tmp should not be None"
            if type == "observation":
                assert isinstance(input, np.ndarray), "input should be vector"
                if self.type == "te":
                    assert len(input.shape) == 1, "array should be 1D"
                else:
                    assert len(input.shape) == 2, "array should be 2D"
            if type == "forecast":
                assert isinstance(
                    input, dict
                ), "input should be a dict for temporal hierarchy"
                assert set(input.keys()) == set(
                    self.indices_tmp
                ), f"should have forecasts for agg_periods {self.indices_tmp.unique()}"
                length = input[self.k].shape[0]
                for key, value in input.items():
                    assert isinstance(value, np.ndarray), "value should be an vector"
                    assert len(value.shape) == 1, "array should be 1D"
                    assert value.shape[0] == length * (
                        self.k // key
                    ), f"should have {length * (self.k // key)} forecasts for agg_period {key}"

    def _index_cs(self, indices: pd.DataFrame) -> List:
        assert self.indices_cs is not None, "indices_cs should not be None"
        bottom_idx = self.indices_cs.dropna(axis=0).reset_index(drop=True)
        bottom_idx["_idx"] = bottom_idx.index
        row_indices: List[int] = []
        for row in range(indices.shape[0]):
            row = indices.iloc[row : (row + 1),].dropna(axis=1)
            row_idx = row.merge(bottom_idx, how="left", on=list(row.columns))[
                "_idx"
            ].values.tolist()
            assert not np.isnan(row_idx).any(), f"{ dict(row) } not found"
            row_indices.append(row_idx[0])
        return row_indices

    def _temporal_aggregate(self, bts: np.ndarray, agg_period: int) -> np.ndarray:
        if len(bts.shape) == 1:
            return bts.reshape((-1, agg_period)).sum(axis=1).reshape((-1, 1))
        if len(bts.shape) == 2:
            return bts.reshape((-1, agg_period, bts.shape[1])).sum(axis=1)
        raise ValueError("bts should be 1D or 2D")

    # TODO: test input to matrix with aggregate_ts
    def _input_to_mat(self, input) -> np.ndarray:
        if self.type == "cs":
            return input
        if self.type == "te":
            assert self.indices_tmp is not None, "indices_tmp should not be None"
            max_k = max(self.indices_tmp)
            return np.concatenate(
                [input[key].reshape(-1, max_k // key) for key in self.indices_tmp],
                axis=1,
            )
        if self.type == "ct":
            assert self.indices_tmp is not None, "indices_tmp should not be None"
            n = self.n
            max_k = max(self.indices_tmp)
            return np.concatenate(
                [
                    np.concatenate(
                        [
                            input[i][:, j].reshape(-1, max_k // i)
                            for i in self.indices_tmp
                        ],
                        axis=1,
                    )
                    for j in range(n)
                ],
                axis=1,
            )

    def aggregate_ts(
        self, bts: np.ndarray, indices: Optional[List[Dict]] = None
    ) -> Union[np.ndarray, Dict[int, np.ndarray]]:
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
        self._check_input(bts, type="observation")
        if self.type != "cs":
            assert self.indices_tmp is not None, "indices_tmp should not be None"
            max_k = max(self.indices_tmp)
            time_window = bts.shape[0]
            if time_window % max_k != 0:
                T_ = time_window // max_k * max_k
                bts = bts[(bts.shape[0] - T_) :,]
                Warning(
                    f"the observations at the first {time_window - T_} timestamps are dropped"
                )
        # filter series accorrding to indices
        if indices is not None:
            assert isinstance(indices, list), "indices should be a list of dict"
            assert all(
                [isinstance(i, dict) for i in indices]
            ), "each element in indices should be a dict"

            indices_df = pd.DataFrame(indices)
            # pure cross-sectional aggregation
            if self.type == "cs" or (
                "agg_period" not in indices_df.columns and self.type == "ct"
            ):
                match_idx = self._index_cs(indices_df)
                assert len(match_idx) > 0, "no series found"
                return np.stack([bts[:, i].sum(axis=1) for i in match_idx], axis=1)
            # pure temporal aggregation
            elif self.type == "te":
                assert (
                    "agg_period" in indices_df.columns
                ), "agg_period should be specified"
                assert (
                    indices_df.shape[1] == 1
                ), "non agg_period parameters are specified to aggregate a temporal hierarchy."
                assert (len(bts.shape) == 1) or (
                    bts.shape[1] == 1
                ), "temporal hierarchy can only be applied to univariate time series."
                agg_periods: List[int] = indices_df["agg_period"].unique().tolist()
                return {i: self._temporal_aggregate(bts, i) for i in agg_periods}
            else:
                assert self.indices_tmp is not None, "indices_tmp should not be None"
                if "agg_period" in indices_df.columns:
                    indices_df = indices_df.fillna({"agg_period": 1})
                for agg_period in indices_df["agg_period"].unique():
                    assert agg_period in set(
                        self.indices_tmp
                    ), f"agg_period {agg_period} not found"
                    cs = indices_df[indices_df["agg_period"] == agg_period].drop(
                        columns="agg_period"
                    )
                    if cs.shape[1] > 0:
                        idx = self._index_cs(cs)
                        agg_ts = np.stack([bts[:, i].sum(axis=1) for i in idx], axis=1)
                        output[agg_period] = self._temporal_aggregate(
                            agg_ts, agg_period
                        )
                    else:
                        bts = bts.sum(axis=1)
                        output[agg_period] = self._temporal_aggregate(bts, agg_period)

                return output
        elif self.type == "cs":
            return bts.dot(self.s_cs.toarray().T)
        # temporal hierarchy and cross-temporal hierarchy
        else:
            if self.type == "ct":
                bts = bts.dot(self.s_cs.toarray().T)
            for period in set(self.indices_tmp):
                output[period] = self._temporal_aggregate(bts, period)
            return output

    def reconcile(
        self,
        base_forecasts: Union[np.ndarray, Dict[int, np.ndarray]],
        method: str,
        cov_method: Optional[str] = None,
        residuals: Optional[np.ndarray] = None,
    ):
        """ Point forecast reconciliation

        :param base_forecasts: base forecasts, array of shape (h, n) for cross-sectional hierarchy, \
            vector of shape (h, ) for temporal hierarchy, and dict of arrays of shape (h_k, n) \
            for cross-temporal hierarchy, each value corresponds to a frequency.
        :param method: method to reconcile forecasts, choices include "ols", "wls", "mint".
        :param cov_method: method to calculate variance covariance matrix of base forecast errors, \
            only used for "wls" and "mint" method, \
            for "wls", choices include "structural", "variance", \
            for "mint", choices include "shrinkage", "sample". \
            We recommend using "shrinkage" for better performance.
        :param residuals: residuals of in-sample base forecasts, array of shape (T, n) for \
            cross-sectional hierarchy, vector of shape (T, ) for temporal hierarchy, and dict of arrays \
            of shape (T_k, n) for cross-temporal hierarchy, each value corresponds to a frequency. \
            It is required for "variance", "shrinkage" and "sample" cov_method.
        """
        # check inputs
        assert method in [
            "ols",
            "wls",
            "mint",
        ], "method should be one of 'ols', 'wls', 'mint'"
        self._check_input(base_forecasts, type="forecast", message="base forecasts")

        S_mat = self.s_mat
        if method == "ols":
            W = identity(S_mat.shape[0])
        elif method == "wls":
            if cov_method == "structural":
                W = S_mat.dot(np.array([1] * S_mat.shape[0]).reshape((-1, 1)))
            elif cov_method == "variance":
                self._check_input(residuals, type="forecast", message="residuals")
                residuals = self._input_to_mat(residuals)
                W = diags(1 / residuals.var(axis=0))
            else:
                raise ValueError(
                    "cov_method for wls should be one of 'structural', 'variance'"
                )
        else:
            self._check_input(residuals, type="forecast", message="residuals")
            residuals = self._input_to_mat(residuals)
            W = residuals.dot(residuals.T) / residuals.shape[0]
            if cov_method == "shrinkage":
                lamb = _lamb_estimate(residuals)
                W = lamb * np.diag(np.diag(W)) + (1 - lamb) * W
            elif cov_method == "sample":
                W = W
            else:
                raise ValueError(
                    "cov_method for mint should be one of 'shrinkage', 'sample'"
                )
            W = np.linalg.inv(W)

        G = self.compute_g_mat(W)
        return G.dot(self._input_to_mat(base_forecasts).T).T

    def _construct_u_mat(self, immutable_set: Optional[List[int]] = None):
        """construct U matrix used in solution.

        :param immutable_set:
        :return:
        """
        s_mat = self.s_mat
        n, m = s_mat.shape
        u1 = identity(n - m)
        u2 = 0 - s_mat[: (n - m), :]
        u_mat = vstack([u1, u2])
        if immutable_set:
            u_up = identity(n)[immutable_set]
            return hstack([u_up, u_mat]).T
        return u_mat.T

    # TODO: fix immutable_set
    # TODO: fix for sparse matrix
    def compute_g_mat(self, W, immutable_set: Optional[List[int]] = None):
        """Compute G matrix given the weight_matrix.

        :param hierarchy:
        :param weight_matrix:
        :param immutable_set: the subset of time series to be unchanged during reconciliation
        :return:
        """

        m = self.m * self.k
        n = self.n * self.K
        if immutable_set:
            immutable_set = list(immutable_set)
            k = len(immutable_set)
            assert (
                k <= self.m
            ), f"The number of immutable series can not be bigger than the number of bottom-level series {self.m}."
        u = self._construct_u_mat(immutable_set=immutable_set)
        J = vstack([csr_matrix((m, n - m)), identity(m)])
        v = csr_matrix((n - m, n))
        if immutable_set:
            v = hstack([identity(n)[immutable_set], v])
        target = u.T.dot(W).dot(u)
        x, lower = lg.cho_factor(target)
        inv_dot = lg.cho_solve((x, lower), (u.T - v))
        return J - J.dot(W).dot(u).dot(inv_dot)
