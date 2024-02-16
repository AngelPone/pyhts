import numpy as np
from numpy import ndarray
import pandas as pd
from typing import Union, List, Optional, Dict, Iterable, Tuple
from itertools import combinations, product
from scipy.sparse import block_diag, csr_matrix, identity, vstack, kron, diags, hstack
import scipy.linalg as lg
from abc import ABC, abstractmethod

__all__ = ["CrossSectionalHierarchy", "TemporalHierarchy"]


def _lamb_estimate(x: ndarray) -> float:
    """Estimate :math`\\lambda` used in :ref:`shrinkage` estimator of mint method.

    :param x: in-sample 1-step-ahead forecast error.
    :return: :math`\\lambda`.
    """
    T = x.shape[0]
    covm = x.T.dot(x) / T
    xs = x / np.sqrt(np.diag(covm))
    corm = xs.T.dot(xs) / T
    np.fill_diagonal(corm, 0)
    d = np.sum(np.square(corm))
    xs2 = np.square(xs)
    v = 1 / (T * (T - 1)) * (xs2.T.dot(xs2) - 1 / T * np.square(xs.T.dot(xs)))
    np.fill_diagonal(v, 0)
    lamb = np.max(np.min([np.sum(v) / d, 1]), 0)
    return lamb


def get_all_combinations(lst: List[Tuple[str, ...]]) -> Iterable[List[str]]:
    n = len(lst)
    for r in range(1, n + 1):
        for comb in combinations(lst, r):
            if len(comb) == 1:
                for i in range(len(comb[0])):
                    yield list(comb[0][: (i + 1)])
            else:
                idxs: List[List[int]] = [list(range(len(i))) for i in comb]
                for i in product(*idxs):
                    output = []
                    for j, k in zip(comb, i):
                        output.extend(j[: (k + 1)])
                    yield output


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
        self, input: Union[ndarray, Dict[int, ndarray]], type: str, message: str
    ):
        pass

    def _check_aggregate_ts(self, bts: ndarray):
        self._check_input(bts, type="observation", message="bts")

    def _construct_u_mat(self, immutable_set: Optional[List[int]] = None) -> csr_matrix:
        """construct U matrix used in solution.

        :param immutable_set:
        :return:
        """
        s_mat = self.s_mat
        n, m = s_mat.shape
        u1 = identity(n - m, dtype="int8")
        u2 = 0 - s_mat[: (n - m), :]
        u_mat = hstack([u1, u2])
        if immutable_set:
            u_up = csr_matrix(identity(n, dtype="int8", format="csr"))[immutable_set]
            return hstack([u_up, u_mat]).T
        return csr_matrix(u_mat.T)

    def compute_W(self, residuals, method) -> Union[ndarray, csr_matrix]:
        method = getattr(self, f"_W_{method}")
        return method(residuals)

    def _W_sample(self, residuals: ndarray) -> ndarray:
        return residuals.T.dot(residuals) / residuals.shape[0]

    def _W_shrinkage(self, residuals: ndarray) -> ndarray:
        sample = residuals.T.dot(residuals) / residuals.shape[0]
        lamb = _lamb_estimate(residuals)
        return lamb * np.diag(np.diag(sample)) + (1 - lamb) * sample

    def _W_ols(self, residuals) -> csr_matrix:
        return csr_matrix(identity(self.n, dtype="int8", format="csr"))

    def _W_structural(self, residuals) -> csr_matrix:
        return csr_matrix(
            diags(
                self.s_mat.toarray().dot(np.array([[1]] * self.m)).reshape(-1),
                format="csr",
            )
        )

    # TODO: support for immutable_set
    def compute_g_mat(self, W) -> ndarray:
        """Compute G matrix given the weight_matrix."""

        m = self.m
        n = self.n
        # if immutable_set:
        #     immutable_set = list(immutable_set)
        #     k = len(immutable_set)
        #     assert (
        #         k <= self.m
        #     ), f"The number of immutable series can not be bigger than the number of bottom-level series {self.m}."
        u = self._construct_u_mat(immutable_set=None)
        J = csr_matrix(hstack([csr_matrix((m, n - m)), identity(m, format="csr")]))
        v = csr_matrix((n - m, n))
        # if immutable_set:
        #     v = hstack([identity(n)[immutable_set], v])
        if isinstance(W, csr_matrix):
            target = u.T.dot(W).dot(u).toarray()
        else:
            target = u.toarray().T.dot(W).dot(u.toarray())
        x, lower = lg.cho_factor(target)
        inv_dot = lg.cho_solve((x, lower), (u.T - v).toarray())
        if isinstance(W, csr_matrix):
            return J.toarray() - J.dot(W).dot(u).toarray().dot(inv_dot)
        else:
            return J.toarray() - J.toarray().dot(W).dot(u.toarray()).dot(inv_dot)


class TemporalHierarchy(Hierarchy):
    """Class for temporal hierarchy"""

    def __init__(self, s_mat: csr_matrix, indices: List[int]) -> None:
        """
        :param s_mat: summing matrix of the hierarchy.
        :param indices: list of integers representing the aggregation periods.
        :return: TemporalHierarchy object.
        """

        self._s_mat = s_mat
        self.indices = indices

    @property
    def s_mat(self) -> csr_matrix:
        """summing matrix of the hierarchy"""
        return self._s_mat

    @property
    def n(self) -> int:
        """Number of time series in the temporal hierarchy"""
        max_indice = max(self.indices)
        return sum([max_indice // i for i in self.indices])

    @property
    def m(self) -> int:
        """number of bottom level series in the temporal hierarchy."""
        return max(self.indices)

    @classmethod
    def new(cls, agg_periods: Iterable[int]) -> "TemporalHierarchy":
        """Constructor of TemporalHierarchy based on aggregation periods.

        :param: agg_periods: list of integers representing the aggregation periods.
        :return: TemporalHierarchy object.
        """

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
        return cls(csr_matrix(vstack(rows)), unique_periods)

    def aggregate_ts(
        self, bts: ndarray, agg_periods: Optional[List[int]] = None
    ) -> Dict[int, ndarray]:
        """Aggregate the bottom level time series to the higher levels.

        :param bts: univariate time series of shape (T,).
        :param agg_periods: if not None, only return aggregated time series for the
            given aggregation periods.
        :return: aggregated time series. Dict. keys are the aggregation periods.
            values are the aggregated time series of shape (T//agg_period,).
        """
        self._check_aggregate_ts(bts)
        output: Dict[int, ndarray] = {}

        if len(bts) % self.m != 0:
            truncted_len = len(bts) - len(bts) % self.m
            bts = bts[-truncted_len:]
            Warning(
                f"The length of the time series is not multiple of the maximum period \
                and the first {len(bts) % self.m} is truncated."
            )
        if agg_periods is None:
            agg_periods = self.indices
        for key in agg_periods:
            assert (
                key in self.indices
            ), f"{key} is not in the aggregation periods {self.indices}"
            output[key] = self._temporal_aggregate(bts, key)
        return output

    def _temporal_aggregate(self, bts: ndarray, agg_period: int) -> ndarray:
        return bts.reshape((-1, agg_period)).sum(axis=1)

    def _check_input(
        self, input: Union[ndarray, Dict[int, ndarray]], type: str, message: str
    ):
        if type == "observation":
            assert isinstance(input, ndarray)
            assert (
                len(input.shape) == 1
            ), "TemporalHierarchy can be only applied to univariate time series"
        if type == "forecast":
            assert isinstance(
                input, dict
            ), f"{message} should be dict for TemporalHierarchy"
            for key in self.indices:
                assert key in input.keys(), f"{key} not in {message}"

    def _input_to_mat(self, input: Dict[int, ndarray]) -> ndarray:
        series_list: List[ndarray] = []
        for key in self.indices:
            series_list.insert(0, input[key].reshape((-1, max(self.indices) // key)))
        return np.concatenate(series_list, axis=1)

    def _W_wlsv(self, residuals: ndarray) -> csr_matrix:
        vars = []
        current_k = 0
        for key in self.indices[::-1]:
            k = max(self.indices) // key
            current_res = residuals[:, current_k : (current_k + k)].reshape((-1,))
            var = current_res.T.dot(current_res) / current_res.shape[0]
            vars.extend([var] * k)
            current_k += k
        return csr_matrix(diags(vars, format="csr"))

    def _W_wlsh(self, residuals: ndarray) -> csr_matrix:
        vars = np.apply_along_axis(lambda x: x.T.dot(x) / x.shape[0], 0, residuals)
        return csr_matrix(diags(vars, format="csr"))

    def _check_reconcile(
        self,
        fcasts: Union[ndarray, Dict[int, ndarray]],
        method: str,
        residuals: Optional[Union[ndarray, Dict[int, ndarray]]],
    ):
        self._check_input(fcasts, type="fcasts", message="fcasts")
        assert method in ["ols", "structural", "wlsv", "wlsh", "shrinkage"]
        if method in ["wlsv", "wlsh"]:
            assert residuals is not None
            self._check_input(residuals, type="residuals", message="residuals")

    def reconcile(
        self,
        fcasts: Dict[int, ndarray],
        method: str,
        residuals: Optional[Dict[int, ndarray]] = None,
    ) -> ndarray:
        """Reconciliation method.

        :param fcasts: dict of forecasts. keys are the aggregation periods.
            values are the forecasts of shape (h//agg_period,).
        :param method: method of reconciliation. "ols", "structural", "wlsv", "wlsh", "shrinkage".
           - **wlsv**: series variance scaling
           - **structural**: structural scaling
           - **ols**: no scaling
           - **wlsh**: hierarchy variance scaling
        :param residuals: dict of residuals used for covariance estimation.
            keys are the aggregation periods.
            values are the residuals of shape (T//agg_period,).
            Required for "wlsv", "wlsh", "sample", "shrinkage".
        :return: reconciled forecasts of shape (h,).
        """
        self._check_reconcile(fcasts, method, residuals)
        residuals_mat = self._input_to_mat(residuals)
        W = self.compute_W(residuals_mat, method)
        G = self.compute_g_mat(W)
        fcasts_mat = self._input_to_mat(fcasts)
        return G.dot(fcasts_mat.T).reshape((-1,))


class CrossSectionalHierarchy(Hierarchy):
    """Class for cross-sectional hierarchy."""

    def __init__(self, s_mat: csr_matrix, indices: pd.DataFrame) -> None:
        """

        :param s_mat: summing matrix of the hierarchy.
        :param indices: DataFrame contains keys for determining cross-section structural.
            Each row responsents a cross-sectional time series and corresponds to a row in the summing matrix.
            Each col represents a attribute which determines the cross-sectional structure.
            It is automatically constructed by the new method.
            Unless you are familiar with the structure of indices, you should not
            construct it by yourself. Use the new method instead.
            The np.nan in the indices indicates "aggregation".
        """

        self._s_mat = s_mat
        self.indices = indices

    @property
    def s_mat(self) -> csr_matrix:
        """summing matrix of the hierarchy"""
        return self._s_mat

    @property
    def n(self) -> int:
        """Number of time series in the cross-sectional hierarchy"""
        return self.s_mat.shape[0]

    @property
    def m(self) -> int:
        """number of bottom level series in the cross-sectional hierarchy."""
        return self.s_mat.shape[1]

    def _check_input(
        self, input: Union[ndarray, Dict[int, ndarray]], type: str, message: str
    ):
        assert isinstance(input, ndarray), f"{message} should be an array"
        assert len(input.shape) == 2, f"{message} should be 2D"
        if type == "observation":
            assert input.shape[1] == self.m, f"{message} should be of  (T, {self.m})"
        if type == "forecast":
            assert (
                input.shape[1] == self.n
            ), f"{message} should be of shape (T, {self.n})"

    @classmethod
    def new(
        cls, structures: pd.DataFrame, trees: List[Tuple[str, ...]]
    ) -> "CrossSectionalHierarchy":
        """Constructor of CrossSectionalHierarchy

        :param structures: DataFrame contains keys for determining cross-section structural. Each row responsents a
            cross-sectional time series in the bottom level. Each col represents a attribute which determines the
            cross-sectional structure.
        :param trees: list of tuples. Each tuple represents top-down a tree structure.
            A tree structure means that a series can only have one parent series. For example,
            we have a (Category, SubCategory) sales hierarchy, the tree structure is [("Category", "SubCategory")].
            Then, the series "Apple" in SubCategory can only have one parent series "Fruit" in the Category.
            Each string in the tuple represents a level of the tree and a column in the structures.
            Multiple trees form a grouped cross-sectional hierarchy.
        :return: CrossSectionalHierarchy object.

        **Examples**

        The following example constructs a cross-sectional hierarchy with 4 bottom-level (SubCategory level) series,
        2 middle-level (Category level, Fruit and Meat) series and 1 top-level series.

            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...         "Category": ["Fruit", "Fruit", "Meat", "Meat"],
            ...         "SubCategory": ["Apple", "Orange", "Beef", "Pork"],
            ...     })
            >>> ht = CrossSectionalHierarchy.new(df, trees = [("Category", "SubCategory")])
            >>> ht.s_mat.toarray()
            array([[1, 1, 1, 1]
                   [1, 1, 0, 0],
                   [0, 0, 1, 1],
                   [1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1],
                   [1, 0, 0, 0]])
            >>> ht.m
            4
            >>> ht.n
            7

        The following example constructs a grouped cross-sectional hierarchy formed by two trees.
        The first tree is ("Category", "SubCategory") and the second tree is ("Region",).

            >>> df1 = pd.DataFrame({
            ...         "Category": ["Fruit", "Fruit", "Meat", "Meat"],
            ...         "SubCategory": ["Apple", "Orange", "Beef", "Pork"],
            ...     })
            >>> df2 = df1.copy()
            >>> df1["Region"] = "North"
            >>> df2["Region"] = "South"
            >>> df = pd.concat([df1, df2], axis=0, ignore_index=True)
            >>> ht = CrossSectionalHierarchy.new(df, trees = [("Category", "SubCategory"), ("Region",)])
            >>> ht.s_mat.toarray()
            array([[1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 0, 0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0, 0, 1, 1],
                   [1, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 0, 0, 0, 0],
                   [0 ,0, 0, 0, 1, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 1],
                   [1, 0, 0, 0, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0, 1, 0, 0],
                   [0, 0, 1, 0, 0, 0, 1, 0],
                   [0, 0, 0, 1, 0, 0, 0, 1],
                   [1, 1, 1, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 1, 1, 1],
                   [1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1]])
            >>> ht.m
            8
            >>> ht.n
            21
        """
        all_columns = []
        for tree in trees:
            all_columns.extend(tree)
        if len(all_columns) != sum([len(tree) for tree in trees]):
            raise ValueError("duplicated columns in trees")
        if structures[all_columns].drop_duplicates().shape[0] != structures.shape[0]:
            raise ValueError("duplicated rows in structures")
        for tree in trees:
            assert all(
                [i in structures.columns for i in tree]
            ), f"column not found in structures"
            if len(tree) == 1:
                continue
            df = structures.loc[:, tree].drop_duplicates().reset_index(drop=True)
            for i in range(1, len(tree)):
                assert (
                    df.loc[:, tree[: (i + 1)]]
                    .drop_duplicates()
                    .groupby(tree[i])
                    .count()
                    .max()
                    == 1
                ).all(), f"tree {tree} is not a tree"
        indices = pd.DataFrame()
        indptr = np.array([0], dtype="int")
        indices_s = np.zeros(0, dtype="int")
        current_row = 0
        for comb in get_all_combinations(trees):
            if (
                structures.loc[:, pd.Index(comb)].drop_duplicates().shape[0]
                == structures.shape[0]
            ):
                continue
            keys = structures.groupby(list(comb))
            keys_dict = []
            for key, idx in keys.indices.items():
                indptr = np.append(indptr, indptr[-1] + len(idx))
                current_row += 1
                indices_s = np.concatenate([indices_s, idx])
                keys_dict.append(key)
            tmp_df = pd.DataFrame(keys_dict, columns=pd.Index(list(comb)))
            indices = pd.concat([indices, tmp_df], axis=0, ignore_index=True)
        s_mat = csr_matrix(
            (np.array([1] * len(indices_s), dtype="int"), indices_s, indptr),
            shape=(current_row, structures.shape[0]),
        )
        total_s = csr_matrix(np.array([1] * structures.shape[0], dtype="int"))
        s_mat = csr_matrix(
            vstack(
                [
                    total_s,
                    s_mat,
                    identity(structures.shape[0], dtype="int", format="csr"),
                ]
            )
        )
        indices = pd.concat(
            [
                pd.DataFrame({key: [np.nan] for key in indices.columns}),
                indices,
                structures.copy(),
            ],
            axis=0,
            ignore_index=True,
        )
        return cls(s_mat, indices)

    def aggregate_ts(
        self, bts: ndarray, indices: Optional[List[int]] = None
    ) -> ndarray:
        """Aggregate the bottom level time series to the higher levels.

        :param bts: bottom-level time series of shape (T, m).
        :param indices: list of integers representing the row absolute indices of
            self.indices indicates which series will be calculated.
            If None, all series are calculated.
        """
        self._check_aggregate_ts(bts)
        if indices is None:
            indices = list(range(self.n))
        return bts.dot(self.s_mat[indices, :].toarray().T)

    def _check_reconcile(
        self,
        fcasts: ndarray,
        method: str,
        residuals: Optional[ndarray],
    ):
        self._check_input(fcasts, type="forecast", message="fcasts")
        assert method in [
            # "bu",
            # "tdhp",
            # "tdfp",
            "ols",
            "structural",
            "wlsv",
            "shrinkage",
            "sample",
        ]
        if method in ["wlsv", "sample", "shrinkage"]:
            assert residuals is not None
            self._check_input(residuals, type="residuals", message="residuals")

    def _W_wlsv(self, residuals: ndarray) -> csr_matrix:
        vars = np.apply_along_axis(lambda x: x.T.dot(x) / x.shape[0], 0, residuals)
        return csr_matrix(diags(vars, format="csr"))

    def reconcile(
        self,
        fcasts: ndarray,
        method: str,
        residuals: Optional[ndarray] = None,
    ) -> ndarray:
        """Reconciliation method.

        :param fcasts: forecasts of shape (h, n).
        :param method: method of reconciliation. "bu", "tdfp", "tdhp", "ols", "structural", "wlsv","shrinkage".
        :param residuals: residuals used for covariance estimation. of shape (T, n).
            Required for "variance", "sample", "shrinkage" method.
        :return: reconciled forecasts of shape (h, m).
        """
        self._check_reconcile(fcasts, method, residuals)
        W = self.compute_W(residuals, method)
        G = self.compute_g_mat(W)
        return G.dot(fcasts.T).T


class CrossTemporalHierarchy(Hierarchy):
    """Class for CrossTemporalHierarchy"""

    def __init__(
        self,
        s_cs: csr_matrix,
        s_tem: csr_matrix,
        indices_cs: pd.DataFrame,
        indices_temp: List[int],
    ) -> None:

        self.s_cs = s_cs
        self.s_temp = s_tem
        self.indices_cs = indices_cs
        self.indices_temp = indices_temp

    @property
    def s_mat(self) -> csr_matrix:
        return csr_matrix(kron(self.s_cs, self.s_temp, format="csr"))

    @property
    def n(self) -> int:
        return self.s_cs.shape[0] * self.s_temp.shape[0]

    @property
    def m(self) -> int:
        return self.s_cs.shape[1] * self.s_temp.shape[1]

    @classmethod
    def new(
        cls,
        structures: pd.DataFrame,
        trees: List[Tuple[str, ...]],
        agg_periods: Iterable[int],
    ) -> "CrossTemporalHierarchy":
        """Constructor

        :param structure: DataFrame contains keys for determining cross-section structural. Each row responsents a
            cross-sectional time series in the bottom level. Each col represents a attribute of the time series.
            The attributes are used to define the cross-sectional structure of the hierarchical time series.
        :param trees: list of tuples. Each tuple represents top-down a tree structure.
            See :ref:`CrossSectionalHierarchy.new` for more details.
        :param agg_periods: list of aggregation periods used to construct temporal hierarchy.
        :return: CrossTemporalHierarchy object.
        """
        ht_temp = TemporalHierarchy.new(agg_periods)
        ht_cs = CrossSectionalHierarchy.new(structures, trees)
        return cls(ht_cs.s_mat, ht_temp.s_mat, ht_cs.indices, ht_temp.indices)

    def _check_input(
        self, input: Union[ndarray, Dict[int, ndarray]], type: str, message: str
    ):
        assert input is not None, f"{message} should not be None"
        if type == "forecast":
            assert isinstance(input, dict), f"{message} should be dict"
            for key in self.indices_temp:
                assert key in input.keys(), f"{key} not in {message}"
                assert isinstance(input[key], ndarray), f"{message} should be ndarray"
                assert len(input[key].shape) == 2, f"{message} should be 2D"
                assert (
                    input[key].shape[1] == self.s_cs.shape[0]
                ), f"{message} should be of shape (-1, {self.s_cs.shape[0]})"
        if type == "observation":
            assert isinstance(input, ndarray), f"{message} should be ndarray"
            assert len(input.shape) == 2, f"{message} should be 2D"
            assert (
                input.shape[1] == self.s_cs.shape[1]
            ), f"{message} should be of shape (-1, {self.s_cs.shape[1]})"

    def _temporal_aggregate(self, bts: ndarray, agg_period: int) -> ndarray:
        return bts.reshape((-1, agg_period, bts.shape[1])).sum(axis=1)

    def aggregate_ts(
        self, bts: ndarray, indices: Optional[Dict[int, List[int]]] = None
    ) -> Dict[int, ndarray]:
        """Aggregate the bottom level time series to the higher levels.

        :param bts: bottom-level time series at the highest frequency.
        :param indices: dict. keys are the aggregation periods. values are the row absolute indices of
            self.indices_cs indicates which series will be calculated at the given aggregation period.
        :return: aggregated time series. Dict. keys are the aggregation periods.

        **Examples**

            >>> ht.aggregate_ts(bts, {1: [0, 1], 2: [0, 1, 2]})

        This example will obtain a dict, keys are 1 and 2. Values are [0, 1] aggregated series
        at the highest frequency (1) and [0, 1, 2] aggregated series at the second highest frequency (2).
        """
        self._check_aggregate_ts(bts)
        output = {}
        if indices is None:
            indices = {
                key: list(range(self.s_cs.shape[0])) for key in self.indices_temp
            }
        for key in indices:
            cs_bt = self._temporal_aggregate(bts, key)
            output[key] = cs_bt.dot(self.s_cs[indices[key], :].toarray().T)
        return output


#     def _check_input(self, input, type="forecast", message="forecast"):
#         assert input is not None, f"{message} should not be None"
#         if self.type == "cs":
#             assert isinstance(
#                 input, ndarray
#             ), "input should be an array for cross-sectional hierarchy"
#             assert len(input.shape) == 2, "array should be 2D"
#             if type == "forecast":
#                 assert input.shape[1] == self.n, f"should have {self.n} forecasts"
#             if type == "observation":
#                 assert (
#                     input.shape[1] == self.m
#                 ), f"should have {self.m} bottom-level series"
#             return
#         else:
#             assert self.indices_tmp is not None, "indices_tmp should not be None"
#             if type == "observation":
#                 assert isinstance(input, ndarray), "input should be vector"
#                 if self.type == "te":
#                     assert len(input.shape) == 1, "array should be 1D"
#                 else:
#                     assert len(input.shape) == 2, "array should be 2D"
#             if type == "forecast":
#                 assert isinstance(
#                     input, dict
#                 ), "input should be a dict for temporal hierarchy"
#                 assert set(input.keys()) == set(
#                     self.indices_tmp
#                 ), f"should have forecasts for agg_periods {self.indices_tmp.unique()}"
#                 length = input[self.k].shape[0]
#                 for key, value in input.items():
#                     assert isinstance(value, ndarray), "value should be an vector"
#                     assert len(value.shape) == 1, "array should be 1D"
#                     assert value.shape[0] == length * (
#                         self.k // key
#                     ), f"should have {length * (self.k // key)} forecasts for agg_period {key}"

#     def _index_cs(self, indices: pd.DataFrame) -> List:
#         assert self.indices_cs is not None, "indices_cs should not be None"
#         bottom_idx = self.indices_cs.dropna(axis=0).reset_index(drop=True)
#         bottom_idx["_idx"] = bottom_idx.index
#         row_indices: List[int] = []
#         for row in range(indices.shape[0]):
#             row = indices.iloc[row : (row + 1),].dropna(axis=1)
#             row_idx = row.merge(bottom_idx, how="left", on=list(row.columns))[
#                 "_idx"
#             ].values.tolist()
#             assert not np.isnan(row_idx).any(), f"{ dict(row) } not found"
#             row_indices.append(row_idx[0])
#         return row_indices

#     def _temporal_aggregate(self, bts: ndarray, agg_period: int) -> ndarray:
#         if len(bts.shape) == 1:
#             return bts.reshape((-1, agg_period)).sum(axis=1).reshape((-1, 1))
#         if len(bts.shape) == 2:
#             return bts.reshape((-1, agg_period, bts.shape[1])).sum(axis=1)
#         raise ValueError("bts should be 1D or 2D")

#     # TODO: test input to matrix with aggregate_ts
#     def _input_to_mat(self, input) -> ndarray:
#         if self.type == "cs":
#             return input
#         if self.type == "te":
#             assert self.indices_tmp is not None, "indices_tmp should not be None"
#             max_k = max(self.indices_tmp)
#             return np.concatenate(
#                 [input[key].reshape(-1, max_k // key) for key in self.indices_tmp],
#                 axis=1,
#             )
#         if self.type == "ct":
#             assert self.indices_tmp is not None, "indices_tmp should not be None"
#             n = self.n
#             max_k = max(self.indices_tmp)
#             return np.concatenate(
#                 [
#                     np.concatenate(
#                         [
#                             input[i][:, j].reshape(-1, max_k // i)
#                             for i in self.indices_tmp
#                         ],
#                         axis=1,
#                     )
#                     for j in range(n)
#                 ],
#                 axis=1,
#             )

#     def aggregate_ts(
#         self, bts: ndarray, indices: Optional[List[Dict]] = None
#     ) -> Union[ndarray, Dict[int, ndarray]]:
#         """Aggregate bottom-level time series.

#         :param bts: bottom-level time series, array-like of shape (T, m)
#         :param indices: indices of the series to aggregate. For example:
#         [{"Category": "Fruit", "SubCategory": "Apple"}, {"Category": "Meat"}] will aggregate the series of \
#         "Apple" subcategory and "Meat" category.
#         [{"Category": "Fruit", "agg_period": 12}] will aggregate the series of "Fruit" category at yearly level.
#         If agg_period is not specified, return most frequencist series, i.e., agg_period = 1
#         :return: y.
#         """
#         output = {}
#         self._check_input(bts, type="observation")
#         if self.type != "cs":
#             assert self.indices_tmp is not None, "indices_tmp should not be None"
#             max_k = max(self.indices_tmp)
#             time_window = bts.shape[0]
#             if time_window % max_k != 0:
#                 T_ = time_window // max_k * max_k
#                 bts = bts[(bts.shape[0] - T_) :,]
#                 Warning(
#                     f"the observations at the first {time_window - T_} timestamps are dropped"
#                 )
#         # filter series accorrding to indices
#         if indices is not None:
#             assert isinstance(indices, list), "indices should be a list of dict"
#             assert all(
#                 [isinstance(i, dict) for i in indices]
#             ), "each element in indices should be a dict"

#             indices_df = pd.DataFrame(indices)
#             # pure cross-sectional aggregation
#             if self.type == "cs" or (
#                 "agg_period" not in indices_df.columns and self.type == "ct"
#             ):
#                 match_idx = self._index_cs(indices_df)
#                 assert len(match_idx) > 0, "no series found"
#                 return np.stack([bts[:, i].sum(axis=1) for i in match_idx], axis=1)
#             # pure temporal aggregation
#             elif self.type == "te":
#                 assert (
#                     "agg_period" in indices_df.columns
#                 ), "agg_period should be specified"
#                 assert (
#                     indices_df.shape[1] == 1
#                 ), "non agg_period parameters are specified to aggregate a temporal hierarchy."
#                 assert (len(bts.shape) == 1) or (
#                     bts.shape[1] == 1
#                 ), "temporal hierarchy can only be applied to univariate time series."
#                 agg_periods: List[int] = indices_df["agg_period"].unique().tolist()
#                 return {i: self._temporal_aggregate(bts, i) for i in agg_periods}
#             else:
#                 assert self.indices_tmp is not None, "indices_tmp should not be None"
#                 if "agg_period" in indices_df.columns:
#                     indices_df = indices_df.fillna({"agg_period": 1})
#                 for agg_period in indices_df["agg_period"].unique():
#                     assert agg_period in set(
#                         self.indices_tmp
#                     ), f"agg_period {agg_period} not found"
#                     cs = indices_df[indices_df["agg_period"] == agg_period].drop(
#                         columns="agg_period"
#                     )
#                     if cs.shape[1] > 0:
#                         idx = self._index_cs(cs)
#                         agg_ts = np.stack([bts[:, i].sum(axis=1) for i in idx], axis=1)
#                         output[agg_period] = self._temporal_aggregate(
#                             agg_ts, agg_period
#                         )
#                     else:
#                         bts = bts.sum(axis=1)
#                         output[agg_period] = self._temporal_aggregate(bts, agg_period)

#                 return output
#         elif self.type == "cs":
#             return bts.dot(self.s_cs.toarray().T)
#         # temporal hierarchy and cross-temporal hierarchy
#         else:
#             if self.type == "ct":
#                 bts = bts.dot(self.s_cs.toarray().T)
#             for period in set(self.indices_tmp):
#                 output[period] = self._temporal_aggregate(bts, period)
#             return output

#     def reconcile(
#         self,
#         base_forecasts: Union[ndarray, Dict[int, ndarray]],
#         method: str,
#         cov_method: Optional[str] = None,
#         residuals: Optional[ndarray] = None,
#     ):
#         """ Point forecast reconciliation

#         :param base_forecasts: base forecasts, array of shape (h, n) for cross-sectional hierarchy, \
#             vector of shape (h, ) for temporal hierarchy, and dict of arrays of shape (h_k, n) \
#             for cross-temporal hierarchy, each value corresponds to a frequency.
#         :param method: method to reconcile forecasts, choices include "ols", "wls", "mint".
#         :param cov_method: method to calculate variance covariance matrix of base forecast errors, \
#             only used for "wls" and "mint" method, \
#             for "wls", choices include "structural", "variance", \
#             for "mint", choices include "shrinkage", "sample". \
#             We recommend using "shrinkage" for better performance.
#         :param residuals: residuals of in-sample base forecasts, array of shape (T, n) for \
#             cross-sectional hierarchy, vector of shape (T, ) for temporal hierarchy, and dict of arrays \
#             of shape (T_k, n) for cross-temporal hierarchy, each value corresponds to a frequency. \
#             It is required for "variance", "shrinkage" and "sample" cov_method.
#         """
#         # check inputs
#         assert method in [
#             "ols",
#             "wls",
#             "mint",
#         ], "method should be one of 'ols', 'wls', 'mint'"
#         self._check_input(base_forecasts, type="forecast", message="base forecasts")

#         S_mat = self.s_mat
#         if method == "ols":
#             W = identity(S_mat.shape[0])
#         elif method == "wls":
#             if cov_method == "structural":
#                 W = S_mat.dot(np.array([1] * S_mat.shape[0]).reshape((-1, 1)))
#             elif cov_method == "variance":
#                 self._check_input(residuals, type="forecast", message="residuals")
#                 residuals = self._input_to_mat(residuals)
#                 W = diags(1 / residuals.var(axis=0))
#             else:
#                 raise ValueError(
#                     "cov_method for wls should be one of 'structural', 'variance'"
#                 )
#         else:
#             self._check_input(residuals, type="forecast", message="residuals")
#             residuals = self._input_to_mat(residuals)
#             W = residuals.dot(residuals.T) / residuals.shape[0]
#             if cov_method == "shrinkage":
#                 lamb = _lamb_estimate(residuals)
#                 W = lamb * np.diag(np.diag(W)) + (1 - lamb) * W
#             elif cov_method == "sample":
#                 W = W
#             else:
#                 raise ValueError(
#                     "cov_method for mint should be one of 'shrinkage', 'sample'"
#                 )
#             W = np.linalg.inv(W)

#         G = self.compute_g_mat(W)
#         return G.dot(self._input_to_mat(base_forecasts).T).T

#     def _construct_u_mat(self, immutable_set: Optional[List[int]] = None):
#         """construct U matrix used in solution.

#         :param immutable_set:
#         :return:
#         """
#         s_mat = self.s_mat
#         n, m = s_mat.shape
#         u1 = identity(n - m)
#         u2 = 0 - s_mat[: (n - m), :]
#         u_mat = vstack([u1, u2])
#         if immutable_set:
#             u_up = identity(n)[immutable_set]
#             return hstack([u_up, u_mat]).T
#         return u_mat.T
