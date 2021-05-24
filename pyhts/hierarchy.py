import numpy as np
import pandas as pd
from copy import copy
from typing import Union, List
from . import accuracy


class Hierarchy:
    """Class for hierarchy structure.

    **Attributes**

        .. py:attribute:: s_mat

            summing matrix.
        .. py:attribute:: period

            frequency of time series.
        .. py:attribute:: node_name

            name of each node.
        .. py:attribute:: level_n

            number of levels.
        .. py:attribute:: node_level

            level of each node.
    """

    def __init__(self, s_mat, node_level, names, period):
        self.s_mat = s_mat.astype('int8')
        """ summing matrix. """
        self.node_level = np.array(node_level)
        self.level_n = max(node_level) + 1
        self.node_name = np.array(names)
        self.period = period

    @classmethod
    def from_node_list(cls, nodes, period: int = 1) -> "Hierarchy":
        """Constructs hierarchy from a list of lists, each sublist contains number of children nodes of all nodes in a
        hierarchical level. More specifically, the first sublist contains the number of children nodes of all nodes in
        Level0, e.g. the root node.(sum of all bottom level time series), and so on.

        **Examples**

            >>> from pyhts.hierarchy import Hierarchy
            >>> nodes = [[2], [2, 2]]
            >>> hierarchy = Hierarchy.from_node_list(nodes, 12)
            >>> hierarchy.s_mat
            array([[1, 1, 1, 1],
                   [1, 1, 0, 0],
                   [0, 0, 1, 1],
                   [1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=int8)

        :code:`[[2], [2, 2]]` defines a hierarchy that in which the root node have two sub-nodes, the fist
        sub-node contains two sub-nodes and the second sub-node also contains two sub-nodes. There are 4 bottom time
        series in total.

        :param nodes: list of children nodes number lists.
        :param period: frequency of the time series, 1 means non-seasonality data, 12 means monthly data.
        :return:
        """
        nodes = copy(nodes)
        n = sum(nodes[-1])
        m = sum(map(sum, nodes)) + 1
        node_level = n * [len(nodes)]
        nodes.append([1] * n)
        s = np.zeros([m, n])
        c_row_start = m - n
        s[c_row_start:, :] = np.identity(n)
        bts_count = nodes[-1]
        for level_idx in range(len(nodes) - 2, -1, -1):
            c_cum = 0
            c_x = 0
            level = nodes[level_idx]
            c_row_start = c_row_start - len(level)
            new_bts_count = []
            c_row = c_row_start
            for node_idx in range(len(nodes[level_idx])):
                n_x = c_x + level[node_idx]
                new_bts_count.append(sum(bts_count[c_x:n_x]))
                n_cum = c_cum + new_bts_count[-1]
                s[c_row, c_cum:n_cum] = 1
                c_cum = n_cum
                c_row += 1
                c_x = n_x
                node_level.insert(0, level_idx)
            bts_count = new_bts_count
        return cls(s, np.array(node_level), node_level, period)

    @classmethod
    def from_names(cls, names: List[str], chars: List[int], period: int = 1) -> "Hierarchy":
        """Construct Hierarchy from column names of bottom time series. The name of bottom series should be consist of
        several parts. Each part points to a specific level of the hierarchy and it should have fixed length that used to
        split and recognize hierarchy.

        For example :code:`AA`. The first :code:`A` represents :code:`A` series in level1,
        the second :code:`A` represents the :code:`A` series in level 2 and its parent node is :code:`A` series in level 1. This method
        will add a :code:`Total` level.

        **Examples**

            >>> from pyhts.hierarchy import Hierarchy
            >>> names = ['AA', 'AB', 'BA', 'BB']
            >>> hierarchy = Hierarchy.from_names(names, chars=[1, 1], period=12)
            >>> hierarchy.s_mat
            array([[1, 1, 1, 1],
                   [1, 1, 0, 0],
                   [0, 0, 1, 1],
                   [1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=int8)

        The example above define a hierarchy same as the hierarchy in example of
        :meth:`~pyhts.hierarchy.Hierarchy.from_node_list()`.

        :param names: columns of all **bottom** level time series.
        :param chars: character length of each part.
        :param period: frequency of the time series, 1 means non-seasonality data, 12 means monthly data.
        :return:
        """
        df = pd.DataFrame()
        df['bottom'] = names
        df['top'] = 'Total'
        total_index = 0
        names = ['Total']
        node_level = [0]
        index = 0
        for index in range(len(chars)-1):
            df[index+1] = df['bottom'].apply(lambda x: x[:total_index+chars[index]])
            total_index += chars[index]
            names += list(df[index+1].unique())
            node_level += [index+1] * len(df[index+1].unique())
        cols = ['top', *[i+1 for i in range(len(chars)-1)], 'bottom']
        df = df[cols]
        s_mat = pd.get_dummies(df).values.T
        names += list(df['bottom'])
        node_level += [index+2] * len(df['bottom'])
        return cls(s_mat, np.array(node_level), names, period)

    @classmethod
    def from_balance_group(cls, group_list: List[List[str]], period=1) -> "Hierarchy":
        """Construct group hierarchical structure.

        **Examples**

            >>> from pyhts.hierarchy import Hierarchy
            >>> groups = [['A', 'B'], ['Item1', 'Item2']]
            >>> hierarchy = Hierarchy.from_balance_group(groups)
            >>> hierarchy.node_name
            array(['Total', 'A', 'B', 'Item1', 'Item2', 'A_Item1', 'A_Item2',
                   'B_Item1', 'B_Item2'], dtype='<U7')
            >>> hierarchy.s_mat
            array([[1, 1, 1, 1],
                   [1, 1, 0, 0],
                   [0, 0, 1, 1],
                   [1, 0, 1, 0],
                   [0, 1, 0, 1],
                   [1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=int8)


        :param group_list: Two group types are supported for now.
        :param period: frequency of the time series, 1 means non-seasonality data, 12 means monthly data.
        :return:
        """
        from itertools import product
        df = pd.DataFrame(product(*group_list))
        cols = ['top', *[i for i in range(len(group_list))], 'bottom']
        df['bottom'] = df.apply(lambda x: '_'.join(x), axis=1)
        df['top'] = 'total'
        df = df[cols]
        dummy_df = pd.get_dummies(df)
        s_mat = dummy_df.values.T
        node_level = [0]
        names = ['Total']
        i = 0
        for i in range(len(group_list)):
            node_level += [i+1]*len(group_list[i])
            names += group_list[i]
        node_level += [i+2] * df.shape[0]
        names += list(df['bottom'])
        return cls(s_mat, np.array(node_level), names, period)

    @classmethod
    def from_long(cls, df: pd.DataFrame, keys: List[str], period=1) -> "Hierarchy":
        """Construct hierarchy from long data table that each row represents a bottom level time series. This method is
        suitable for complex hierarchical structure.

        **Examples**

            >>> from pyhts.hierarchy import Hierarchy
            >>> df = pd.DataFrame({"City": ["A", "A", "B", "B"], "Store": ["Store1", "Store2", "Store3", "Store4"]})
            >>> hierarchy = Hierarchy.from_long(df, ["City", "Store"])
            >>> hierarchy.node_name
            array(['Total', 'A', 'B', 'Store1', 'Store2', 'Store3', 'Store4'],
                  dtype='<U8')
            >>> hierarchy.s_mat
            array([[1, 1, 1, 1],
                   [1, 1, 0, 0],
                   [0, 0, 1, 1],
                   [1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=int8)
            >>> df = pd.DataFrame({"City": ["A", "A", "B", "B"], "Category": ["C1", "C2", "C1", "C2"]})
            >>> hierarchy = Hierarchy.from_long(df, ["City", "Category"])
            >>> hierarchy.s_mat
            array([[1, 1, 1, 1],
                   [1, 1, 0, 0],
                   [0, 0, 1, 1],
                   [1, 0, 1, 0],
                   [0, 1, 0, 1],
                   [1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=int8)

        :param df: DataFrame contains keys for determining hierarchical structural.
        :param keys: column names that each column represents a level.
        :param period: frequency of the time series, 1 means non-seasonality data, 12 means monthly data.
        :return:
        """
        new_df = copy(df[keys]).astype('string')
        new_df['Total'] = 'Total'
        node_level = [0] + [1] * len(new_df[keys[0]].unique())
        names = ['Total'] + list(new_df[keys[0]].unique())
        n = new_df.shape[0]
        for index in range(1, len(keys)):
            names += list(new_df[keys[index]].unique())
            node_level += [index+1] * len(new_df[keys[index]].unique())

        if len(new_df[keys[-1]].unique()) != new_df.shape[0]:
            new_df['bottom'] = new_df[keys].apply(lambda x: '-'.join(x), axis=1)
            names += list(new_df['bottom'])
            node_level += [len(keys)+1] * n
            s_mat = pd.get_dummies(new_df[['Total'] + keys + ['bottom']], columns=['Total'] + keys + ['bottom']).values.T
        else:
            s_mat = pd.get_dummies(new_df[['Total'] + keys], columns=["Total"] + keys).values.T
        return cls(s_mat, np.array(node_level), names, period=period)

    def aggregate_ts(self, bts: np.ndarray, levels: Union[int, List[int]] = None) -> np.ndarray:
        """aggregate bottom-levels time series.

        :param levels: which levels you want.
        :param bts: bottom time series
        :return: upper-level time series.
        """
        if isinstance(levels, int):
            s = self.s_mat[np.where(self.node_level == levels)]
            return s.dot(bts.T).T
        if isinstance(levels, list):
            s = self.s_mat[np.isin(self.node_level, levels)]
            return s.dot(bts.T).T
        return self.s_mat.dot(bts.T).T

    def check_hierarchy(self, *hts):
        for ts in hts:
            if ts.shape[1] == self.s_mat.shape[1]:
                return True
            else:
                return False

    def accuracy_base(self, real, pred, hist=None,
                      levels: Union[int, None, List] = None,
                      measure: List[str] = None) -> pd.DataFrame:
        """calculate forecast accuracy of base forecast.

        :param real: real future observations, array-like of shape (forecast_horizon, m)
        :param pred: forecast values, array-like of shape (forecast_horizon, n)
        :param levels: which level, None means all levels.
        :param hist: history time series.
        :param measure: list of measures, e.g. ['mase'], ['mse', 'mase'].
        :return: forecast accuracy of base forecasts.
        """
        assert self.check_hierarchy(real), f"true observations should be of shape (h, m)"
        assert real.shape[0] == pred.shape[0], \
            f" {real.shape} true observations have different length with {real.shape} forecasts"
        agg_true = self.aggregate_ts(real, levels=levels)
        if measure is None:
            measure = ['mase', 'mape', 'rmse']
        if 'mase' in measure or 'smape' in measure or 'rmsse' in measure:
            assert hist is not None
            hist = self.aggregate_ts(hist, levels=levels)
        if hist is None:
            hist = [None] * self.s_mat.shape[0]
        accs = pd.DataFrame()
        for me in measure:
            try:
                accs[me] = np.array(list(map(lambda x: getattr(accuracy, me)(*x),
                                             zip(agg_true.T, pred.T, hist.T, [self.period] * self.s_mat.shape[0]))))
            except AttributeError:
                print(f'Forecasting measure {me} is not supported!')
        accs.index = self.node_name[np.isin(self.node_level, levels)]
        return accs

    def accuracy(self, real, pred, hist=None,
                 levels: Union[int, None, List] = None,
                 measure: List[str] = None):
        """calculate forecast accuracy.

        :param real: real future observations, array-like of shape (forecast_horizon, m)
        :param pred: forecast values, array-like of shape (forecast_horizon, m)
        :param levels: which level, None means all levels.
        :param hist: history time series.
        :param measure: list of measures, e.g. ['mase'], ['mse', 'mase'].
        :return: forecast accuracy of reconciled forecasts.
        """
        assert self.check_hierarchy(real), f"true observations should be of shape (h, m)"
        assert self.check_hierarchy(pred), f"forecast values should be of shape (h, m)"
        assert real.shape[0] == pred.shape[0], \
            f" {real.shape} true observations have different length with {real.shape} forecasts"
        if measure is None:
            measure = ['mase', 'mape', 'rmse']
        if 'mase' in measure or 'smape' in measure or 'rmsse' in measure:
            assert hist is not None
            hist = self.aggregate_ts(hist, levels=levels).T
        agg_true = self.aggregate_ts(real, levels=levels)
        agg_pred = self.aggregate_ts(pred, levels=levels)
        accs = pd.DataFrame()
        for me in measure:
            try:
                accs[me] = np.array(list(map(lambda x: getattr(accuracy, me)(*x),
                                             zip(agg_true.T, agg_pred.T, hist, [self.period] * self.s_mat.shape[0]))))
            except AttributeError:
                print('this forecasting measure is not supported!')
        accs.index = self.node_name[np.isin(self.node_level, levels)]
        return accs
