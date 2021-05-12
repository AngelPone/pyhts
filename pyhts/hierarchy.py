from typing import List
import numpy as np
import pandas as pd
from copy import copy
from typing import Union
from . import accuracy


class Hierarchy:
    """Class for hierarchy tree.

    """

    def __init__(self, s_mat, node_level, names, period):
        self.s_mat = s_mat
        self.node_level = np.array(node_level)
        self.level_n = max(node_level) + 1
        self.node_name = np.array(names)
        self.period = period

    @classmethod
    def from_node_list(cls, nodes: List[List[int]], period=1):
        """Construct Hierarchy from node list.

        :param nodes: e.g. [[2], [2, 2]]
        :param period: seasonality
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
    def from_names(cls, names: List[str], chars: List[int], period: int = 1):
        """Construct Hierarchy from column names

        :param names:
        :param chars:
        :param period:
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
    def from_balance_group(cls, group_list: List[List[str]], period=1):
        """Constructor for balanced group.

        :param group_list: e.g. [['A', 'B', 'C'], [1, 2, 3, 4]]
        :param period:
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
    def from_long(cls, df: pd.DataFrame, keys: List[str], period=1):
        """Constructor for data that contains complex hierarchies

        :param df: DataFrame contains keys for determining hierarchical structural.
        :param keys: column names
        :param period:
        :return:
        """
        new_df = copy(df[keys]).astype('string')
        new_df['Total'] = 'Total'
        node_level = [0] + [1] * len(new_df[keys[0]].unique())
        names = ['Total'] + list(new_df[keys[0]].unique())
        for index in range(1, len(keys)):
            new_df[keys[index]] = new_df[keys[index-1]] + '-' + new_df[keys[index]]
            names += list(new_df[keys[index]].unique())
            node_level += [index+1] * len(new_df[keys[index]].unique())
        s_mat = pd.get_dummies(new_df[['Total'] + keys], columns=['Total'] + keys).values.T
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
        """calculate forecast accuracy of base forecast

        :param real: real future observations, array-like of shape (forecast_horizon, m)
        :param pred: forecast values, array-like of shape (h, n)
        :param levels: which level.
        :param hist: history time series.
        :param s: seasonality of series
        :param measure:
            mase, mse is supported for now. e.g. ['mase'], ['mse', 'mase'].
            if None, mase is calculated.
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
        """calculate forecast accuracy, mase is supported only for now.

        :param real: real observations.
        :param pred: forecasts.
        :param hist: history observations
        :param s: seasonality
        :param levels: which level.
        :param measure:
            mase, mse is supported for now. e.g. ['mase'], ['mse', 'mase'].
            if None, mase is calculated.
        :return: forecast accuracy.
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


if __name__ == '__main__':
    # test for characters
    # groups = ['AA', 'AB', 'AC', 'AE', 'BA', 'BB', 'BC', 'BD']
    # hierarchy = Hierarchy.from_names(groups, [1, 1])

    # test for balance group
    # groups = [['A', 'B', 'C'], ['1', '2', '3', '4']]
    # hierarchy = Hierarchy.from_balance_group(groups)

    # test for long data
    df = pd.DataFrame({'Country': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
                       'City': ['City1', 'City1', 'City2', 'City2', 'City3', 'City3', 'City4', 'City4'],
                       'Station': [str(i) for i in [1, 2, 3, 4, 5, 6, 7, 8]]})
    hierarchy = Hierarchy.from_long(df, keys=['Country', 'City', 'Station'])
    print(hierarchy.s_mat.toarray())
    print(hierarchy.node_name)
    print(hierarchy.node_level)

    from sklearn.metrics import mean_squared_error