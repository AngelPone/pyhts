from typing import List, Iterable
import numpy as np
import pandas as pd
from copy import copy
from scipy.sparse import csr_matrix


class Hierarchy:
    """Class for hierarchy tree.

    """

    def __init__(self, s_mat, node_level, names):
        self.s_mat = csr_matrix(s_mat)
        self.node_level = node_level
        self.level_n = max(node_level) + 1
        self.node_name = names

    @classmethod
    def from_node_list(cls, nodes: List[List[int]]):
        """Construct Hierarchy from node list.

        :param nodes: e.g. [[2], [2, 2]]
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
        return cls(s, node_level, node_level)

    @classmethod
    def from_names(cls, names: List[str], chars: List[int]):
        """Construct Hierarchy from column names

        :param names:
        :param chars:
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
            df[index+1] = df['bottom'].apply(lambda x: x[:total_index+index+1])
            total_index += chars[index]
            names += list(df[index+1].unique())
            node_level += [index+1] * len(df[index+1].unique())
        cols = ['top', *[i+1 for i in range(len(chars)-1)], 'bottom']
        df = df[cols]
        s_mat = pd.get_dummies(df).values.T
        names += list(df['bottom'])
        node_level += [index+2] * len(df['bottom'])
        return cls(s_mat, node_level, names)

    @classmethod
    def from_balance_group(cls, group_list: List[List[str]]):
        """Constructor for balanced group.

        :param group_list: e.g. [['A', 'B', 'C'], [1, 2, 3, 4]]
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
        return cls(s_mat, node_level, names)

    @classmethod
    def from_long(cls, df: pd.DataFrame, keys: Iterable[str]):
        """Constructor for data that contains complex hierarchies

        :param df: DataFrame contains keys for determining hierarchical structural.
        :param keys: column names
        :return:
        """
        new_df = df[keys]
        new_df['Total'] = 'Total'
        cols = ['Total'] + keys
        new_df = new_df[cols]
        dummy_df = pd.get_dummies(new_df)
        s_mat = dummy_df.values.T
        names = ['Total']
        node_level = [0]
        index = 1
        for key in keys:
            names += list(df[key].unique())
            node_level += [index] * len(list(df[key].unique()))
            index += 1
        return cls(s_mat, node_level, names)


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