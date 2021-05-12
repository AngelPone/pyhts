import pandas as pd
import numpy as np
from pyhts.hierarchy import Hierarchy


def construct_hierarchy_from_names():
    df = pd.read_csv('data/Tourism.csv')
    hts = Hierarchy.from_names(df.columns, chars=[1, 1, 1], period=12)
    return hts


def construct_hierarchy_from_node_list():
    hts = Hierarchy.from_node_list([[2], [2, 2], [3, 4, 3, 4]])
    return hts


def construct_hierarchy_from_long():
    array = np.random.random(70).reshape((7, 10))
    df_array = pd.DataFrame(array).rename(lambda x: f'T_{x+1}', axis=1)
    df_index = pd.DataFrame({'City': ['City_A'] * 3 + ['City_B'] * 4,
                             'Store': ['Store1', 'Store2', 'Store3'] * 2 + ['Store4']})
    df = pd.concat([df_index, df_array], axis=1)
    return Hierarchy.from_long(df, keys=['City', 'Store'])


def construct_hierarchy_from_balance_group():
    hts = Hierarchy.from_balance_group([['C1', 'C2', 'C3'], ['P1', 'P2', 'P3', 'P4']])
    return hts
