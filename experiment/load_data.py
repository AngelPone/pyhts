import pandas as pd
from pyhts.hts import Hts

def load_tourismV4():
    """
    Tourism 数据，从1998年1月到2017年12月，共240个观测，
    :return:
    """
    tourism = pd.read_csv("data/TourismData_v4.csv")
    nodes = [[7], [], []]
    for i in "ABCDEFG":
        for j in "ABCDEFG":
            s = tourism.columns.str.startswith(i+j).sum()
            if s > 0:
                nodes[2].append(s)
            else:
                break
        nodes[1].append("ABCDEFG".find(j))

    hts_train = Hts.from_hts(tourism.values[:228, :], nodes, m=12)
    hts_test = Hts.from_hts(tourism.values[228:, :], nodes, m=12)
    return hts_train, hts_test


if __name__ == '__main__':
    load_tourismV4()


