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

    hts_train = Hts.from_hts(tourism.values[:228, :], m=12, nodes=nodes)
    hts_test = Hts.from_hts(tourism.values[228:, :], m=12, nodes=nodes)
    return hts_train, hts_test


if __name__ == '__main__':
    # import numpy as np
    # from time import time
    from pyhts.hts import _constraints_from_chars

    tourism = pd.read_csv("data/TourismData_v4.csv")

    a, b = _constraints_from_chars(list(tourism.columns), [1,1,1])
    c, d = load_tourismV4()
    import numpy as np
    print(np.all(a==c.constraints))
    print(np.all(b==c.node_level))


    # hts_train, test = load_tourismV4()
    # S = hts_train.constraints.toarray()
    # weight_matrix = np.identity(S.shape[0])*10
    # start = time()
    # for i in range(1000):
    #     np.linalg.inv(S.T.dot(weight_matrix).dot(S)).dot(S.T).dot(weight_matrix)
    # end = time()
    # print(f"pure numpy product cost {end-start}s")
    #
    # S = hts_train.constraints
    # start = time()
    # for i in range(1000):
    #     np.linalg.inv(S.T.dot(weight_matrix).dot(S.toarray())).dot(S.T.toarray()).dot(weight_matrix)
    # end = time()
    # print(f"sparse matrix mixed with numpy cost {end-start} s")
    #
    # start = time()
    # for i in range(1000):
    #     S.toarray()
    # end = time()
    # print(f"S.toarray() cost {end - start} s")





