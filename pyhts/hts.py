import numpy as np
from typing import List


def _nodes2constraints(nodes: List):
    n = sum(nodes[-1])
    m = sum(map(sum, nodes)) + 1
    node_level = n * [len(nodes)]
    nodes.append([1]*n)
    s = np.zeros([m, n])
    c_row_start = m - n
    s[c_row_start:, :] = np.identity(n)
    bts_count = nodes[-1]
    for level_idx in range(len(nodes)-2, -1, -1):

        c_cum = 0
        c_x = 0
        level = nodes[level_idx]
        c_row_start = c_row_start - len(level)
        new_bts_count = []
        c_row = c_row_start
        print(c_row)
        for node_idx in range(len(nodes[level_idx])):
            n_x = c_x+level[node_idx]
            new_bts_count.append(sum(bts_count[c_x:n_x]))
            n_cum = c_cum + new_bts_count[-1]
            s[c_row, c_cum:n_cum] = 1
            c_cum = n_cum
            c_row += 1
            c_x = n_x
            node_level.insert(0, level_idx)
        bts_count = new_bts_count

    return s, np.array(node_level)


class Hts:

    def __init__(self):
        self.constraints = None
        self.bts = None
        self.node_level = None

    @classmethod
    def from_hts(cls, bts, nodes):
        hts = cls()
        hts.bts = bts
        hts.constraints, hts.node_level = _nodes2constraints(nodes)
        return hts

    def aggregate_ts(self, levels=0):
        if isinstance(levels, int):
            s = self.constraints[np.where(self.node_level == levels)]
            return s.dot(self.bts.T).T
        if isinstance(levels, list):
            s = self.constraints[np.where(self.node_level in levels)]
            return s.dot(self.bts.T).T
        return None


if __name__ == '__main__':
    a = np.random.random((100, 14))
    nodes = [[2], [2, 2], [3, 4, 3, 4]]
    hts = Hts.from_hts(a, nodes)
    a = hts.aggregate_ts(levels=0)
    b = hts.aggregate_ts(levels=1)