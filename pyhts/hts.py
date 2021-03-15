import numpy as np
from pyhts.accuracy import *
from typing import List

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
forecast = importr("forecast")



#TODO: change to sparse matrix
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
        self.m = None

    @classmethod
    def from_hts(cls, bts, nodes, m):
        hts = cls()
        hts.bts = bts
        hts.constraints, hts.node_level = _nodes2constraints(nodes)
        hts.m = m
        return hts

    def aggregate_ts(self, levels=None):
        if isinstance(levels, int):
            s = self.constraints[np.where(self.node_level == levels)]
            return s.dot(self.bts.T).T
        if isinstance(levels, list):
            s = self.constraints[np.where(self.node_level in levels)]
            return s.dot(self.bts.T).T
        return self.constraints.dot(self.bts.T).T

    # TODO: 优化结构
    def forecast(self, base_forecast, reconciliation_method="ols"):
        import pyhts.reconciliation as fr
        if reconciliation_method == "ols":

            return fr.wls(self, base_forecast)
        if reconciliation_method in ["shrinkage", "wls", "nseries", "sample"]:
            return fr.min_trace(self, base_forecast, reconciliation_method)
        return None

    def generate_base_forecast(self, method="arima", h=1, keep_fitted=False):
        ts = robjects.r['ts']

        y = self.constraints.dot(self.bts.T)
        if keep_fitted:
            f_casts = np.zeros([h+y.shape[1], y.shape[0]])
        else:
            f_casts = np.zeros([h, y.shape[0]])
        if method == "arima":
            auto_arima = forecast.auto_arima
            for i in range(y.shape[0]):
                series = ts(FloatVector(y[i, :]), frequency=self.m)
                model = forecast.forecast(auto_arima(series), h=12)
                if keep_fitted:
                    f_casts[:y.shape[1], i] = np.array(model.rx2["fitted"])
                    f_casts[y.shape[1]:, i] = np.array(model.rx2["mean"])
                else:
                    f_casts[:, i] = np.array(model.rx2["mean"])
            return f_casts

    def accuracy(self, y_true, y_pred, levels=0):
        # MASE
        agg_ts = self.aggregate_ts(levels=levels) # history
        agg_true = y_true.aggregate_ts(levels=levels)
        agg_pred = y_pred.aggregate_ts(levels=levels)
        mases = np.array(list(map(lambda x,y: mase(*x, y), zip(agg_ts.T, agg_true.T, agg_pred.T), [12]*agg_ts.shape[1])))
        return mases


if __name__ == '__main__':
    a = np.random.random((100, 14))
    nodes = [[2], [2, 2], [3, 4, 3, 4]]
    hts = Hts.from_hts(a, nodes, m=1)
    a = hts.aggregate_ts(levels=0)
    b = hts.aggregate_ts(levels=1)