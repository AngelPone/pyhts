import unittest
import pickle as pkl
from pathlib import Path

import numpy as np

from pyhts import load_tourism, HFModel, Hierarchy, TemporalHierarchy, TemporalHFModel

def prepare_data():
    tourism_data = load_tourism()
    train = tourism_data.iloc[:, 4:-12].T.values
    test = tourism_data.iloc[:, -12:].T.values
    hierarchy = Hierarchy.new(tourism_data, [('state', 'region', 'city')])
    with open(Path(__file__).parent / "tourism_model.pkl", 'rb') as f:
        base_forecasters = pkl.load(f)
    return train, test, hierarchy, base_forecasters

class TestWorkflow(unittest.TestCase):

    def test_unconstrained_reconciliation(self):
        train, test, ht, base_fcasters = prepare_data()

        model_ols = HFModel(hierarchy=ht, base_forecasters=base_fcasters,
                            hf_method="comb", comb_method="ols")
        model_ols.fit(train)
        model_wls = HFModel(hierarchy=ht, base_forecasters=base_fcasters,
                            hf_method="comb", comb_method="wls", weights="structural")
        model_wls.fit(train)
        model_wlsv = HFModel(hierarchy=ht, base_forecasters=base_fcasters,
                            hf_method="comb", comb_method="mint", weights="variance")
        model_wlsv.fit(train)
        model_shrinkage = HFModel(hierarchy=ht, base_forecasters=base_fcasters,
                                  hf_method="comb", comb_method="mint", weights="shrinkage")
        model_shrinkage.fit(train)
        ols = model_ols.predict(horizon=12)
        wlss = model_wls.predict(horizon=12)
        wlsv = model_wlsv.predict(horizon=12)
        shrink = model_shrinkage.predict(horizon=12)

        accuracy = [ht.accuracy(test, fcast, hist=train, measure=['mase', 'rmse'])
                    for fcast in (ols, wlss, wlsv, shrink)]
        base_forecasts = model_ols.generate_base_forecast(horizon=12)
        accuracy_base = ht.accuracy_base(test, base_forecasts, hist=train, measure=['mase', 'rmse'])

    def test_constrained_reconciliation(self):
        train, test, ht, base_fcasters = prepare_data()

        model_ols = HFModel(hierarchy=ht, base_forecasters=base_fcasters,
                            hf_method="comb", comb_method="ols", immutable_set=[0])
        model_ols.fit(train)
        ols1 = model_ols.predict(horizon=12)
        base_forecasts = model_ols.generate_base_forecast(horizon=12)
        self.assertTrue(np.allclose(base_forecasts[:, [0]], ht.aggregate_ts(ols1, ht.node_name[0])))


        # immutable series from different levels
        model_ols = HFModel(hierarchy=ht, base_forecasters=base_fcasters,
                            hf_method="comb", comb_method="ols", immutable_set=[0, 100])
        model_ols.fit(train)
        ols1 = model_ols.predict(horizon=12)
        self.assertTrue(np.allclose(base_forecasts[:, [0, 100]], ht.aggregate_ts(ols1, ht.node_name[[0, 100]])))

def nonoverlapping_aggregate(x, size):
    assert len(x) % size == 0
    all_windows = np.lib.stride_tricks.sliding_window_view(x, size).sum(axis=1)
    idxer = [i for i in range(len(x) - size + 2) if i % size == 0]
    return all_windows[idxer]

class TestTemporalWorkflow(unittest.TestCase):

    def testNew(self):
        ht = TemporalHierarchy.new([12, 6, 4, 3, 2, 1], 12)

        self.assertTrue(np.array_equal(ht.s_mat, np.concatenate([
            np.ones((1, 12)),
            np.kron(np.eye(2, dtype=int), np.ones((1, 6))),
            np.kron(np.eye(3, dtype=int), np.ones((1, 4))),
            np.kron(np.eye(4, dtype=int), np.ones((1, 3))),
            np.kron(np.eye(6, dtype=int), np.ones((1, 2))),
            np.identity(12)
        ], axis=0)))
        self.assertEqual(ht.period, 12)

    def test_aggregate_ts(self):
        ht = TemporalHierarchy.new([12, 6, 4, 3, 2, 1], 12)
        bts = np.random.random(100)

        ats = ht.aggregate_ts(bts)

        self.assertIsInstance(ats, dict)

        bts = bts[-96:]
        self.assertTrue(np.allclose(ats['agg_1'], bts))
        self.assertTrue(np.allclose(ats['agg_2'], nonoverlapping_aggregate(bts, 2)))
        self.assertTrue(np.allclose(ats['agg_3'], nonoverlapping_aggregate(bts, 3)))
        self.assertTrue(np.allclose(ats['agg_4'], nonoverlapping_aggregate(bts, 4)))
        self.assertTrue(np.allclose(ats['agg_6'], nonoverlapping_aggregate(bts, 6)))
        self.assertTrue(np.allclose(ats['agg_12'], nonoverlapping_aggregate(bts, 12)))

        ats = ht.aggregate_ts(bts, levels=['agg_1', 'agg_3'])
        self.assertEqual(len(ats.keys()), 2)

    def test_dict2array(self):
        ht = TemporalHierarchy.new([12, 6, 4, 3, 2, 1], 12)
        bts = np.random.random(96)

        tmp1 = ht.aggregate_ts(bts)

        tmp2 = ht._temporal_dict2array(tmp1)

        self.assertEqual(tmp2.shape[0], 8)
        self.assertEqual(tmp2.shape[1], ht.s_mat.shape[0])

        self.assertTrue(np.allclose(tmp2[:, 0], tmp1['agg_12']))
        for i in range(8):
            self.assertTrue(np.allclose(tmp2[i, 1:3], tmp1['agg_6'][i*2:2*(i+1)]))

    def test_workflow(self):
        ts = np.random.random(120)
        ht = TemporalHierarchy.new([12, 6, 4, 3, 2, 1], 12)

        hf = TemporalHFModel(ht, "arima")
        hf.fit(ts)
        pred = hf.predict(horizon=2)

        # accuracy
        model_wls = TemporalHFModel(hierarchy=ht, base_forecasters=hf.base_forecasters,
                            hf_method="comb", comb_method="wls", weights="structural")
        model_wls.fit(ts)
        model_wlsv = TemporalHFModel(hierarchy=ht, base_forecasters=hf.base_forecasters,
                             hf_method="comb", comb_method="mint", weights="variance")
        model_wlsv.fit(ts)
        model_shrinkage = TemporalHFModel(hierarchy=ht, base_forecasters=hf.base_forecasters,
                                          hf_method="comb", comb_method="mint", weights="shrinkage")
        model_shrinkage.fit(ts)
        ols = pred
        wlss = model_wls.predict(horizon=2)
        wlsv = model_wlsv.predict(horizon=2)
        shrink = model_shrinkage.predict(horizon=2)

        accs = [ht.accuracy(np.random.random(24), f, ts) for f in (ols, wlss, wlsv, shrink)]


if __name__ == '__main__':
    unittest.main()