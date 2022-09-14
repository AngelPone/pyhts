import unittest
import pickle as pkl
from pathlib import Path

import numpy as np

from pyhts import load_tourism, HFModel, Hierarchy

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

if __name__ == '__main__':
    unittest.main()