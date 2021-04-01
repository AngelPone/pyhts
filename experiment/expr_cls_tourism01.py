from load_data import load_tourismV4
import pickle
import numpy as np
import pyhts.reconciliation as fr
from pyhts.accuracy import mase
hts_trian, hts_test = load_tourismV4()

with open("base_forecast.pkl", "rb") as f:
    res = pickle.load(f)

# constrained ols
reconciled_hts = fr.constrained_wls(hts_trian, res[228:], 1)

# ols
reconciled_hts2 = fr.wls(hts_trian, res[228:])

from copy import deepcopy
base_hts = deepcopy(hts_test)
base_hts.bts = res[228:, (111-76):]


combined_hts = deepcopy(hts_test)
combined_hts.bts = np.mean([fr.constrained_wls(hts_trian, res[228:], i).bts for i in range(3)], axis=0)

for i in range(4):
    accs = hts_trian.accuracy(hts_test, reconciled_hts, i)
    accs2 = hts_trian.accuracy(hts_test, reconciled_hts2, i)
    accs3 = hts_trian.accuracy(hts_test, base_hts, i)
    accs4 = hts_trian.accuracy(hts_test, combined_hts, i)

    his = hts_trian.aggregate_ts(i)
    y_true = hts_test.aggregate_ts(i)
    base = res[228:, np.where(hts_trian.node_level == i)]
    mases = np.array(list(map(lambda x, y: mase(*x, y), zip(his.T,
                                                            y_true.T,
                                                            base.T), [12] * his.shape[1])))
    print(f"level{i}:")
    print(f"bu mean: ", np.mean(accs3))
    print("cls  mean: ", np.mean(accs))
    print("ols  mean: ", np.mean(accs2))
    print("base mean: ", np.mean(mases))
    print("combined mean: ", np.mean(accs4))
    print("\n")


