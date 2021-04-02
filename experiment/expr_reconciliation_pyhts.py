



# ---

# Tourism Data

from load_data import load_tourismV4
import pickle
from pyhts.reconciliation import wls
hts_trian, hts_test = load_tourismV4()

# res = hts_trian.generate_base_forecast("arima", 12, keep_fitted=True)

# save base forecast for debug
# with open("base_forecast.pkl", "wb") as f:
#    pickle.dump(res, f)

with open("base_forecast.pkl", "rb") as f:
     res = pickle.load(f)

for level in range(4):
     print("level:", level)
     reconciled_hts = wls(hts_trian, res[228:], method="ols", constraint=False)
     print("ols: ", hts_trian.accuracy(hts_test, reconciled_hts, level).mean())

     reconciled_hts = wls(hts_trian, res[228:], method="ols", constraint=True)
     print("cls: ", hts_trian.accuracy(hts_test, reconciled_hts, level).mean())

     reconciled_hts = wls(hts_trian, res, method="mint", weighting="shrink", constraint=False)
     print("mint shrinkage: ", hts_trian.accuracy(hts_test, reconciled_hts, level).mean())

     reconciled_hts = wls(hts_trian, res, method="mint", weighting="shrink", constraint=True)
     print("c mint shrinkage: ", hts_trian.accuracy(hts_test, reconciled_hts, level).mean())

     reconciled_hts = wls(hts_trian, res, method="mint", weighting="var", constraint=False)
     print("mint var: ", hts_trian.accuracy(hts_test, reconciled_hts, level).mean())

     reconciled_hts = wls(hts_trian, res, method="mint", weighting="var", constraint=True)
     print("c mint var: ", hts_trian.accuracy(hts_test, reconciled_hts, level).mean())

     reconciled_hts = wls(hts_trian, res[228:], method="wls", weighting="nseries", constraint=False)
     print("wls: ", hts_trian.accuracy(hts_test, reconciled_hts, level).mean())

     reconciled_hts = wls(hts_trian, res[228:], method="wls", weighting="nseries", constraint=True)
     print("c wls: ", hts_trian.accuracy(hts_test, reconciled_hts, level).mean())

     # reconciled_hts = wls(hts_trian, res, method="mint", weighting="cov", constraint=False)
     ##print("mint cov: ", hts_trian.accuracy(hts_test, reconciled_hts, level).mean())

     # reconciled_hts = wls(hts_trian, res, method="mint", weighting="cov", constraint=True)
     # print("c mint cov: ", hts_trian.accuracy(hts_test, reconciled_hts, level).mean())

