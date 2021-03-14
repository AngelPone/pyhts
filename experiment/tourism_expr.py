



# ---

# Tourism Data

from load_data import load_tourismV4
import pickle
hts_trian, hts_test = load_tourismV4()

res = hts_trian.generate_base_forecast("arima", 12)

# save base forecast for debug
with open("base_forecast.pkl", "wb") as f:
    pickle.dump(res, f)

# with open("base_forecast.pkl", "rb") as f:
#     res = pickle.load(f)

res = hts_trian.forecast(res, "ols")

print(hts_trian.accuracy(hts_test, res))