
from load_data import load_tourismV4

hts_trian, hts_test = load_tourismV4()

res = hts_trian.forecast(h=12, base_method="arima", hf_method="comb",
                   weights_method="ols")

print(hts_trian.accuracy_base(hts_test))
print(hts_trian.accuracy(hts_test, res))