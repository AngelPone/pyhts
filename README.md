# `pyhts`
A python package for hierarchical forecasting, inspired by the [hts](https://cran.r-project.org/web/packages/hts/index.html) package in R.

## Features

- Support pupular forecast reconciliation models in the literature, e.g. ols, wls, mint et al. Forecasting with temporal hierarchies will be supported in the future. 
- Multiple methods for the construction of hierarchy.
- Use different base forecasters for different hierarchical levels.
- Sklearn-like API.


## Quick Demo

- Load the Australia tourism flows data.

```python
from pyhts._dataset import load_tourism

tourism_data = load_tourism()
train = tourism_data.iloc[:-12, :]
test = tourism_data.iloc[-12:, :]
```


- Define the hierarchy.

```python
from pyhts import Hierarchy

hierarchy = Hierarchy.from_names(tourism_data.columns, chars=[1, 1, 1])
print(hierarchy.node_name)
```

- Create an ols forecasting reconciliation model with sklearn-like API.

```python
from pyhts import HFModel

model_ols = HFModel(hierarchy=hierarchy, base_forecasters="arima",
                hf_method="comb", comb_method="ols")
```

- Fit the model and produce forecasts.

```python
model_ols.fit(train)
ols = model.predict(horizon=12)
```

* `model.fit()` fits the `baseforecasters` and computes the weighting matrix used to reconcile the base forecasts.

* `model.predict()` calculates the base forecasts for all levels and reconciles the base forecasts.

- Obtain coherent forecasts of all the hierarchical levels.

```python
all_level_ols = hierarchy.aggregate_ts(ols)
```

- fit other methods using fitted base forecasters

```python
model_wlss = HFModel(hierarchy, base_forecasters=model_ols.base_forecasters,
                     hf_method="comb", comb_method="wls", weights="structural")
model_wlss.fit(train)
wlss = model_wlss.predict(horizon=12)

model_wlsv = HFModel(hierarchy, base_forecasters=model_ols.base_forecasters,
                     hf_method="comb", comb_method="mint", weights="variance")
model_wlsv.fit(train)
wlsv = model_wlsv.predict(horizon=12)

model_shrink = HFModel(hierarchy, base_forecasters=model_ols.base_forecasters,
                       hf_method="comb", comb_method="mint", weights="shrinkage")
model_shrink.fit(train)
shrink = model_shrink.predict(horizon=12)
```

- Evaluate the forecasting accuracy.

```python
# accuracy of reconciled forecasts
accuracy = [hierarchy.accuracy(test, fcast, hist=train, measure=['mase', 'rmse'])
            for fcast in (ols, wlss, wlsv, shrink)]

# accuracy of base forecasts
base_forecasts = model_ols.generate_base_forecast(horizon=12)
accuracy_base = hierarchy.accuracy_base(test, base_forecasts, hist=train, measure=['mase', 'rmse'])
```

Because of the incoherence of base forecasts, `base_forecasts` are forecasts of all time series in the hierarchy, while coherent `forecasts` are forecasts of the bottom-level time series.  



## Documentation
See documentation [here](https://angelpone.github.io/pyhts/).
