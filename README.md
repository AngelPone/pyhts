# py-hts

A python package for hierarchical forecasting, inspired by [hts](https://cran.r-project.org/web/packages/hts/index.html) package in R.

## features

- support pupular forecast reconciliation models in the literature, e.g. ols, wls, mint et al. Temporal Hierarchy will be supported in the future. 
- multiple methods for the construction of hierarchy.
- use different base forecasters for different hierarchical levels.
- familiar sklearn-like API


## Quick Demo

Load tourism data, which is tourism demand measured by the number of "visitor nights" in Australia.

```python
from pyhts.dataset import load_tourism

tourism_data = load_tourism()
train = tourism_data.iloc[:-12, :]
test = tourism_data.iloc[-12:, :]
```


Define hierarchy

```python
from pyhts.hierarchy import Hierarchy
hierarchy = Hierarchy.from_names(tourism_data.columns, chars=[1, 1, 1])
print(hierarchy.node_name)
```

Create an ols forecasting reconciliation model with sklearn-like API.

```python
from pyhts.HFModel import HFModel
model = HFModel(hierarchy=hierarchy, base_forecasters="arima", 
                hf_method="comb", comb_method="ols")
```

Fit the model and forecast.

```python
model.fit(train)
forecasts = model.predict(horizon=12)
```

* `model.fit()` will fit the `baseforecasters` and compute the weighting matrix used to reconcile the base forecast.

* `model.forecast()` will calculate base forecasts of all levels and reconcile the base forecasts.

Obtain coherent forecasts of all hierarchical levels.

```python
all_level_forecasts = hierarchy.aggregate_ts(forecasts)
```

evaluate forecasting accuracy

```python
# accuracy of reconciled forecasts
hierarchy.accuracy(test, forecasts, hist=train, measure=['mase', 'rmse'])

# accuracy of base forecasts
base_forecasts = model.generate_base_forecast(horizon=12)
hierarchy.accuracy_base(test, base_forecasts, hist=train, measure=['mase', 'rmse'])
```

because of the incoherence of base forecasts, `base_forecasts` are forecasts of all time series in the hierarchy, while 
coherent `forecasts` are forecasts of bottom time series.  









## Documentation
see documentation here https://angelpone.github.io/pyhts/.
