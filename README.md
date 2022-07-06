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
from pyhts._hierarchy import Hierarchy

hierarchy = Hierarchy.from_names(tourism_data.columns, chars=[1, 1, 1])
print(hierarchy.node_name)
```

- Create an ols forecasting reconciliation model with sklearn-like API.

```python
from pyhts._HFModel import HFModel

model = HFModel(hierarchy=hierarchy, base_forecasters="arima",
                hf_method="comb", comb_method="ols")
```

- Fit the model and produce forecasts.

```python
model.fit(train)
forecasts = model.predict(horizon=12)
```

* `model.fit()` fits the `baseforecasters` and computes the weighting matrix used to reconcile the base forecasts.

* `model.forecast()` calculates the base forecasts for all levels and reconciles the base forecasts.

- Obtain coherent forecasts of all the hierarchical levels.

```python
all_level_forecasts = hierarchy.aggregate_ts(forecasts)
```

- Evaluate the forecasting accuracy.

```python
# accuracy of reconciled forecasts
hierarchy.accuracy(test, forecasts, hist=train, measure=['mase', 'rmse'])

# accuracy of base forecasts
base_forecasts = model.generate_base_forecast(horizon=12)
hierarchy.accuracy_base(test, base_forecasts, hist=train, measure=['mase', 'rmse'])
```

Because of the incoherence of base forecasts, `base_forecasts` are forecasts of all time series in the hierarchy, while coherent `forecasts` are forecasts of the bottom-level time series.  



## Documentation
See documentation [here](https://angelpone.github.io/pyhts/).
