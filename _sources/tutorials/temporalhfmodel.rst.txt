Temporal hierarchical forecasting model
#######################################

Temporal hierarchical forecasting model is defined by :class:`~pyhts.TemporalHFModel`


TemporalHFModel API
====================

.. autoclass:: pyhts.TemporalHFModel
   :members: __init__, fit, predict, generate_base_forecast



Examples 
========

Randomly generate some dataset and define the hierarchy. Here, we
consider a monthly time series and aggregation perios are 2, 3, 6, 12.

.. code-block:: python

   >>> import numpy as np
   >>> from pyhts import TemporalHierarchy
   >>> ts = np.random.random(120)
   >>> ht = TemporalHierarchy.new(agg_periods=[1, 2, 3, 6, 12], forecast_frequency=12)
   >>> ht.level_name
   ['agg_12', 'agg_6', 'agg_3', 'agg_2', 'agg_1']



Define and fit temporal hierarchical forecasting model
------------------------------------------------------

Then we can construct a temporal hierarchical forecasting model using
the ols reconciliation method and arima models as base forecasters.

.. code-block:: python
   
   >>> from pyhts import TemporalHFModel
   >>> hfmodel = TemporalHFModel(ht, "arima", hf_method='comb', comb_method='ols')
   >>> hfmodel.fit(ts)


Predict and evaluate forecasts
------------------------------

The horizon here should be the corresponding forecast horizon for the top level.
For example, if we want to predict the next 12 months, the horizon
should be 1 (the top level is year).

.. code-block:: python

   >> fcasts = hfmodel.predict(1)
   {'agg_12': array([5.89485775]),
   'agg_6': array([2.94742887, 2.94742887]),
   'agg_3': array([1.47371444, 1.47371444, 1.47371444, 1.47371444]),
   'agg_2': array([0.98247629, 0.98247629, 0.98247629, 0.98247629, 0.98247629,
         0.98247629]),
   'agg_1': array([0.49123815, 0.49123815, 0.49123815, 0.49123815, 0.49123815,
         0.49123815, 0.49123815, 0.49123815, 0.49123815, 0.49123815,
         0.49123815, 0.49123815])}

The forecasts can be evaluated using :class:`~pyhts.TemporalHierarchy.accuracy()`
method. We pass the future observations, forecasts and historical 
observations (if needed). The measures for each level are returned.

.. code-block:: python

   >> real = np.random.random(12)
   >> ht.accuracy(real, pred, ts)
               mase      mape      rmse
      agg_12  3.023196  0.187241  1.358036
      agg_6   1.248526  0.186900  0.683057
      agg_3   0.742426  0.178560  0.455563
      agg_2   1.144290  0.504512  0.525026
      agg_1   0.810293  2.290112  0.317474


