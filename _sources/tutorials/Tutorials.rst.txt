Tutorials
============

Before reading this tutorial, you'd better read :doc:`notation` first. If you're not familiar with Hierarchical
forecasting and forecast reconciliation, read the papers provided in :ref:`references`.

Construction of Hts object
--------------------------

There are three types of hierarchical time series supported for now, e.g. 
cross-sectional hierarchical time series, cross-sectional grouped time
series and temporal hierarchies :ref:`[1]<references>`. For the first two types, you can use
:class:`~pyhts.hts.Hts` and construct from its classmethod,
respectively :meth:`~pyhts.hts.Hts.from_hts()` and :meth:`~pyhts.hts.Hts.from_gts()`.

As an example of the hierarchy shown in the figure below, use :meth:`~pyhts.hts.Hts.from_hts()` to construct
Hts object.

.. image:: ../media/01_hierarchy.png

.. code-block:: python

    from pyhts.hts import Hts
    import numpy as np
    bts = np.random.random(96).reshape([24, 4])
    nodes = [[2], [2, 2]]
    m = 12
    hts = Hts.from_hts(bts, m=m, nodes=nodes)

use `nodes` to demonstrate the hierarchy, and we construct a hts of 4 bottom series with 24 observations each.
`m` is frequency of time series, `m=12` means monthly series.

Also, parameter `characters` can be used to represent hierarchical structures, see code examples below.

.. code-block:: python

    import pandas as pd
    bts = pd.DataFrame(bts)
    bts.columns = ["AAA", "AAB", "ABC", "ABD"]
    hts = Hts.from_hts(bts, m=m, characters=[1,1,1])

Aggregation
-----------

You can use :meth:`~pyhts.hts.Hts.aggregate_ts()` to aggregate bottom-level time series.

.. code-block:: python

    >>> hts.aggregate_ts()
    array([[1.24324536, 0.70214108, 0.54110428, 0.46049853, 0.24164255,
        0.0250346 , 0.51606968],
        ...
       [2.63623735, 1.04859469, 1.58764265, 0.97204591, 0.07654878,
        0.84455429, 0.74308837]])
    >>> hts.aggregate_ts().shape
    (24, 7)

You can also specify `levels` to get aggregated time series of specific levels.

.. code-block:: python

    >>> hts.aggregate_ts(levels=0)
    array([[1.24324536],
           ...
           [2.63623735]])
    >>> hts.aggregate_ts(levels=[0, 1])
    array([[1.24324536, 0.70214108, 0.54110428],
           ...
           [2.63623735, 1.04859469, 1.58764265]])
    >>> hts.aggregate_ts(levels=[0, 1]).shape
    (24, 3)

forecast
--------

forecast with builtin base method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: python

    >>> reconciled_forecast = hts.forecast(h=12, base_method = "arima", hf_method = "comb", weights_method="mint", variance="shrink")
    >>> hts.accuracy(reconciled_forecast, levels=0)


Customize Base forecaster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Also, you can use different base forecaster for different level, just inherent
:class:`~pyhts.forecaster.BaseForecaster` and implement :meth:`~pyhts.forecaster.BaseForecaster.forecast()`
method, and pass list of baseforecasters into `base_forecaster` parameter of :meth:`~pyhts.hts.forecast()`.
See examples below.

Use same forecaster for all levels.

.. code-block:: python

    >>> from pyhts.forecaster import AutoArimaForecaster
    >>> forecaster = AutoArimaForecaster(m=12)
    >>> reconciled_forecast = hts.forecast(h=12, base_forecaster = forecaster, hf_method = "comb", weights_method="mint", variance="shrink")


Use different forecaster for different levels.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. code-block:: python

    >>>  from pyhts.forecaster import AutoArimaForecaster, EtsForecaster
    >>>  arima = AutoArimaForecaster(m=12)
    >>>  ets = EtsForecaster(m=12)
    >>>  base_forecasters = [arima, ets, ets, ets]
    >>> reconciled_forecast = hts.forecast(h=12, base_forecaster = base_forecasters, hf_method = "comb", weights_method="mint", variance="shrink")

Make your own forecasters, you can config you base forecasters, but it must
implement `forecast` method, which at least takes hist, h, keep_fitted as parameters.

* `hist`: history series
* `h`: forecast horizon
* `keep_fitted`: if in_sample fitted values are needed.

This method should return :class:`~numpy.ndarray` as forecasts.

.. code-block:: python

    from pyhts.forecaster import Baseforecaster
    class custom_forecaster(Baseforecaster):
        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2
        def forecast(self, hist, h, keep_fitted, param3, param4):
            return myforecast_method(hist, h, keep_fitted, param3, param4)









