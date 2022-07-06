Quick Start
===========


Introduction
------------

:code:`pyths` is a python package used for hierarchical forecasting, it implements multiple forecast reconciliation methods which are popular in forecasting literatures such as
ols [#ols]_ 、wls [#wls]_ 、mint [#mint]_ et al.

**features**:

- support popular forecast reconciliation models in the literature, e.g. ols, wls, mint et al. Temporal Hierarchy will be supported in the future.

- multiple methods for the construction of hierarchy.

- use different base forecasters for different hierarchical levels.

- familiar sklearn-like API.

The steps of using this package are as follows:

1. Define a hierarchy structure according to your data.

2. Define a :class:`~pyhts.HFModel`

3. fit the :class:`~pyhts.HFModel` using history time series

4. generate base forecasts and reconcile the forecasts to obtain coherent point forecasts.

Define the Hierarchy
--------------------

You can use classmethods of :class:`pyhts.Hierarchy` to define hierarchy structure. see details in :doc:`/tutorials/hierarchy`

As an example of the hierarchy shown in the figure below, let’s construct the hierarchy using :class:`pyhts.hierarchy.Hierarchy.from_node_list()` and :class:`pyhts.hierarchy.Hierarchy.from_names()`.

.. image:: ../media/01_hierarchy.png

.. code-block:: python

    from pyhts import Hierarchy
    node_list = [[2], [2, 2]]
    period = 12
    hierarchy = Hierarchy.from_node_list(node_list, period=period)

    col_names = ['AA', 'AB', 'BC', 'BD']
    hierarchy2 = Hierarchy.from_names(col_names, chars=[1,1], period=period)
    print(hierarchy2.node_name)
    # array(['Total', 'A', 'B', 'AA', 'AB', 'BC', 'BD'], dtype='<U5')

where period is frequency of time series, m=12 means monthly series.

Define HFModel
---------------

Let’s define a simple ols reconciliation model that use :code:`auto.arima` as the base forecasting model, see details in :doc:`/tutorials/hfmodel` .

.. code-block:: python

    from pyhts import HFModel
    ols_model = HFModel(hierarchy=hierarchy, base_forecasters='arima', hf_method='comb', comb_method='ols')

where

- :code:`hierarchy` is the hierarchy define above.
- :code:`base_forecasters` are base methods that used to generate base forecasts. :code:`arima` and :code:`ets` are supported for now, which are implemented by :code:`forecast` package in R called by :code:`rpy2`.You can also define your custom base forecasters for each level, see details in :ref:`base`.
- :code:`hf_model` is the method used for hierarchical forecasting, :code:`comb` that means forecast reconciliation is supported for now. Classical methods such as Top-Down、Bottom-up and middle-out will be supported in the future.
- :code:`comb_method` is the forecast reconciliation method. mint、wls、ols are supported. see details in :doc:`/tutorials/hfmodel`.

fit model
---------

:meth:`pyhts.HFModel.fit()` would fit base forecasting models for each time series and compute the reconciliation matrix.

.. code-block:: python

    import numpy as np
    data = np.random.random((108, 4))
    train = data[:-12, :]
    test = data[-12:, :]
    model.fit(train)

forecast
--------

:meth:`pyhts.HFModel.forecast()` would generate base forecasts for each time series and reconcile base forecasts to get coherent forecasts.

.. code-block:: python

    reconciled_forecasts = model.predict(horizon=12)
    print(reconciled_forecasts.shape)
    # (12, 4)

:code:`reconciled_forecasts` just contain reconciled forecasts of bottom level, you can use :meth:`~pyhts.Hierarchy.aggregate_ts()` to get reconciled forecasts of all levels.

.. code-block:: python

    reconciled_forecasts_all_levels = hierarchy.aggregate_ts(reconciled_forecasts)
    # (12, 7)

measurement
-----------
You can evaluate forecasting accuracy of both base forecasts and reconciled forecasts, using :meth:`~pyhts.Hierarchy.accuray_base()` and :meth:`~pyhts.Hierarchy.accuracy()` respectively.

.. code-block:: python

    base_forecasts = model.generate_base_forecasts(horizon=12)
    hierarchy.accuracy_base(test, base_forecasts, hist=train, levels=None, measure=['mase', 'mape'])
    hierarchy.accuracy(test, reconciled_forecasts, hist=train, levels=None, measure=['mase', 'mape'])

where :code:`levels=None` means accuracy of all levels are returned. :code:`hist` are history time series that are needed by :code:`mase` measure.


.. [#ols] Hyndman, R. A. Ahmed, G. Athanasopoulos, and H. L. Shang, “Optimal combination forecasts for hierarchical time series,” Computational Statistics & Data Analysis, vol. 55, no. 9, pp. 2579–2589, Sep. 2011, doi: 10.1016/j.csda.2011.03.006.


.. [#wls] Panagiotelis, G. Athanasopoulos, P. Gamakumara, and R. J. Hyndman, “Forecast reconciliation: A geometric view with new insights on bias correction,” International Journal of Forecasting, vol. 37, no. 1, pp. 343–359, Jan. 2021, doi: 10.1016/j.ijforecast.2020.06.004.


.. [#mint] Wickramasuriya, G. Athanasopoulos, and R. J. Hyndman, “Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series Through Trace Minimization,” Journal of the American Statistical Association, vol. 114, no. 526, pp. 804–819, Apr. 2019, doi: 10.1080/01621459.2018.1448825.