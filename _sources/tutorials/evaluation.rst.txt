Model Evaluation
================

After obtaining base forecasts or coherent reconciled forecasts, you can use evaluate the forecasting accuracy
through :meth:`pyhts.hierarchy.Hierarchy.accuracy_base` or :meth:`pyhts.hierarchy.Hierarchy.accuracy` respectively.


.. automethod:: pyhts.hierarchy.Hierarchy.accuracy_base

.. automethod:: pyhts.hierarchy.Hierarchy.accuracy


Assuming :code:`ht` is a defined hierarchy, :code:`model` is a fitted :class:`~pyhts.HFModel.HFModel`, :code:`test`
is real observations in the forecasting horizon. :code:`train` is the history bottom time series.



.. code-block:: python

    >>> forecasts = model.predict(h=12)
    >>> base_forecasts = model.generate_base_forecasts(h=12)
    >>> ht.accuracy_base(test, base_forecasts, hist=train, levels=[0], measures=['mase', 'rmse'])
    >>> ht.accuracy(test, forecasts, hist=train, levels=[0], measures=['mase', 'rmse'])


Supported forecasting measurements:

.. autofunction:: pyhts.accuracy.mase

.. autofunction:: pyhts.accuracy.rmse

.. autofunction:: pyhts.accuracy.mse

.. autofunction:: pyhts.accuracy.mape

.. autofunction:: pyhts.accuracy.mae

.. autofunction:: pyhts.accuracy.smape
