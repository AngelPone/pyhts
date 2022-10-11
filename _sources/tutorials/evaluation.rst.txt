Model Evaluation
================

After obtaining base forecasts or coherent reconciled forecasts, you can use evaluate the forecasting accuracy
through :meth:`pyhts.Hierarchy.accuracy_base`, :meth:`pyhts.Hierarchy.accuracy`, or :meth:`pyhts.TemporalHierarchy.accuracy` (for temporal hierarchies).


.. automethod:: pyhts.Hierarchy.accuracy_base

.. automethod:: pyhts.Hierarchy.accuracy

.. automethod:: pyhts.TemporalHierarchy.accuracy


Assuming :code:`ht` is a defined hierarchy, :code:`model` is a fitted :class:`~pyhts.HFModel`, :code:`test`
is real observations in the forecasting horizon. :code:`train` is the history bottom time series.



.. code-block:: python

    >>> forecasts = model.predict(h=12)
    >>> base_forecasts = model.generate_base_forecasts(h=12)
    >>> ht.accuracy_base(test, base_forecasts, hist=train, levels=[0], measures=['mase', 'rmse'])
    >>> ht.accuracy(test, forecasts, hist=train, levels=[0], measures=['mase', 'rmse'])


Supported forecasting measurements:

.. autofunction:: pyhts.mase

.. autofunction:: pyhts.rmse

.. autofunction:: pyhts.mse

.. autofunction:: pyhts.mape

.. autofunction:: pyhts.mae

.. autofunction:: pyhts.smape
