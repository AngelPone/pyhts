Hierarchical Forecasting Model
##############################

Hierarchical Forecasting Model (HFModel) is defined by :class:`pyhts.HFModel`

HFModel API
==============

.. autoclass:: pyhts.HFModel
    :members: __init__, fit, predict

Examples
========

load Tourism dataset

.. code-block:: python

    >>> from pyhts import load_tourism
    >>> from pyhts import Hierarchy
    >>> dataset = load_tourism()
    >>> dataset.columns
    Index(['AAA', 'AAB', 'ABA', 'ABB', 'ACA', 'ADA', 'ADB', 'ADC', 'ADD', 'AEA',
           'AEB', 'AEC', 'AED', 'AFA', 'BAA', 'BAB', 'BAC', 'BBA', 'BCA', 'BCB',
           'BCC', 'BDA', 'BDB', 'BDC', 'BDD', 'BDE', 'BDF', 'BEA', 'BEB', 'BEC',
           'BED', 'BEE', 'BEF', 'BEG', 'BEH', 'CAA', 'CAB', 'CAC', 'CBA', 'CBB',
           'CBC', 'CBD', 'CCA', 'CCB', 'CCC', 'CDA', 'CDB', 'DAA', 'DAB', 'DAC',
           'DBA', 'DBB', 'DBC', 'DCA', 'DCB', 'DCC', 'DCD', 'DDA', 'DDB', 'EAA',
           'EAB', 'EAC', 'EBA', 'ECA', 'FAA', 'FBA', 'FBB', 'FCA', 'FCB', 'GAA',
           'GAB', 'GAC', 'GBA', 'GBB', 'GBC', 'GBD'],
          dtype='object')
    >>> train = dataset.iloc[:-12,]
    >>> test = dataset.iloc[-12:,]
    >>> ht = Hierarchy.from_names(dataset.columns, [1, 1, 1], period=12)

Define HFModel
---------------

Then, define the hierarchical forecasting model.

**ols** proposed by Hyndman et al.(2011) [#ols]_

.. code-block:: python

    model = HFModel(ht, base_forecasters = "arima", hf_method="comb", comb_method="ols")

**WLS_Structural**, also known as :math:`WLS_s` in  Wickramasuriya et al.(2019) [#mint]_

.. code-block:: python

    model = HFModel(ht, "arima", "comb", "wls", weights="structural")

**Mint_variance**, also known as :math:`WLS_v` in Wickramasuriya et al.(2019) [#mint]_

.. code-block:: python

    model = HFModel(ht, "arima", "comb", "mint", weights="variance")

**Mint_sample**, also known as Mint_sample in Wickramasuriya et al.(2019) [#mint]_

.. code-block:: python

    model = HFModel(ht, "arima", "comb", "mint", weights="sample")

**Mint_shrinakge**, also known as Mint_shrinkage in Wickramasuriya et al.(2019) [#mint]_


.. code-block:: python

    >>> model = HFModel(ht, "arima", "comb", "mint", weights="shrinkage")

.. _base:

customize base forecasters
^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: update docs

To generate base forecasts, you can use :code:`arima`, which are implemented by
:code:`statsforecast`, or you can define your own base forecasters using whatever
forecasting methods you want, machine learning or deeplearning methods.

- :code:`fit(ts, xreg, **kwargs)` is needed for the base forecaster, which is used to fit base forecasting model and generate insample forecasts :code:`fitted`, which is necessary when :code:`comb_method` is :code:`mint`.

- :code:`forecast(h, xreg, **kwargs)` is needed for the base forecaster, which is used to generate base forecasts.

As an example, here is the implementation of :class:`~pyhts.AutoArimaBaseForecaster`

.. code-block:: python

    class AutoArimaForecaster(BaseForecaster):
    """autoarima forecaster, adapted from statsforecast.arima.auto_arima_f

    """

    def __init__(self, period: int = 1):
        super().__init__()
        self.period = period
        self.hist = None
        self.model = None

    def fit(self, hist: np.ndarray, xreg=None, **kwargs):
        """fit the AutoArimaForecaster

        :param hist: observations
        :param xreg: covariates of arima model
        :param kwargs: other parameters passed to  `statsforecast.arima.auto_arima_f`
        :return: self
        """
        self.hist = hist
        self.model = auto_arima_f(hist, period=self.period, xreg=xreg, **kwargs)
        self.fitted = True
        return self

    def forecast(self, h: int, xreg=None, **kwargs) -> np.ndarray:
        """forecast

        :param h: forecast horizons
        :param xreg: covariates
        :param kwargs: other parameters passed to `statsforecast.arima.forecast_arima`
        :return:
        """
        return forecast_arima(self.model, h, xreg=xreg, **kwargs)['mean']

Assuming you have trained a global XGBoost model, you can wrap the xgboost model and pass it to
base_forecasters.

.. code-block:: python

    class XGBoostWrapper:

        def __init__(self):
            self._model = trained_xgbmodel
            self.fitted = None

        def fit(self, hist, x_reg, **kwargs):
            self.fitted = self._model.predict(x_reg)

        def forecast(self, h: int, x_reg, **kwargs):
            return self._model.predict(x_reg)

    xgb_model = HFModel(ht, base_forecasters=[XGBoostWrapper], hf_method='comb', comb_method='ols')


Fit HFModel
-----------

:meth:`~pyhts.HFModel.fit()` will fit the base forecasters and reconcile the incoherent base forecasts to coherent forecasts.

Here is an example that fit a mint shrinkage model with :code:`AAA` type :code:`ets` base forecasters.

.. code-block::

    >>> ols_model = HFModel(ht, "ets", "comb", "mint", weights="shrinkage")
    >>> ols_model.fit(train, model ='AAA')

the parameter :code:`model` will be passed to :code:`ETSBaseForecaster().fit()` then be passed to
:code:`ets()` function in :code:`forecast` package.

Also, you can specify the parameter :code:`model` in your customize BaseForecasters, as mentioned above.


Forecast HFModel
----------------

:meth:`~pyhts.HFModel.forecast` will generate h-step-ahead base forecasts and reconcile
the base forecasts to coherent forecasts in bottom level.

    >>> forecasts = ols_model.forecast(horizon=12)
    >>> all_level_forecasts = ht.aggregate_ts(forecasts)









References
----------

.. [#ols] Hyndman, R. A. Ahmed, G. Athanasopoulos, and H. L. Shang, “Optimal combination forecasts for hierarchical time series,” Computational Statistics & Data Analysis, vol. 55, no. 9, pp. 2579–2589, Sep. 2011, doi: 10.1016/j.csda.2011.03.006.

.. [#mint] Wickramasuriya, G. Athanasopoulos, and R. J. Hyndman, “Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series Through Trace Minimization,” Journal of the American Statistical Association, vol. 114, no. 526, pp. 804–819, Apr. 2019, doi: 10.1080/01621459.2018.1448825.