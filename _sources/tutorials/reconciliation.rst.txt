Forecast reconciliation
=======================

The forecast reconciliation methods supported by `pyhts` are listed below:

1. ols proposed by :ref:`Hyndman et al.(2010)<ols>`
2. structural scaling, known as `WLSs` in :ref:`Wickramasuriya et al.(2019)<mint>`
3. variance scaling, known as `WLSv` in :ref:`Wickramasuriya et al.(2019)<mint>`
4. minimum trace with shrinkage estimator of covariance matrix, known as `MinT(Shrink)` in :ref:`Wickramasuriya et al.(2019)<mint>`
5. minimum trace with covariance matrix, known as `MinT(Sample)` in :ref:`Wickramasuriya et al.(2019)<mint>`


Implementation
--------------

all these reconciliation method can be implemented through an unified interface.

**generate base forecast and reconciled forecast through** :class:`~pyhts.hts.Hts` **object**

.. code-block:: python

    from pyhts.hts import Hts
    hts = Hts.from_hts(series, m=12, nodes=[[2], [2, 2]])
    reconciled_hts_ols = hts.forecast(h=12, base_method='arima', hf_method='comb', comb_method='ols')
    reconciled_hts_wlss = hts.forecast(h=12, base_method='arima', hf_method='comb', comb_method='wls', weights='structural')
    reconciled_hts_wlsv = hts.forecast(h=12, base_method='arima', hf_method='comb', comb_method='mint', weights='variance')
    reconciled_hts_mint_shrink = hts.forecast(h=12, base_method='arima', hf_method='comb', comb_method='mint', weights='shrinkage')
    reconciled_hts_mint_sample = hts.forecast(h=12, base_method='arima', hf_method='comb', comb_method='wls', weights='sample')


**generate reconciled forecast through** :func:`~pyhts.reconciliation.wls`

.. warning::
    Not recommended unless you are familiar with the package.

.. code-block:: python

    import pyhts.reconciliation as fr
    reconciled_hts_ols = fr.wls(hts, base_forecast, method='ols')
    reconciled_hts_wlss = fr.wls(hts, base_forecast, method='wls', weighting='structural')
    reconciled_hts_wlsv = fr.wls(hts, base_forecast, method='mint', weighting='variance')
    reconciled_hts_mint_shrink = fr.wls(hts, base_forecast, method='mint', weighting='shrinkage')
    reconciled_hts_mint_sample = fr.wls(hts, base_forecast, method='wls', weighting='sample')



