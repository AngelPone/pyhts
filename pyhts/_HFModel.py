import numpy as np
from typing import List, Union, Optional, Iterable
from pyhts._forecaster import BaseForecaster, AutoArimaForecaster
import pyhts._reconciliation as fr
from pyhts._hierarchy import *
import pandas as pd

__all__ = ["HFModel", "TemporalHFModel"]


class HFModel:
    """Model for hierarchical forecasting.

    """

    def __init__(self,
                 hierarchy: Hierarchy,
                 base_forecasters: Union[List[BaseForecaster], str],
                 hf_method: str = 'comb',
                 comb_method: str = 'ols',
                 weights: Optional[Union[str, np.ndarray]] = None,
                 immutable_set: Optional[Iterable[int]] = None):
        """Define a hierarchical forecasting model.

        :param hierarchy: hierarchical structure of the data.
        :param base_forecasters: base forecasting method, list of custom forecaster objects or str :code:`arima` \
        implemented by :code:`statsforecast`.
        :param hf_method: method for hierarchical forecasting, :code:`comb` is only supported for now, which represents\
        optimal combination method.
        :param comb_method: method for forecast reconciliation, ols, wls or mint.
        :param weights: weighting matrix used in wls and mint, "structural" or custom symmetric matrix for wls, \
        "shrinkage", "sample", "variance" for mint.
        :param immutable_set: the subset of time series to be unchanged during reconciliation
        """
        self.hierarchy = hierarchy
        self.base_forecasters = base_forecasters
        self.hf_method = hf_method
        self.comb_method = comb_method
        self.period = self.hierarchy.period
        self.weights = weights
        self.immutable_set = immutable_set
        self.G = None

    def fit(self, ts: pd.DataFrame or np.ndarray, xreg: Optional[np.array] = None, **kwargs):
        """Fit a base forecast model and calculate the reconciliation matrix used for reconciliation.

        :param ts: T * m, each column represents one bottom-level time series. The order of series should be same as \
        the order when defining hierarchy.
        :param xreg: explanatory variables with shape (n, T, k), where n is number of time series, T is history length,\
        and k is dimension of explanatory variables.
        :param kwargs: parameters passed to :code:`base_forecasters`.
        :return: fitted HFModel
        """
        assert self.hierarchy.check_hierarchy(ts), "Only bottom series are needed to fit the model."
        s_matrix = self.hierarchy.s_mat
        n, m = self.hierarchy.s_mat.shape
        if isinstance(ts, np.ndarray):
            ts = ts.dot(s_matrix.T)
        else:
            ts = ts.values.T.dot(s_matrix)

        if isinstance(self.base_forecasters, str):
            if self.base_forecasters == 'arima':
                self.base_forecasters = [
                    AutoArimaForecaster(self.period).fit(ts[:, i], xreg=None if xreg is None else xreg[i],
                                                         **kwargs)
                    for i in range(n)]
            else:
                raise ValueError("This base forecasting method is not supported.")
        elif isinstance(self.base_forecasters, List):
            self.base_forecasters = [
                self.base_forecasters[i].fit(ts[:, i], xreg=None if xreg is None else xreg[i])
                if not self.base_forecasters[i].fitted else self.base_forecasters[i]
                for i in range(n)]
        else:
            raise ValueError("This base forecasting method is not supported.")

        if self.hf_method == "comb":
            if self.comb_method == "ols":
                error = None
            elif self.comb_method == "wls":
                assert self.weights == "structural" or isinstance(self.weights, np.ndarray), "This weighting method for\
                 wls is not supported."
                error = None
            elif self.comb_method == "mint":
                assert self.weights in ["sample", "shrinkage", "variance"] or isinstance(self.weights, np.ndarray), \
                    "This weighting method for mint is not supported"
                error = np.stack([forecaster.residuals for forecaster in self.base_forecasters], axis=0)
            else:
                raise ValueError("This combination method is not supported.")

            self.G = fr.mint(self.hierarchy, error=error, method=self.comb_method, weighting=self.weights,
                             immutable_set=self.immutable_set)
        else:
            raise NotImplementedError("This method is not implemented.")

    def generate_base_forecast(self, horizon: int = 1, xreg=None, **kwargs):
        forecasts = np.stack([self.base_forecasters[i].forecast(h=horizon, xreg=None if xreg is None else xreg[i],
                                                                **kwargs) for i in range(len(self.base_forecasters))])
        return forecasts.T

    def predict(self, horizon: int = 1, xreg=None, **kwargs) -> np.array:
        """Generate horizon-step-ahead reconciled base forecasts of the bottom level.

        :param horizon: forecast horizon
        :param xreg: explanatory variables with shape (n, h, k), where n is number of time series, h is forecast\
        horizon, and k is dimension of explanatory variables.
        :param kwargs: other parameters passed to base forecasters.
        :return: coherent forecasts of the bottom level.
        """
        forecasts = self.generate_base_forecast(horizon=horizon, xreg=xreg, **kwargs)
        return self.G.dot(forecasts.T).T


class TemporalHFModel:
    """Temporal hierarchical forecasting model
    """

    def __init__(self, hierarchy: TemporalHierarchy,
                 base_forecasters: Union[dict, str],
                 hf_method='comb', comb_method='ols', weights=None,
                 immutable_set=None):
        """Constructor

        :param hierarchy: hierarchical structure.
        :param base_forecasters: base forecasters, It should be string or dictionary. \
        If string, it could be "arima" implemented by statsforecast. \
        If dict, its keys should be the level name of the hierarchy, corresponding values represent the fitted \
        or not fitted Forecaster.
        :param hf_method: :code:`comb` for optimal combination
        :param comb_method: :code:`ols`, :code:`wls`, :code:`mint`
        :param weights: weighting strategies, same as HFModel
        :param immutable_set: immutable nodes, same as HFModel
        """

        self.hierarchy = hierarchy
        self.base_forecasters = base_forecasters
        assert hf_method in ["comb"], f"{hf_method} is not implemented!"
        self.hf_method = hf_method
        if hf_method == "comb":
            assert comb_method in ["wls", "ols", "mint"], f"{comb_method} combination method is not implemented"
        self.comb_method = comb_method
        self.weights = weights
        self.immutable_set = immutable_set
        self.G = None

    def _get_residuals(self):
        res_dict = {key: self.base_forecasters[key].residuals for key in self.base_forecasters}
        return self.hierarchy._temporal_dict2array(res_dict)

    def fit(self, ts: np.ndarray, xreg: dict = None, **kwargs):
        """fit base models and reconciliation matrix

        :param ts: univariate time series
        :param xreg: dict containing covariates passed to each level
        :param kwargs: other possible arguments passed to forecaster
        :return:
        """

        ht = self.hierarchy
        ats = ht.aggregate_ts(ts)

        # fit base forecasters
        fcasters = self.base_forecasters
        if self.base_forecasters == 'arima':
            self.base_forecasters = {l: AutoArimaForecaster(ht.period // int(l.split('_')[1])) for l in ats}
        for j in self.base_forecasters:
            if not self.base_forecasters[j].fitted:
                self.base_forecasters[j].fit(ats[j], xreg=xreg[j] if xreg else None, **kwargs)


        # fit reconciliation weights
        if self.hf_method == 'comb':
            if self.comb_method == "ols":
                error = None
            elif self.comb_method == "wls":
                assert self.weights == "structural" or isinstance(self.weights, np.ndarray), "This weighting method for\
                 wls is not supported."
                error = None
            else:
                assert self.weights in ["sample", "shrinkage", "variance"] or isinstance(self.weights, np.ndarray), \
                    "This weighting method for mint is not supported"
                error = self._get_residuals().T
            self.G = fr.mint(self.hierarchy, error=error, method=self.comb_method, weighting=self.weights,
                             immutable_set=self.immutable_set)
        elif self.hf_method == 'bu':
            pass
        elif self.hf_method == 'td':
            pass

    def generate_base_forecast(self, horizon: int = 1, xreg: dict = None, **kwargs) -> dict:
        """generate base forecasts

        :param horizon: horizon for the top level
        :param xreg: covariates dict passed to forecaster
        :param kwargs: other arguments passed to forecaster
        :return: dict containing base forecasts of each level
        """
        agg_periods = [int(i.split('_')[1]) for i in self.hierarchy.level_name]
        hs = {key: agg_periods[0]//int(key.split('_')[1]) * horizon for key in self.base_forecasters}
        return {key: self.base_forecasters[key].forecast(hs[key], xreg[key] if xreg else None, **kwargs) for key in self.base_forecasters}

    def predict(self, horizon: int = 1, xreg: dict = None, **kwargs):
        """predict base forecasts and reconcile them.

        :param horizon: horizon for the top level
        :param xreg: covariates dict passed to forecaster
        :param kwargs: other arguments passed to forecaster
        :return: dict containing reconciled forecasts of each level
        """

        forecasts = self.generate_base_forecast(horizon=horizon, xreg=xreg, **kwargs)
        forecasts = self.hierarchy._temporal_dict2array(forecasts)
        forecasts = self.hierarchy.s_mat.dot(self.G.dot(forecasts.T)).T
        return self.hierarchy._temporal_array2dict(forecasts)
