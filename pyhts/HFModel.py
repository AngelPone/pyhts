import numpy as np
from typing import List, Union, Optional
from .forecaster import BaseForecaster, EtsForecaster, AutoArimaForecaster
from . import reconciliation as fr
from .hierarchy import Hierarchy
import pandas as pd


class HFModel:
    """Model for hierarchical forecasting.

    """

    def __init__(self,
                 hierarchy: Hierarchy,
                 base_forecasters: Union[List[BaseForecaster], str],
                 hf_method: str = 'comb',
                 comb_method: str = 'ols',
                 weights: Optional[Union[str, np.ndarray]] = None,
                 constrain_level: int = -1):
        """Define a hierarchical forecasting model.

        :param hierarchy: hierarchical structure of the data.
        :param base_forecasters: base forecasting method, custom forecasters or arima, ets, implemented by :code:`forecast` in R called by :code:`rpy2`
        :param hf_method: method for hierarchical forecasting, :code:`comb` is only supported for now, which represents optimal combination method.
        :param comb_method: method for forecast reconciliation, ols, wls or mint.
        :param weights: weighting matrix used in wls and mint, "structural" or custom symmetric matrix for wls. "shrinkage", "sample", "variance" for mint.
        :param constrain_level: -1 means no constraints.
        """
        self.hierarchy = hierarchy
        self.base_forecasters = base_forecasters
        self.hf_method = hf_method
        self.comb_method = comb_method
        self.period = self.hierarchy.period
        self.weights = weights
        self.constrain_level = constrain_level
        self.G = None

    def fit(self, ts: pd.DataFrame, x_reg: Optional[np.array]=None, **kwargs) -> "HFModel":
        """Fit a base forecast model and calculate the reconciliation matrix used for reconciliation.

        :param ts: a DataFrame in which each column contains a bottom-level time series.
        :param x_reg: explantory variables with shape (n, T, k), where n is number of time series, T is history length, and k is dimension of explantory variables.
        :return: fitted HFModel
        """
        assert self.hierarchy.check_hierarchy(ts), "Only bottom series are needed to fit the model."
        s_matrix = self.hierarchy.s_mat
        n, m = self.hierarchy.s_mat.shape
        try:
            ts = s_matrix.dot(ts[self.hierarchy.node_name[-m:]].T)
        except KeyError:
            ts = s_matrix.dot(ts)
        if isinstance(self.base_forecasters, str):
            if self.base_forecasters == 'arima':
                self.base_forecasters = [AutoArimaForecaster(m=self.period).fit(ts[i, :], x_reg=None if x_reg is None else x_reg[i])
                                         for i in range(n)]
            elif self.base_forecasters == 'ets':
                self.base_forecasters = [EtsForecaster(m=self.period).fit(ts[i, :], x_reg=None if x_reg is None else x_reg[i])
                                         for i in range(n)]
            else:
                raise ValueError("This base forecasting method is not supported.")
        elif isinstance(self.base_forecasters, List):
            if len(self.base_forecasters) == 1:
                self.base_forecasters = [self.base_forecasters[0]().fit(ts[i, :], x_reg=None if x_reg is None else x_reg[i])
                                         for i in range(n)]
            if len(self.base_forecasters) == self.hierarchy.level_n:
                self.base_forecasters = [self.base_forecasters[self.hierarchy.node_level[i]]().fit(ts[i, :], x_reg=None if x_reg is None else x_reg[i])
                                         for i in range(n)]
            if len(self.base_forecasters) == n:
                self.base_forecasters = [self.base_forecasters[i]().fit(ts[i, :], x_reg=None if x_reg is None else x_reg[i])
                                         for i in range(n)]
        else:
            raise ValueError("This base forecasting method is not supported.")

        if self.hf_method == "comb":
            if self.comb_method == "ols":
                self.G = fr.wls(self.hierarchy, error=None, method="ols",
                                constraint_level=self.constrain_level)
            elif self.comb_method == "wls":
                if self.weights == "structural":
                    self.G = fr.wls(self.hierarchy, error=None, method="wls", weighting="structural",
                                    constraint_level=self.constrain_level)
                elif isinstance(self.weights, np.ndarray):
                    self.G = fr.wls(self.hierarchy, error=None, method="wls", weighting=self.weights,
                                    constraint_level=self.constrain_level)
                else:
                    raise ValueError("This weighting method for wls is not supported.")
            elif self.comb_method == "mint":
                sample_forecast = np.stack([model.fitted for model in self.base_forecasters], axis=0)
                sample_error = ts - sample_forecast
                self.G = fr.wls(self.hierarchy, sample_error, method="mint", weighting=self.weights,
                                constraint_level=self.constrain_level)
            else:
                raise ValueError("This comination method is not supported.")
        else:
            raise NotImplementedError("This method is not implemented.")
        return self

    def generate_base_forecast(self, horizon: int = 1, x_reg=None, **kwargs):
        forecasts = np.stack([self.base_forecasters[i].forecast(h=horizon, x_reg=None if x_reg is None else x_reg[i],
                                                                **kwargs) for i in range(len(self.base_forecasters))])
        return forecasts.T

    def predict(self, horizon: int = 1, x_reg=None, **kwargs) -> np.array:
        """Generate horizon-step-ahead reconciled base forecasts of the bottom level.

        :param horizon: forecast horizon
        :param x_reg: explantory variables with shape (n, h, k), where n is number of time series, h is forecast horizon, and k is dimension of explantory variables.
        :param kwargs: other parameters passed to base forecasters.
        :return: coherent forecasts of the bottom level.
        """
        forecasts = self.generate_base_forecast(horizon=horizon, x_reg=x_reg, **kwargs)
        return self.G.dot(forecasts.T).T
