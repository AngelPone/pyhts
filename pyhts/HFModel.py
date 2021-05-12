import numpy as np
from pyhts.hierarchy import Hierarchy
from typing import List, Union, Optional
from pyhts.forecaster import BaseForecaster, EtsForecaster, AutoArimaForecaster
import pyhts.reconciliation as fr
import pandas as pd


class HFModel:

    def __init__(self,
                 hierarchy: Hierarchy,
                 base_forecasters: Union[List[BaseForecaster], str],
                 hf_method: str = 'comb',
                 comb_method: str = 'ols',
                 weights: Optional[Union[str, np.ndarray]] = None,
                 constrain_level: int = -1,
                 **kwargs):
        self.hierarchy = hierarchy
        self.base_forecasters = base_forecasters
        self.hf_method = hf_method
        self.comb_method = comb_method
        self.period = self.hierarchy.period
        self.weights = weights
        self.model_params = kwargs
        self.constrain_level = constrain_level
        self.G = None

    def fit(self, ts: pd.DataFrame) -> "HFModel":
        """Fit base forecast model and calculate :math:`SG` used for reconciliation.

        :param ts: DataFrame that each column contains a bottom time series.
        :return: HFModel
        """
        assert self.hierarchy.check_hierarchy(ts)
        s_matrix = self.hierarchy.s_mat
        n, m = self.hierarchy.s_mat.shape
        try:
            ts = s_matrix.dot(ts[self.hierarchy.node_name[-m:]].T)
        except KeyError:
            ts = s_matrix.dot(ts)
        if isinstance(self.base_forecasters, str):
            if self.base_forecasters == 'arima':
                self.base_forecasters = [AutoArimaForecaster(m=self.period, **self.model_params).fit(ts[i, :])
                                         for i in range(n)]
            elif self.base_forecasters == 'ets':
                self.base_forecasters = [EtsForecaster(m=self.period, **self.model_params).fit(ts[i, :]) for i in range(n)]
            else:
                raise ValueError("not supported base method")

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
                    raise ValueError("weights for wls method is not supported")
            elif self.comb_method == "mint":
                sample_forecast = np.stack([model.fitted for model in self.base_forecasters], axis=0)
                sample_error = ts - sample_forecast
                self.G = fr.wls(self.hierarchy, sample_error, method="mint", weighting=self.weights,
                                constraint_level=self.constrain_level)
            else:
                raise ValueError("this comination method is not supported")
        else:
            raise NotImplementedError("this method is not implemented")
        return self

    def generate_base_forecast(self, horizon: int = 1, **kwargs):
        forecasts = np.stack([model.forecast(h=horizon, **kwargs) for model in self.base_forecasters])
        return forecasts

    def predict(self, horizon: int = 1, **kwargs):
        forecasts = self.generate_base_forecast(horizon=horizon, **kwargs)
        return self.G.dot(forecasts).T


