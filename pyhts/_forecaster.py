import numpy as np
from statsforecast.arima import auto_arima_f, forecast_arima


__all__ = [
    "BaseForecaster", "AutoArimaForecaster"
]


class BaseForecaster:
    """Base class for forecasters.

    """

    def __init__(self):
        self.fitted = None

    def fit(self, hist, **kwargs):
        raise NotImplementedError

    def forecast(self, h: int, **kwargs) -> np.ndarray:
        raise NotImplementedError


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
