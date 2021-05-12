import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
forecast = importr("forecast")
ts = robjects.r["ts"]


class BaseForecaster:
    """Base Class for forecaster

    """
    def __init__(self, m: int = 12):
        self.m = m
        self.model = None
        self.hist = None
        self.fitted = None

    def forecast(self, h: int, **kwargs) -> np.ndarray:
        raise NotImplementedError


class AutoArimaForecaster(BaseForecaster):
    """auto.arima forecaster.

    """

    def __init__(self, m: int = 12, **kwargs):
        super().__init__(m)
        self.model_params = kwargs

    def fit(self, hist: np.ndarray):
        self.hist = ts(FloatVector(hist), frequency=self.m)
        arima = forecast.auto_arima
        self.model = arima(self.hist, **self.model_params)
        self.fitted = self.model.rx2["fitted"]
        return self

    def forecast(self, h: int, **kwargs) -> np.ndarray:
        return np.array(forecast.forecast(self.model, h=h, **kwargs).rx2["mean"])


class EtsForecaster(BaseForecaster):
    """Ets forecaster.

    """

    def __init__(self, m: int = 12, **kwargs):
        super().__init__(m)
        self.model_params = kwargs

    def fit(self, hist: np.ndarray):
        self.hist = ts(FloatVector(hist), frequency=self.m)
        ets = forecast.ets
        self.model = ets(self.hist, **self.model_params)
        self.fitted = self.model.rx2["fitted"]
        return self

    def forecast(self, h: int, **kwargs) -> np.ndarray:
        return np.array(forecast.forecast(self.model, h=h, **kwargs).rx2["mean"])
