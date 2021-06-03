import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
forecast = importr("forecast")
ts = robjects.r["ts"]


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
    """auto.arima forecaster.

    """

    def __init__(self, m: int = 1):
        super().__init__()
        self.m = m
        self.hist = None
        self.model = None

    def fit(self, hist: np.ndarray, x_reg=None, **kwargs):
        self.hist = ts(FloatVector(hist), frequency=self.m)
        arima = forecast.auto_arima
        self.model = arima(self.hist, xreg=x_reg if x_reg is not None else robjects.r('NULL'), **kwargs)
        self.fitted = self.model.rx2["fitted"]
        return self

    def forecast(self, h: int, x_reg=None, **kwargs) -> np.ndarray:
        return np.array(forecast.forecast(self.model, h=h, xreg=x_reg if x_reg is not None else robjects.r('NULL'),
                                          **kwargs).rx2["mean"])


class EtsForecaster(BaseForecaster):
    """ets forecaster.

    """

    def __init__(self, m: int = 1):
        super().__init__()
        self.m = m
        self.hist = None
        self.model = None

    def fit(self, hist: np.ndarray, x_reg=None, **kwargs):
        self.hist = ts(FloatVector(hist), frequency=self.m)
        ets = forecast.ets
        self.model = ets(self.hist, **kwargs)
        self.fitted = self.model.rx2["fitted"]
        return self

    def forecast(self, h: int, **kwargs) -> np.ndarray:
        return np.array(forecast.forecast(self.model, h=h).rx2["mean"])
