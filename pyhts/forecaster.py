import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
forecast = importr("forecast")
ts = robjects.r["ts"]


class BaseForecaster:
    """Base Class for forecaster

    """

    def forecast(self, hist: np.ndarray, h: int, keep_fitted=False) -> np.ndarray:
        raise NotImplementedError


class AutoArimaForecaster(BaseForecaster):
    """auto.arima forecaster

    """

    def __init__(self, m: int = 12):
        self.m = m

    def forecast(self, hist: np.ndarray, h: int, keep_fitted=False) -> np.ndarray:
        auto_arima = forecast.auto_arima
        series = ts(FloatVector(hist), frequency=self.m)
        model = forecast.forecast(auto_arima(series), h=12)
        if keep_fitted:
            return np.concatenate([np.array(model.rx2["fitted"]), np.array(model.rx2["mean"])])
        else:
            return np.array(model.rx2["mean"])


class EtsForecaster(BaseForecaster):
    """Ets forecaster.

    """

    def __init__(self, m: int = 12):
        self.m = m

    def forecast(self, hist: np.ndarray, h: int, keep_fitted=False) -> np.ndarray:
        ets = forecast.ets
        series = ts(FloatVector(hist), frequency=self.m)
        model = forecast.forecast(ets(series), h=12)
        if keep_fitted:
            return np.concatenate([np.array(model.rx2["fitted"]), np.array(model.rx2["mean"])])
        else:
            return np.array(model.rx2["mean"])
