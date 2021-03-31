import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
forecast = importr("forecast")
ts = robjects.r["ts"]


class BaseForecaster:
    """Base Class for forecaster

    """

    def __int__(self, name=""):
        self.name = name

    @classmethod
    def forecast(cls, hts: np.ndarray, h: int, freq: int, keep_fitted=False) -> np.ndarray:
        raise NotImplementedError


class AutoArimaForecaster(BaseForecaster):
    """auto.arima forecaster

    """

    def __int__(self, name="auto.arima"):
        self.name = name

    @classmethod
    def forecast(cls, hist: np.ndarray, h: int, freq: int, keep_fitted=False) -> np.ndarray:
        auto_arima = forecast.auto_arima
        series = ts(FloatVector(hist), frequency=freq)
        model = forecast.forecast(auto_arima(series), h=12)
        if keep_fitted:
            return np.concatenate([np.array(model.rx2["fitted"]), np.array(model.rx2["mean"])])
        else:
            return np.array(model.rx2["mean"])


class EtsForecaster(BaseForecaster):
    """Ets forecaster.

    """

    def __int__(self, name="ets"):
        self.name = name

    @classmethod
    def forecast(cls, hist: np.ndarray, h: int, freq: int, keep_fitted=False) -> np.ndarray:
        ets = forecast.ets
        series = ts(FloatVector(hist), frequency=freq)
        model = forecast.forecast(ets(series), h=12)
        if keep_fitted:
            return np.concatenate([np.array(model.rx2["fitted"]), np.array(model.rx2["mean"])])
        else:
            return np.array(model.rx2["mean"])
