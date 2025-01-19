from typing import (
    Dict,
    Optional
)

import numpy as np

import pandas as pd
from pandas import DataFrame

from .qsrr_data import QsrrData
from qsrr_ic.error_handling import ErrorHandling
from qsrr_ic.metrics import Metrics


class QsrrMetrics:

    def __init__(
        self,
        qsrr_data: QsrrData,
        qsrr_predictions: QsrrData
    ):
        self.qsrr_data = qsrr_data
        self.qsrr_predictions = qsrr_predictions

        self.__r2: Optional[float] = None
        self.__perc_mse: Optional[float] = None
        self.__perc_rmse: Optional[float] = None
        self.__mae: Optional[float] = None
        self.__mse: Optional[float] = None
        self.__rmse: Optional[float] = None

    @property
    def perc_mse(self) -> float:
        if self.__perc_mse is None:
            self.__perc_mse = Metrics.perc_mse(y=self.qsrr_data.y, y_hat=self.qsrr_predictions.y)
        return self.__perc_mse

    @perc_mse.setter
    def perc_mse(self, value: float):
        raise ErrorHandling.get_property_value_error("perc_mse")

    @property
    def perc_rmse(self) -> float:
        if self.__perc_rmse is None:
            self.__perc_rmse = Metrics.perc_rmse(y=self.qsrr_data.y, y_hat=self.qsrr_predictions.y)
        return self.__perc_rmse

    @perc_rmse.setter
    def perc_rmse(self, value: float):
        raise ErrorHandling.get_property_value_error("perc_rmse")

    @property
    def r2(self) -> float:
        if self.__r2 is None:
            self.__r2 = Metrics.r2(y=self.qsrr_data.y, y_hat=self.qsrr_predictions.y)
        return self.__r2

    @r2.setter
    def r2(self, value: float):
        raise ErrorHandling.get_property_value_error("r2")

    @property
    def mae(self) -> float:
        if self.__mae is None:
            self.__mae = Metrics.mae(y=self.qsrr_data.y, y_hat=self.qsrr_predictions.y)
        return self.__mae

    @mae.setter
    def mae(self, value: float):
        raise ErrorHandling.get_property_value_error("mae")

    @property
    def mse(self) -> float:
        if self.__mse is None:
            self.__mse = Metrics.mse(y=self.qsrr_data.y, y_hat=self.qsrr_predictions.y)
        return self.__mse

    @mse.setter
    def mse(self, value: float):
        raise ErrorHandling.get_property_value_error("mse")

    @property
    def rmse(self) -> float:
        if self.__rmse is None:
            self.__rmse = Metrics.rmse(y=self.qsrr_data.y, y_hat=self.qsrr_predictions.y)
        return self.__rmse

    @rmse.setter
    def rmse(self, value: float):
        raise ErrorHandling.get_property_value_error("rmse")

    def to_dict(self) -> Dict[str, float]:

        metrics_dict = {
            "R2": self.r2,
            "MAE": self.mae,
            "MSE": self.mse,
            "RMSE": self.rmse
        }

        if self.perc_mse != np.inf:
            metrics_dict["%MSE"] = self.perc_mse

        if self.perc_rmse != np.inf:
            metrics_dict["%RMSE"] = self.perc_rmse

        return metrics_dict

    def to_df(self) -> DataFrame:
        return pd.DataFrame.from_dict(
            self.to_dict(),
            orient="index"
        )
