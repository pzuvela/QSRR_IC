from typing import Optional

import numpy as np

from .qsrr_data import QsrrData
from qsrr_ic.metrics import Metrics


class QsrrMetrics:

    def __init__(
        self,
        qsrr_data: QsrrData,
        qsrr_predictions: QsrrData
    ):
        self.qsrr_data = qsrr_data
        self.qsrr_predictions = qsrr_predictions

        self.__perc_mse: Optional[float] = None
        self.__perc_rmse: Optional[float] = None
        self.__rmse: Optional[float] = None

    @property
    def perc_mse(self) -> float:
        if self.__perc_mse is None:
            self.__perc_mse = Metrics.perc_mse(y=self.qsrr_data.y, y_hat=self.qsrr_predictions.y)
        return self.__perc_mse

    @perc_mse.setter
    def perc_mse(self, value: float):
        raise ValueError("Percentage MSE cannot be set !")

    @property
    def perc_rmse(self) -> float:
        if self.__perc_rmse is None:
            self.__perc_rmse = Metrics.perc_rmse(y=self.qsrr_data.y, y_hat=self.qsrr_predictions.y)
        return self.__perc_rmse

    @perc_rmse.setter
    def perc_rmse(self, value: float):
        raise ValueError("Percentage RMSE cannot be set !")

    @property
    def rmse(self) -> float:
        if self.__rmse is None:
            self.__rmse = Metrics.rmse(y=self.qsrr_data.y, y_hat=self.qsrr_predictions.y)
        return self.__perc_mse

    @rmse.setter
    def rmse(self, value: float):
        raise ValueError("RMSE cannot be set !")
