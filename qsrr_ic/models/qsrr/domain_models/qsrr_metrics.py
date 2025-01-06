from typing import Optional

import numpy as np

from .qsrr_data import QsrrData


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
            self.__perc_mse = np.mean((self.qsrr_predictions.y - self.qsrr_data.y) / self.qsrr_data.y)  * 100
        return self.__perc_mse

    @perc_mse.setter
    def perc_mse(self, value: float):
        raise ValueError("Percentage MSE cannot be set !")

    @property
    def perc_rmse(self) -> float:
        if self.__perc_rmse is None:
            self.__perc_rmse = np.sqrt(
                np.mean(((self.qsrr_predictions.y - self.qsrr_data.y) / self.qsrr_data.y) ** 2)
            ) * 100
        return self.__perc_rmse

    @perc_rmse.setter
    def perc_rmse(self, value: float):
        raise ValueError("Percentage RMSE cannot be set !")

    @property
    def rmse(self) -> float:
        if self.__rmse is None:
            self.__rmse = np.sqrt(
                np.mean((self.qsrr_predictions.y - self.qsrr_data.y) ** 2)
            )
        return self.__perc_mse

    @rmse.setter
    def rmse(self, value: float):
        raise ValueError("RMSE cannot be set !")
