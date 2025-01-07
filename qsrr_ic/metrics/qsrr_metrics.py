import numpy as np
from numpy import ndarray


class Metrics:

    @staticmethod
    def perc_mse(y: ndarray, y_hat: ndarray) -> float:
        """
        Percentage Mean Squared Error (MSE).
        """
        return np.mean((y_hat - y) ** 2 / y ** 2) * 100

    @staticmethod
    def perc_rmse(y: ndarray, y_hat: ndarray) -> float:
        """
        Percentage Root Mean Squared Error (RMSE).
        """
        return np.sqrt(np.mean((y_hat - y) ** 2 / y ** 2)) * 100

    @staticmethod
    def rmse(y: ndarray, y_hat: ndarray) -> float:
        """
        Root Mean Squared Error (RMSE).
        """
        return np.sqrt(np.mean((y_hat - y) ** 2))
