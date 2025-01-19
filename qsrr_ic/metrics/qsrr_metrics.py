import numpy as np
from numpy import ndarray


class Metrics:

    @staticmethod
    def perc_mse(y: ndarray, y_hat: ndarray) -> float:
        """
        Percentage Mean Squared Error (MSE).
        """
        if any(y == 0):
            return np.inf
        return np.mean(((y_hat - y) / y) ** 2) * 100

    @staticmethod
    def perc_rmse(y: ndarray, y_hat: ndarray) -> float:
        """
        Percentage Root Mean Squared Error (RMSE).
        """
        if any(y == 0):
            return np.inf
        return np.sqrt(np.mean(((y_hat - y) / y) ** 2)) * 100

    @staticmethod
    def r2(y: ndarray, y_hat: ndarray) -> float:
        """
        Coefficient of determination(R^2)
        """
        return 1 - (np.sum((y.ravel() - y_hat.ravel()) ** 2) / np.sum((y.ravel() - np.mean(y.ravel())) ** 2))

    @staticmethod
    def mae(y: ndarray, y_hat: ndarray) -> float:
        """
        Mean Absolute Error (MAE).
        """
        return np.mean(abs(y_hat.ravel() - y.ravel()))

    @staticmethod
    def mse(y: ndarray, y_hat: ndarray) -> float:
        """
        Mean Squared Error (MSE).
        """
        return np.mean((y_hat - y) ** 2)

    @staticmethod
    def rmse(y: ndarray, y_hat: ndarray) -> float:
        """
        Root Mean Squared Error (RMSE).
        """
        return np.sqrt(Metrics.mse(y, y_hat))
