from dataclasses import dataclass

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


@dataclass
class WilliamsData:
    """
    Data class for encapsulating input data and related parameters for WilliamsModel.
    """
    x1: np.ndarray  # Training input matrix
    x2: np.ndarray  # Testing input matrix
    y1: np.ndarray  # True training output values
    y1_hat: np.ndarray  # Predicted training output values
    y2: np.ndarray  # True testing output values
    y2_hat: np.ndarray  # Predicted testing output values


class WilliamsModel:
    """
    Model responsible for computing residuals, hat matrices, and critical values with lazy loading.
    """

    def __init__(self, data: WilliamsData):
        """
        Initialize the WilliamsModel with a WilliamsData object.

        Args:
            data (WilliamsData): The input data and parameters for the model.
        """
        self._data = data
        self._hat_matrix = None
        self._residuals = None
        self._critical_hat_value = None

    @property
    def hat_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Lazy-loaded property for the hat matrix (leverages).

        Returns:
            tuple[np.ndarray, np.ndarray]: Training and testing leverages.
        """
        if self._hat_matrix is None:
            self._hat_matrix = self._compute_hat_matrix()
        return self._hat_matrix

    @hat_matrix.setter
    def hat_matrix(self, value):
        raise AttributeError("The 'hat_matrix' property is read-only and cannot be set.")

    @property
    def residuals(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Lazy-loaded property for the standardized residuals.

        Returns:
            tuple[np.ndarray, np.ndarray]: Training and testing residuals.
        """
        if self._residuals is None:
            self._residuals = self._compute_residuals()
        return self._residuals

    @residuals.setter
    def residuals(self, value):
        raise AttributeError("The 'residuals' property is read-only and cannot be set.")

    @property
    def critical_hat_value(self) -> float:
        """
        Lazy-loaded property for the critical hat value.

        Returns:
            float: Critical leverage value.
        """
        if self._critical_hat_value is None:
            self._critical_hat_value = self._compute_critical_hat_value()
        return self._critical_hat_value

    @critical_hat_value.setter
    def critical_hat_value(self, value):
        raise AttributeError("The 'critical_hat_value' property is read-only and cannot be set.")

    def _compute_hat_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the training and testing hat matrices.

        Returns:
            tuple[np.ndarray, np.ndarray]: Training and testing leverages.
        """
        x1, x2 = self._data.x1, self._data.x2
        if isinstance(x1, pd.DataFrame):
            x1 = x1.values
            x2 = x2.values
        h_core = np.linalg.pinv(np.matmul(np.transpose(x1), x1))  # Core of the hat matrix
        h1_work = np.matmul(np.matmul(x1, h_core), np.transpose(x1))  # Training hat matrix
        h2_work = np.matmul(np.matmul(x2, h_core), np.transpose(x2))  # Testing hat matrix
        h1 = np.diag(h1_work)
        h2 = np.diag(h2_work)
        return h1, h2

    def _compute_residuals(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the standardized residuals.

        Returns:
            tuple[np.ndarray, np.ndarray]: Training and testing residuals.
        """
        res1 = self._data.y1_hat - self._data.y1
        res2 = self._data.y2_hat - self._data.y2
        s1 = np.std(res1)  # Standard deviation of training residuals
        resid1 = np.divide(res1, s1)
        resid2 = np.divide(res2, s1)
        return resid1, resid2

    def _compute_critical_hat_value(self) -> float:
        """
        Compute the critical hat value.

        Returns:
            float: Critical leverage value.
        """
        k = np.size(self._data.x1, axis=1) + 1  # Number of parameters + intercept
        return 3 * (k / len(self._data.x1))

    def plot_williams(self):
        """
        Constructs a Williams plot with leverages and standardized residuals.
        """
        h1, h2 = self.hat_matrix
        r1, r2 = self.residuals
        hat_star = self.critical_hat_value

        # Warning limit for standardized residuals
        sigma = 3
        plt.figure(figsize=(8, 6))
        plt.scatter(h1, r1, c='C0', label='Training set')
        plt.scatter(h2, r2, c='C1', label='Testing set')
        plt.axhline(y=sigma, xmin=0, xmax=1, color='red', linestyle='dashed', label=f"±{sigma} Residual Limit")
        plt.axhline(y=-sigma, xmin=0, xmax=1, color='red', linestyle='dashed')
        plt.axvline(x=hat_star, ymin=-sigma, ymax=sigma, color='green', linestyle='dashed', label=f"Critical Hat Value: {hat_star:.3f}")
        plt.xlabel("Leverage")
        plt.ylabel("Standardized Residuals")
        plt.title("Williams Plot")
        plt.legend()
        plt.show()
