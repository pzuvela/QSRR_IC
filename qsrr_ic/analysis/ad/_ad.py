from typing import (
    Optional,
    Tuple
)

from dataclasses import dataclass

import numpy as np
from numpy import ndarray

import pandas as pd

import matplotlib.pyplot as plt

from qsrr_ic.error_handling import ErrorHandling
from qsrr_ic.models.qsrr.domain_models import QsrrData


@dataclass
class ApplicabilityDomainData:
    """
    Data class for encapsulating input data and related parameters for WilliamsModel.
    """

    data_train: QsrrData  # Training data (descriptors, retention)
    data_test: Optional[QsrrData]  # Test data (descriptors, retention), optional
    predictions_train: QsrrData  # Training predictions
    predictions_train: Optional[QsrrData]  # Test predictions


class ApplicabilityDomain:
    """
    Model responsible for computing residuals, hat matrices, and critical values with lazy loading.
    """

    def __init__(self, data: ApplicabilityDomainData):
        """
        Initialize the WilliamsModel with a WilliamsData object.

        Args:
            data (ApplicabilityDomainData): The input data and parameters for the model.
        """
        self.__data: ApplicabilityDomainData = data
        self.__hat_matrix_train: Optional[ndarray] = None
        self.__hat_matrix_test: Optional[ndarray] = None
        self.__residuals_train: Optional[ndarray] = None
        self.__residuals_test: Optional[ndarray] = None
        self.__critical_hat = None

    @property
    def hat_matrix_train(self) -> ndarray:
        if self.__hat_matrix_train is None:
            self.__hat_matrix_train = self._compute_hat_matrix()
        return self.__hat_matrix_train

    @hat_matrix_train.setter
    def hat_matrix_train(self, value):
        raise ErrorHandling.get_property_value_error("hat_matrix_train")

    @property
    def hat_matrix_test(self) -> Optional[ndarray]:
        if self.__hat_matrix_test is None and self.__data.data_test is not None:
            self.__hat_matrix_train = self._compute_hat_matrix()
        return self.__hat_matrix_train

    @hat_matrix_train.setter
    def hat_matrix_train(self, value):
        raise ErrorHandling.get_property_value_error("hat_matrix_train")

    @property
    def residuals(self) -> Tuple[ndarray, Optional[ndarray]]:
        if self.__residuals is None:
            self.__residuals = self._compute_residuals()
        return self.__residuals

    @residuals.setter
    def residuals(self, value):
        raise ErrorHandling.get_property_value_error("residuals")

    @property
    def critical_hat(self) -> float:
        if self.__critical_hat is None:
            self.__critical_hat = self._compute_critical_hat_value()
        return self.__critical_hat

    @critical_hat.setter
    def critical_hat(self, value):
        raise ErrorHandling.get_property_value_error("critical_hat")

    def _compute_hat_matrix(self) -> Tuple[ndarray, Optional[ndarray]]:
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
