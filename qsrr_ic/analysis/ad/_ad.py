from typing import (
    Tuple,
    Optional
)

import numpy as np
from numpy import ndarray

from qsrr_ic.analysis.ad.domain_models import ApplicabilityDomainData
from qsrr_ic.error_handling import ErrorHandling



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

        self.__h_core: Optional[ndarray] = None

        self.__hat_matrix_train: Optional[ndarray] = None
        self.__hat_matrix_test: Optional[ndarray] = None

        self.__residuals_train: Optional[ndarray] = None
        self.__residuals_test: Optional[ndarray] = None

        self.__residual_std: Optional[float] = None
        self.__std_residuals_train: Optional[ndarray] = None
        self.__std_residuals_test: Optional[ndarray] = None

        self.__critical_hat = None

    @property
    def hat_matrix_train(self) -> ndarray:
        if self.__hat_matrix_train is None:
            self.__hat_matrix_train = self._compute_hat_matrix(x=self.__data.data_train.x)
        return self.__hat_matrix_train

    @hat_matrix_train.setter
    def hat_matrix_train(self, value):
        raise ErrorHandling.get_property_value_error("hat_matrix_train")

    @property
    def hat_matrix_test(self) -> Optional[ndarray]:
        if self.__hat_matrix_test is None and self.__data.data_test is not None:
            self.__hat_matrix_test = self._compute_hat_matrix(x=self.__data.data_test.x)
        return self.__hat_matrix_test

    @hat_matrix_test.setter
    def hat_matrix_test(self, value):
        raise ErrorHandling.get_property_value_error("hat_matrix_test")

    @property
    def residuals_train(self) -> ndarray:
        if self.__residuals_train is None:
            self.__residuals_train = self._compute_residuals(
                self.__data.data_train.y,
                self.__data.predictions_train.y
            )
        return self.__residuals_train

    @residuals_train.setter
    def residuals_train(self, value):
        raise ErrorHandling.get_property_value_error("residuals_train")

    @property
    def residuals_test(self) -> Optional[ndarray]:
        if self.__residuals_test is None and self.__data.data_test is not None and self.__data.predictions_test is not None:
            self.__residuals_test = self._compute_residuals(
                self.__data.data_test.y,
                self.__data.predictions_test.y
            )
        return self.__residuals_test

    @residuals_test.setter
    def residuals_test(self, value):
        raise ErrorHandling.get_property_value_error("residuals_test")

    @property
    def critical_hat(self) -> float:
        if self.__critical_hat is None:
            self.__critical_hat = self._compute_critical_hat_value()
        return self.__critical_hat

    @critical_hat.setter
    def critical_hat(self, value):
        raise ErrorHandling.get_property_value_error("critical_hat")

    @property
    def _h_core(self):
        if self.__h_core is None:
            x_train = self.__data.data_train.x
            self.__h_core = np.linalg.pinv(np.matmul(x_train.transpose(), x_train))
        return self.__h_core

    @_h_core.setter
    def _h_core(self, value):
        raise ErrorHandling.get_property_value_error("_h core")

    @property
    def _residual_std(self):
        if self.__residual_std is None:
            self.__residual_std = np.std(self.residuals_train, ddof=1)
        return self.__residual_std

    @_residual_std.setter
    def _residual_std(self, value):
        raise ErrorHandling.get_property_value_error("residual std")

    def _compute_hat_matrix(self, x: ndarray) -> ndarray:
        """
        Compute the hat matrix

        Returns:
            ndarray: leverages
        """
        return np.diag(np.matmul(np.matmul(x, self._h_core), x.transpose()))

    @property
    def std_residuals_train(self) -> ndarray:
        if self.__std_residuals_train is None:
            self.__std_residuals_train = self._standardize_residuals(self.residuals_train)
        return self.__std_residuals_train

    @std_residuals_train.setter
    def std_residuals_train(self, value):
        raise ErrorHandling.get_property_value_error("std_residuals_train")

    @property
    def std_residuals_test(self) -> Optional[ndarray]:
        if self.__std_residuals_test is None and self.residuals_test is not None:
            self.__std_residuals_test = self._standardize_residuals(self.residuals_test)
        return self.__std_residuals_test

    @std_residuals_test.setter
    def std_residuals_test(self, value):
        raise ErrorHandling.get_property_value_error("std_residuals_test")

    @staticmethod
    def _compute_residuals(y: ndarray, y_hat: ndarray) -> ndarray:
        return y_hat - y

    def _standardize_residuals(self, residuals: ndarray) -> ndarray:
        """
        Compute the standardized residuals.

        Returns:
            ndarray
        """

        return np.divide(residuals, self._residual_std)

    def _compute_critical_hat_value(self) -> float:
        """
        Compute the critical hat value.

        Returns:
            float: Critical leverage value.
        """
        m, n = self.__data.data_train.x.shape
        k = m + 1  # Number of parameters + intercept
        return 3 * (k / n)


class ApplicabilityDomainPlot:

    def __init__(self, ad: ApplicabilityDomain):
        self._ad = ad

    def plot_williams(
        self,
        n_sigma: int = 3,
        fig_size: Tuple[int, int] = (8, 6)
    ):
        """
        Constructs a Williams plot with leverages and standardized residuals.
        """

        import matplotlib.pyplot as plt  # Lazy load

        # Warning limit for standardized residuals
        plt.figure(figsize=fig_size)
        plt.scatter(self._ad.hat_matrix_train, self._ad.residuals_train, c='C0', label='Training set')
        if self._ad.hat_matrix_test is not None and self._ad.residuals_test is not None:
            plt.scatter(self._ad.hat_matrix_test, self._ad.residuals_test, c='C1', label='Testing set')
        plt.axhline(y=n_sigma, xmin=0, xmax=1, color='red', linestyle='dashed', label=f"±{n_sigma} Residual Limit")
        plt.axhline(y=-n_sigma, xmin=0, xmax=1, color='red', linestyle='dashed')
        plt.axvline(
            x=self._ad.critical_hat,
            ymin=-n_sigma,
            ymax=n_sigma,
            color='green',
            linestyle='dashed',
            label=f"Critical Hat Value: {self._ad.critical_hat:.3f}"
        )
        plt.xlabel("Leverage")
        plt.ylabel("Standardized Residuals")
        plt.title("Williams Plot")
        plt.legend()
        plt.show()
