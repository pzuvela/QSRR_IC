from typing import (
    Any,
    Dict,
    Optional
)

import numpy as np
from numpy import ndarray

from sklearn.metrics import r2_score

from qsrr_ic.models.qsrr.domain_models import (
    QsrrData,
    QsrrMetrics,
    QsrrResults
)
from qsrr_ic.models.qsrr.enums import (
    RegressorType,
    REGRESSOR_MAPPING
)


class QsrrModel:

    def __init__(
        self,
        regressor_type: RegressorType,
        qsrr_train_data: QsrrData,
        qsrr_test_data: Optional[QsrrData] = None,
        hyper_parameters: Optional[Dict[str, Any]] = None
    ):
        self.regressor_type = regressor_type
        self.qsrr_train_data = qsrr_train_data
        self.qsrr_test_data = qsrr_test_data
        self.hyper_parameters = hyper_parameters

        self.is_fitted = False
        self.model: Optional[Any] = None

        self.__train_results: Optional[QsrrResults] = None
        self.__test_results: Optional[QsrrResults] = None
        self.__pls_r2: Optional[ndarray] = None

        self.__validate_inputs()
        self.__instantiate_model()

    def __validate_inputs(self) -> None:
        if not isinstance(self.qsrr_train_data, QsrrData):
            raise TypeError("qsrr_train_data must be an instance of QsrrData.")
        if self.qsrr_test_data and not isinstance(self.qsrr_test_data, QsrrData):
            raise TypeError("qsrr_test_data must be an instance of QsrrData or None.")
        if self.qsrr_train_data.x.shape[0] != self.qsrr_train_data.y.shape[0]:
            raise ValueError("The number of rows in qsrr_train_data.x and qsrr_train_data.y must match.")
        if self.qsrr_test_data:
            if self.qsrr_test_data.x.shape[1] != self.qsrr_train_data.x.shape[1]:
                raise ValueError("The number of columns in qsrr_test_data.x must match qsrr_train_data.x.")

    def __instantiate_model(self) -> None:
        model_kwargs = {}
        if self.regressor_type == RegressorType.xGB:
            model_kwargs["objective"] = "reg:squarederror"
        elif self.regressor_type == RegressorType.ADA:
            model_kwargs["loss"] = "exponential"
        self.model = REGRESSOR_MAPPING[self.regressor_type](**model_kwargs)
        if self.hyper_parameters is not None:
            self.model.set_params(**self.hyper_parameters)

    def fit(self) -> None:
        """
        Train the QSRR model using the provided training data.
        """
        try:
            self.model.fit(self.qsrr_train_data.x, self.qsrr_train_data.y)
        except Exception as e:
            raise RuntimeError(f"An error occurred while fitting the model: {e}")
        self.is_fitted = True
        self.__train_results = None
        self.__test_results = None  # Reset results after refitting

    @property
    def train_results(self) -> QsrrResults:
        """
        Get or compute the training results.
        """
        if self.__train_results is None and self.is_fitted:
            self.__train_results = self._calculate_results(self.qsrr_train_data)
        return self.__train_results

    @train_results.setter
    def train_results(self, value: QsrrResults) -> None:
        raise ValueError("Train Results cannot be set!")

    @property
    def test_results(self) -> Optional[QsrrResults]:
        """
        Get or compute the test results if test data is available.
        """
        if self.__test_results is None and self.qsrr_test_data and self.is_fitted:
            self.__test_results = self._calculate_results(self.qsrr_test_data)
        return self.__test_results

    @test_results.setter
    def test_results(self, value: QsrrResults) -> None:
        raise ValueError("Test Results cannot be set!")

    @property
    def pls_r2(self) -> Optional[ndarray]:
        """
        Get PLS R2
        """
        if self.__pls_r2 is None and self.is_fitted and self.regressor_type == RegressorType.PLS:
            self.__pls_r2 = self._calculate_percentage_variance()
        return self.__pls_r2

    @pls_r2.setter
    def pls_r2(self, value: ndarray) -> None:
        raise ValueError("PLS R2 cannot be set!")

    def _calculate_results(self, data: QsrrData) -> QsrrResults:
        predictions = self.model.predict(data.x).ravel()
        qsrr_predictions = QsrrData(y=predictions)
        qsrr_metrics = QsrrMetrics(qsrr_data=data, qsrr_predictions=qsrr_predictions)
        return QsrrResults(qsrr_predictions=qsrr_predictions, qsrr_metrics=qsrr_metrics)

    def to_pickle(self, file_path: str) -> None:
        """
        Save the trained model to a file.
        """
        import joblib
        if not self.is_fitted:
            raise RuntimeError("The model must be fitted before it can be saved.")
        joblib.dump(self.model, file_path)

    def _calculate_percentage_variance(self) -> ndarray:
        """
        Calculate the percentage of variance explained for PLS models.
        """

        if not hasattr(self.model, "x_scores_"):
            raise AttributeError("PLS model does not have 'x_scores_' attribute.")

        total_x_variance = np.var(self.qsrr_train_data.x, axis=0, ddof=1).sum()
        explained_x_variance = np.var(self.model.x_scores_, axis=0, ddof=1).sum()

        if total_x_variance == 0:
            raise ValueError("Total variance in training data is zero. Cannot compute explained variance.")

        r2_x = explained_x_variance / total_x_variance
        r2_y = r2_score(self.qsrr_train_data.y, self.train_results.qsrr_predictions.y)

        return np.array([r2_x, r2_y])
