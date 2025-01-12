from typing import (
    Optional,
    Tuple
)

from numpy import ndarray

from sklearn.model_selection import train_test_split

from qsrr_ic.config.qsrr_ic_config import TrainTestSplitConfig


class QsrrData:

    def __init__(
        self,
        y: ndarray,
        x: Optional[ndarray] = None,
    ):
        """

        Data class holding the data required to fit a QSRR model or the predictions

        Parameters
        ----------
        y: ndarray
            Vector of isocratic retention times / retention factors, [n_analytes, 1] (training set)
        x: Optional[ndarray]
            Matrix of model predictors, [n_analytes, n_descriptors] (training set)

        """

        self._validate_inputs(x, y)

        self.x = x
        self.y = y

    @staticmethod
    def _validate_inputs(x, y):
        if not isinstance(y, ndarray):
            raise TypeError("y must be a numpy array.")
        if y.ndim != 1 and y.ndim != 2:
            raise ValueError("y must be a 1D or 2D array.")
        if x is not None:
            if not isinstance(x, ndarray):
                raise TypeError("x must be a numpy array.")
            if x.ndim != 2:
                raise ValueError("x must be a 2D array.")
            if x.shape[0] != y.shape[0]:
                raise ValueError("The number of rows in x and y must be the same.")

    def split(self, config: TrainTestSplitConfig) -> Tuple['QsrrData', 'QsrrData']:
        x_train, x_test, y_train, y_test = train_test_split(
            self.x,
            self.y,
            test_size=config.test_ratio,
            random_state=config.random_seed,
            shuffle=config.shuffle
        )
        return QsrrData(y=y_train, x=x_train), QsrrData(y=y_test, x=x_test)
