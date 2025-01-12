from abc import (
    ABC,
    abstractmethod
)
from typing import (
    Any,
    Optional
)
from threading import Lock

from qsrr_ic.config import HyperParameterConfig
from qsrr_ic.enums import RegressorType
from qsrr_ic.models.qsrr.domain_models import QsrrData
from qsrr_ic.models.iso2grad import Iso2Grad
from qsrr_ic.models.iso2grad.domain_models import (
    Iso2GradSettings,
    Iso2GradData
)
from qsrr_ic.models.qsrr import QsrrModel
from qsrr_ic.optimization.domain_models import OptimizerSettings
from qsrr_ic.optimization import QsrrModelOptimizer


class ModelRunner(ABC):

    _instance = None
    _singleton_lock = Lock()

    def __new__(cls):
        """
        Ensure only one instance of the ModelRunner exists.
        """
        if not cls._instance:
            with cls._singleton_lock:
                if not cls._instance:  # Double-checked locking
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initialize the runner with an execution lock.
        """
        if not hasattr(self, "_execution_lock"):
            self._execution_lock = Lock()

    def run(self, *args, **kwargs) -> Any:
        """
        Run a given model. Only one model runs at a time.

        Args:
            *args: Positional arguments to pass to the model's `run` method.
            **kwargs: Keyword arguments to pass to the model's `run` method.

        Returns:
            Any: The result of the model's `run` method.

        Raises:
            AttributeError: If the provided model does not have a `run` method.
        """

        # Ensure exclusive access during execution
        with self._execution_lock:
            return self._run(*args, **kwargs)

    @abstractmethod
    def _run(self, *args, **kwargs) -> Any:
        pass


class QsrrOptimizerRunner(ModelRunner):
    def _run(
        self,
        optimizer_settings: OptimizerSettings,
        qsrr_train_data: QsrrData,
        qsrr_test_data: Optional[QsrrData]
    ):
        optimizer = QsrrModelOptimizer(
            optimizer_settings,
            qsrr_train_data=qsrr_train_data,
            qsrr_test_data=qsrr_test_data
        )
        results = optimizer.optimize()
        return results, optimizer


class QsrrModelRunner(ModelRunner):
    def _run(
        self,
        regressor_type: RegressorType,
        config: HyperParameterConfig,
        qsrr_train_data: QsrrData,
        qsrr_test_data: Optional[QsrrData] = None
    ):
        model = QsrrModel(
            regressor_type=config.regressor_type,
            qsrr_train_data=qsrr_train_data,
            qsrr_test_data=qsrr_test_data,
            hyper_parameters=config.hyper_parameter_registry
        )
        model.fit()
        return model


class QsrrIcModelRunner(ModelRunner):
    def _run(
        self,
        qsrr_model: Any,
        scaler: Any,
        iso2grad_data: Iso2GradData,
        iso2grad_settings: Iso2GradSettings
    ):
        model = Iso2Grad(
            qsrr_model,
            scaler,
            iso2grad_data,
            iso2grad_settings
        )
        model.fit()
        return model
