from typing import (
    Optional,
    Union
)

import numpy as np
from numpy import ndarray

from scipy.optimize import (
    Bounds,
    differential_evolution
)


from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    cross_val_score,
    KFold,
    LeaveOneOut
)

from qsrr_ic.domain_models import (
    HyperParameter,
    HyperParameterRegistry,
)
from qsrr_ic.enums import (
    CrossValidationType,
    RegressorType
)
from qsrr_ic.metrics import Metrics
from qsrr_ic.optimization.domain_models import (
    OptimizerSettings,
    OptimizerResults
)
from qsrr_ic.process.process_curve_data import ProcessCurveData
from qsrr_ic.models.qsrr import QsrrModel
from qsrr_ic.models.qsrr.domain_models import QsrrData


class QsrrModelOptimizer:
    """
    Optimizes a QSRR model based on provided settings and training data.

    Attributes:
        optimizer_settings: Configuration for the optimization process.
        qsrr_train_data: Training data for the QSRR model.
        qsrr_test_data: Optional test data for evaluation.
        optimal_hyper_parameters: Stores the best hyperparameters after optimization.
        optimal_qsrr_model: Stores the best QSRR model after optimization.
        cv: Cross-validation strategy (KFold or LeaveOneOut).
        is_optimized: Indicates whether the optimization process is completed.
    """

    def __init__(
        self,
        optimizer_settings: OptimizerSettings,
        qsrr_train_data: QsrrData,
        qsrr_test_data: Optional[QsrrData] = None,
    ):
        if not isinstance(optimizer_settings, OptimizerSettings):
            raise ValueError("optimizer_settings must be an instance of OptimizerSettings.")
        if not isinstance(qsrr_train_data, QsrrData):
            raise ValueError("qsrr_train_data must be an instance of QsrrData.")

        self.optimizer_settings = optimizer_settings
        self.qsrr_train_data = qsrr_train_data
        self.qsrr_test_data = qsrr_test_data

        self.optimal_hyper_parameters: Optional[HyperParameterRegistry] = None
        self.optimal_qsrr_model: Optional[QsrrModel] = None

        cv_kwargs = {}
        if self.optimizer_settings.cv_settings.cv_type == CrossValidationType.KFold:
            cv_kwargs = {"n_splits": self.optimizer_settings.cv_settings.n_splits}

        self.cv: Union[KFold, LeaveOneOut] = self.optimizer_settings.cv_settings.cv_type.value(**cv_kwargs)

        self.is_optimized: bool = False

    def get_bounds(self) -> Bounds:
        """
        Constructs the bounds for the optimization process.

        Returns:
            Bounds: A Bounds object for the optimizer.
        """
        bounds_lb = []
        bounds_ub = []
        for _, hp in self.optimizer_settings.hyper_parameter_ranges:
            bounds_lb.append(hp.lower.value)
            bounds_ub.append(hp.upper.value)
        return Bounds(bounds_lb, bounds_ub)

    def get_hyper_parameters_registry(self, hyper_parameters: ndarray) -> 'HyperParameterRegistry':
        """
        Converts a flat array of hyperparameters into a HyperParameterRegistry.

        Args:
            hyper_parameters (ndarray): Array of hyperparameter values.

        Returns:
            HyperParameterRegistry: Registry containing the hyperparameter configuration.
        """
        hyper_parameters_dict = {
            name.value: hyper_parameters[idx].item()
            for idx, name in enumerate(self.optimizer_settings.hyper_parameter_ranges.names())
        }
        return HyperParameterRegistry.from_dict(hyper_parameters_dict)

    def get_model(self, hyper_parameters: HyperParameterRegistry) -> QsrrModel:
        """
        Creates a QSRR model using the provided hyperparameters.

        Args:
            hyper_parameters (HyperParameterRegistry): Registry of hyperparameters.

        Returns:
            QsrrModel: A configured QSRR model instance.
        """
        return QsrrModel(
            regressor_type=self.optimizer_settings.regressor_type,
            qsrr_train_data=self.qsrr_train_data,
            qsrr_test_data=self.qsrr_test_data,
            hyper_parameters=hyper_parameters,
        )

    def cross_validate_model(self, qsrr_model: QsrrModel, greater_is_better: bool = True) -> float:
        """
        Performs cross-validation on the given model.

        Args:
            qsrr_model (QsrrModel): The QSRR model to evaluate.
            greater_is_better (bool): Whether a higher score indicates better performance.

        Returns:
            float: The mean cross-validation score.
        """
        score = cross_val_score(
            qsrr_model.model,
            qsrr_model.scaler.transform(self.qsrr_train_data.x),
            self.qsrr_train_data.y,
            cv=self.cv,
            scoring=make_scorer(Metrics.rmse, greater_is_better=greater_is_better),
        )
        return np.mean(score)

    def _objective_function(self, hyper_parameters: HyperParameterRegistry, greater_is_better: bool = True) -> float:
        """
        Calculates the objective function value for optimization.

        Args:
            hyper_parameters (HyperParameterRegistry): Registry of hyperparameters.
            greater_is_better (bool): Whether a higher score indicates better performance.

        Returns:
            float: The calculated objective function value.
        """
        qsrr_model = self.get_model(hyper_parameters)
        return self.cross_validate_model(qsrr_model, greater_is_better=greater_is_better)

    def objective_function(self, hyper_parameters: ndarray) -> float:
        """
        Wrapper for the objective function compatible with scipy.optimize.

        Args:
            hyper_parameters (ndarray): Array of hyperparameter values.

        Returns:
            float: The calculated objective function value.
        """
        hyper_parameters = self.get_hyper_parameters_registry(hyper_parameters)
        return self._objective_function(hyper_parameters)

    def _optimize_using_de(self):

        optimal_hyper_parameters = differential_evolution(
            self.objective_function,
            self.get_bounds(),
            workers=self.optimizer_settings.global_search_settings.n_jobs,
            updating="deferred",
            mutation=self.optimizer_settings.global_search_settings.mutation_rate,
            popsize=self.optimizer_settings.global_search_settings.population_size,
        )
        self.optimal_hyper_parameters = self.get_hyper_parameters_registry(optimal_hyper_parameters.x)

    def _optimize_using_knee(self):
        if len(self.optimizer_settings.hyper_parameter_ranges) > 1:
            raise ValueError("Only regressors with a single hyper-parameter can be optimized using the knee-point approach.")

        name = self.optimizer_settings.hyper_parameter_ranges.names()[0]
        hp_range = self.optimizer_settings.hyper_parameter_ranges.get(name)

        if hp_range.type() != int:
            raise ValueError("Knee-point optimization is only supported for integer hyperparameters.")

        x = []
        errors = []
        for value in range(hp_range.lower.value, hp_range.upper.value + 1):
            hyper_parameters = HyperParameterRegistry()
            hyper_parameters.add(name, HyperParameter(value))
            errors.append(self._objective_function(hyper_parameters, greater_is_better=False))
            x.append(value)

        optimal_hp = ProcessCurveData.knee(np.array(x), np.array(errors))
        self.optimal_hyper_parameters = HyperParameterRegistry()
        self.optimal_hyper_parameters.add(name, HyperParameter(optimal_hp))

    def optimize(self) -> OptimizerResults:
        """
        Runs the optimization process and returns the results.

        Returns:
            OptimizerResults: Contains the best hyperparameters and the optimized model.
        """
        if self.optimizer_settings.regressor_type == RegressorType.PLS:
            self._optimize_using_knee()
        else:
            self._optimize_using_de()

        self.optimal_qsrr_model = self.get_model(self.optimal_hyper_parameters)
        self.optimal_qsrr_model.fit()
        self.is_optimized = True

        return OptimizerResults(
            regressor_type=self.optimizer_settings.regressor_type,
            optimal_hyper_parameters=self.optimal_hyper_parameters,
            optimal_qsrr_model=self.optimal_qsrr_model,
        )
