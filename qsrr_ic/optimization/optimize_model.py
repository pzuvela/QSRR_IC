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

from qsrr_ic.metrics import Metrics
from qsrr_ic.models.qsrr import QsrrModel
from qsrr_ic.models.qsrr.domain_models import QsrrData
from qsrr_ic.models.qsrr.enums import RegressorType
from qsrr_ic.optimization.domain_models import (
    HyperParameter,
    HyperParameterRange,
    HyperParameterRegistry,
    OptimizerSettings,
    OptimizerResults,
)
from qsrr_ic.process.process_curve_data import ProcessCurveData
from qsrr_ic.optimization.enums import (
    CrossValidationType,
    HyperParameterName
)


class QsrrModelOptimizer:

    def __init__(
        self,
        optimizer_settings: OptimizerSettings,
        qsrr_train_data: QsrrData,
        qsrr_test_data: Optional[QsrrData] = None
    ):

        self.optimizer_settings = optimizer_settings
        self.qsrr_train_data = qsrr_train_data
        self.qsrr_test_data = qsrr_test_data

        self.optimal_hyper_parameters: Optional[HyperParameterRegistry] = None
        self.optimal_qsrr_model: Optional[QsrrModel] = None

        cv_kwargs = {}

        if self.optimizer_settings.cv_settings.cv_type == CrossValidationType.KFold:
            cv_kwargs = {"n_splits": self.optimizer_settings.cv_settings.n_splits}

        self.cv: Union[KFold, LeaveOneOut] = self.optimizer_settings.cv_settings.cv_type.value(**cv_kwargs)

        self.is_optimized: bool = True

    def get_bounds(self) -> Bounds:

        bounds_lb = []
        bounds_ub = []

        for _, hp in self.optimizer_settings.hyper_parameter_ranges:
            bounds_lb.append(hp.lower)
            bounds_ub.append(hp.upper)

        return Bounds(bounds_lb, bounds_ub)

    def get_hyper_parameters_registry(self, hyper_parameters: ndarray[float]) -> HyperParameterRegistry:
        hyper_parameters_dict = {
            name: hyper_parameters[idx].item()
            for idx, name in enumerate(self.optimizer_settings.hyper_parameter_ranges.names())
        }
        return HyperParameterRegistry.from_dict(hyper_parameters_dict)

    def get_model(self, hyper_parameters: HyperParameterRegistry) -> QsrrModel:
        return QsrrModel(
            regressor_type=self.optimizer_settings.regressor_type,
            qsrr_train_data=self.qsrr_train_data,
            qsrr_test_data=self.qsrr_test_data,
            hyper_parameters=hyper_parameters
        )

    def cross_validate_model(self, qsrr_model: QsrrModel, greater_is_better: bool = True):
        score = cross_val_score(
            qsrr_model.model,
            self.qsrr_train_data.x,
            self.qsrr_train_data.y,
            cv=self.cv,
            scoring=make_scorer(Metrics.rmse, greater_is_better=greater_is_better)
        )
        return np.mean(score)

    def _objective_function(self, hyper_parameters: HyperParameterRegistry, greater_is_better: bool = True):
        qsrr_model = self.get_model(hyper_parameters)
        cv_score = self.cross_validate_model(qsrr_model, greater_is_better=greater_is_better)
        return cv_score

    def objective_function(self, hyper_parameters: ndarray[float]):
        hyper_parameters = self.get_hyper_parameters_registry(hyper_parameters)
        return self._objective_function(hyper_parameters)

    def _optimize_using_de(self):
        optimal_hyper_parameters = differential_evolution(
            self.objective_function,
            self.get_bounds(),
            workers=self.optimizer_settings.global_search_settings.n_jobs,
            updating='deferred',
            mutation=self.optimizer_settings.global_search_settings.mutation_rate,
            popsize=self.optimizer_settings.global_search_settings.population_size
        )
        self.optimal_hyper_parameters = self.get_hyper_parameters_registry(optimal_hyper_parameters.x)

    def _optimize_using_knee(self):

        if len(self.optimizer_settings.hyper_parameter_ranges) > 1:
            raise ValueError(
                "Only regressors with a single hyper-parameter can be optimized using knee point approach!"
            )

        name: HyperParameterName = self.optimizer_settings.hyper_parameter_ranges.names()[0]  # Only one parameter
        hp_range: HyperParameterRange = self.optimizer_settings.hyper_parameter_ranges.get(name)

        if hp_range.type() != int:
            raise ValueError("Knee point approach is currently only supported for int HPs!")

        min_: int = hp_range.lower.value
        max_: int = hp_range.upper.value

        x = []
        errors = []

        for value in range(min_, max_ + 1):
            hyper_parameters = HyperParameterRegistry()
            hyper_parameters.add(name, HyperParameter(value))
            cv_score = self._objective_function(hyper_parameters, greater_is_better=False)
            x.append(value)
            errors.append(cv_score)

        optimal_hp = ProcessCurveData.knee(np.array(x), np.array(errors))

        self.optimal_hyper_parameters = HyperParameterRegistry()
        self.optimal_hyper_parameters.add(name, HyperParameter(optimal_hp))

    def optimize(self):

        # Run Optimization
        if self.optimizer_settings.regressor_type == RegressorType.PLS:
            self._optimize_using_knee()
        else:
            self._optimize_using_de()

        # Fit Final model
        self.optimal_qsrr_model = self.get_model(self.optimal_hyper_parameters)
        self.optimal_qsrr_model.fit()

        self.is_optimized = True
