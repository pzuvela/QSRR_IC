from typing import (
    Optional,
    Tuple
)

from qsrr_ic.domain_models import HyperParameterRegistry
from qsrr_ic.models.qsrr.enums import RegressorType
from qsrr_ic.optimization.enums import CrossValidationType


class CrossValidationSettings:
    def __init__(
        self,
        cv_type: CrossValidationType = CrossValidationType.KFold,
        n_splits: Optional[int] = 3
    ):
        self.cv_type = cv_type

        if self.cv_type == CrossValidationType.KFold:
            if n_splits is None:
                raise ValueError("n_splits cannot be None for CV type KFold!")
            if n_splits < 0:
                raise ValueError("n_splits cannot be negative!")

        self.n_splits = n_splits


class GlobalSearchSettings:
    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: Tuple[float] = (1.5, 1.9),
        n_jobs: int = -1
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.n_jobs = n_jobs


class OptimizerSettings:
    def __init__(
        self,
        regressor_type: RegressorType,
        hyper_parameter_ranges: HyperParameterRegistry,
        cv_settings: Optional[CrossValidationSettings] = None,
        global_search_settings: Optional[GlobalSearchSettings] = None
    ):
        self.regressor_type = regressor_type
        self.hyper_parameter_ranges = hyper_parameter_ranges

        if cv_settings is None:
            cv_settings = CrossValidationSettings()
        self.cv_settings = cv_settings

        self.global_search_settings = global_search_settings
