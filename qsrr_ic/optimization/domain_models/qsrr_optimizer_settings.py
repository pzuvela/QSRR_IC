from typing import Optional

from qsrr_ic.models.qsrr.enums import RegressorType
from qsrr_ic.optimization.domain_models import HyperParameterRegistry
from qsrr_ic.optimization.enums import CrossValidationType


class CrossValidationSettings:
    def __init__(
        self,
        cv_type: CrossValidationType,
        n_splits: Optional[int] = None
    ):
        self.cv_type = cv_type

        if self.cv_type == CrossValidationType.KFold:
            if n_splits is None:
                raise ValueError("n_splits cannot be None for CV type KFold!")
            if n_splits < 0:
                raise ValueError("n_splits cannot be negative!")

        self.n_splits = n_splits


class OptimizerSettings:
    def __init__(
        self,
        regressor_type: RegressorType,
        hyper_parameter_ranges: HyperParameterRegistry,
        cv_settings: CrossValidationSettings,
        n_jobs: int = -1
    ):
        self.regressor_type = regressor_type
        self.hyper_parameter_ranges = hyper_parameter_ranges
        self.cv_settings = cv_settings
        self.n_jobs = n_jobs
