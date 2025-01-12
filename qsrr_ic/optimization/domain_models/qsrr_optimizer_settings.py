from typing import Optional

from qsrr_ic.domain_models import (
    CrossValidationSettings,
    HyperParameterRegistry,
    GlobalSearchSettings
)
from qsrr_ic.enums import RegressorType




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
