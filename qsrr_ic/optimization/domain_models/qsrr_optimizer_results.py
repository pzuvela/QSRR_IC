from qsrr_ic.models.qsrr.enums import RegressorType
from qsrr_ic.optimization.domain_models import HyperParameterRegistry


class OptimizerResults:
    def __init__(
        self,
        regressor_type: RegressorType,
        optimal_hyper_parameters: HyperParameterRegistry
    ):
        self.regressor_type = regressor_type
        self.optimal_hyper_parameters = optimal_hyper_parameters
