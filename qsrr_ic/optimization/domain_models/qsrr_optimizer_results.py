from qsrr_ic.models.qsrr import QsrrModel
from qsrr_ic.models.qsrr.enums import RegressorType
from qsrr_ic.optimization.domain_models import HyperParameterRegistry


class OptimizerResults:
    def __init__(
        self,
        regressor_type: RegressorType,
        optimal_hyper_parameters: HyperParameterRegistry,
        optimal_qsrr_model: QsrrModel
    ):
        if not isinstance(regressor_type, RegressorType):
            raise ValueError("regressor_type must be an instance of RegressorType.")
        if not isinstance(optimal_hyper_parameters, HyperParameterRegistry):
            raise ValueError("optimal_hyper_parameters must be an instance of HyperParameterRegistry.")
        if not isinstance(optimal_qsrr_model, QsrrModel):
            raise ValueError("optimal_qsrr_model must be an instance of QsrrModel.")

        self._regressor_type = regressor_type
        self._optimal_hyper_parameters = optimal_hyper_parameters
        self._optimal_qsrr_model = optimal_qsrr_model

    @property
    def regressor_type(self) -> RegressorType:
        return self._regressor_type

    @property
    def optimal_hyper_parameters(self) -> HyperParameterRegistry:
        return self._optimal_hyper_parameters

    @property
    def optimal_qsrr_model(self) -> QsrrModel:
        return self._optimal_qsrr_model

    def __repr__(self) -> str:
        return (
            f"OptimizerResults("
            f"regressor_type={self.regressor_type}, "
            f"optimal_hyper_parameters={self.optimal_hyper_parameters}, "
            f"optimal_qsrr_model={self.optimal_qsrr_model})"
        )
