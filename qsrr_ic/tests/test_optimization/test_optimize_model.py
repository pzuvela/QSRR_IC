from pickletools import optimize

import pytest
import numpy as np
from scipy.optimize import Bounds

from qsrr_ic.config import DEFAULT_HYPER_PARAMETER_RANGES
from qsrr_ic.enums import (
    CrossValidationType,
    RegressorType
)
from qsrr_ic.domain_models import (
    CrossValidationSettings,
    GlobalSearchSettings,
    HyperParameter,
    HyperParameterRange,
    HyperParameterRegistry
)
from qsrr_ic.enums import HyperParameterName
from qsrr_ic.optimization.domain_models import OptimizerSettings

from qsrr_ic.optimization.optimize_model import QsrrModelOptimizer
from qsrr_ic.models.qsrr.domain_models import QsrrData


def get_qsrr_data():
    np.random.seed(42)  # For reproducibility
    x = np.random.rand(100, 100)  # 100 samples, 100 features
    y = np.random.rand(100)  # 100 target values
    return QsrrData(x=x, y=y)

def get_optimizer_settings(regressor_type: RegressorType) -> OptimizerSettings:
    return OptimizerSettings(
        regressor_type=regressor_type,
        cv_settings=CrossValidationSettings(
            CrossValidationType.KFold,
            n_splits=3
        ),
        hyper_parameter_ranges=DEFAULT_HYPER_PARAMETER_RANGES[regressor_type],
        global_search_settings=GlobalSearchSettings()
    )

def get_optimizer(regressor_type: RegressorType) -> QsrrModelOptimizer:

    return QsrrModelOptimizer(
        optimizer_settings=get_optimizer_settings(regressor_type),
        qsrr_train_data=get_qsrr_data()
    )


class TestQsrrModelOptimizer:

    """Test suite for QsrrModelOptimizer class."""
    @pytest.mark.parametrize(
        "regressor_type", list(RegressorType)
    )
    def test_initialization(self, regressor_type: RegressorType):
        """Test proper initialization of QsrrModelOptimizer."""
        optimizer = get_optimizer(regressor_type)
        assert optimizer.is_optimized is False
        assert optimizer.optimal_hyper_parameters is None
        assert optimizer.optimal_qsrr_model is None
        assert isinstance(optimizer.cv, optimizer.optimizer_settings.cv_settings.cv_type.value)

    @pytest.mark.parametrize("invalid_input,expected_error", [
        (("invalid", "mock_qsrr_data"), "optimizer_settings must be an instance of OptimizerSettings"),
        (("mock_optimizer_settings", "invalid"), "qsrr_train_data must be an instance of QsrrData")
    ])
    def test_initialization_invalid_inputs(self, invalid_input, expected_error):
        """Test initialization with invalid inputs."""
        input_map = {
            "mock_optimizer_settings": get_optimizer(RegressorType.PLS).optimizer_settings,
            "mock_qsrr_data": get_qsrr_data(),
            "invalid": "invalid"
        }

        with pytest.raises(ValueError, match=expected_error):
            QsrrModelOptimizer(
                optimizer_settings=input_map[invalid_input[0]],
                qsrr_train_data=input_map[invalid_input[1]]
            )

    @pytest.mark.parametrize(
        "regressor_type", list(RegressorType)
    )
    def test_get_bounds(self, regressor_type: RegressorType):
        """Test bounds generation."""
        optimizer = get_optimizer(regressor_type)
        bounds = optimizer.get_bounds()
        assert isinstance(bounds, Bounds)
        assert len(bounds.lb) == len(optimizer.optimizer_settings.hyper_parameter_ranges)
        assert len(bounds.ub) == len(optimizer.optimizer_settings.hyper_parameter_ranges)

    def test_get_hyper_parameters_registry(self):

        """Test conversion of array to hyperparameter registry."""

        test_params = np.array([2.0])

        optimizer = get_optimizer(RegressorType.PLS)

        registry = optimizer.get_hyper_parameters_registry(test_params)

        assert isinstance(registry, HyperParameterRegistry)
        assert registry.get(HyperParameterName.N_COMPONENTS).value == 2

    def test_optimize_using_knee_invalid_params(self):

        """Test knee optimization with invalid parameters."""
        registry = HyperParameterRegistry()
        registry.add(
            HyperParameterName.N_COMPONENTS,
            HyperParameterRange(HyperParameter(1), HyperParameter(5))
        )
        registry.add(
            HyperParameterName.N_ESTIMATORS,
            HyperParameterRange(HyperParameter(1), HyperParameter(5))
        )

        settings = OptimizerSettings(
            regressor_type=RegressorType.PLS,
            cv_settings=CrossValidationSettings(
                CrossValidationType.KFold,
                n_splits=3
            ),
            hyper_parameter_ranges=registry,
            global_search_settings=None
        )

        optimizer = QsrrModelOptimizer(
            optimizer_settings=settings,
            qsrr_train_data=get_qsrr_data()
        )

        with pytest.raises(ValueError, match="Only regressors with a single hyper-parameter"):
            optimizer._optimize_using_knee()

    def test_cross_validate_model(self):

        """Test cross-validation scoring."""

        optimizer = get_optimizer(RegressorType.PLS)
        hyper_parameter_registry = HyperParameterRegistry()
        hyper_parameter_registry.add(HyperParameterName.N_COMPONENTS, HyperParameter(2))
        model = optimizer.get_model(hyper_parameter_registry)
        score = optimizer.cross_validate_model(model)

        assert isinstance(score, float)
        assert -np.inf <= score <= np.inf

    @pytest.mark.parametrize(
        "regressor_type", list(RegressorType)
    )
    def test_full_optimization(self, regressor_type: RegressorType):

        """Test complete optimization process."""
        optimizer = get_optimizer(regressor_type)


        results = optimizer.optimize()

        assert optimizer.is_optimized
        assert optimizer.optimal_hyper_parameters is not None
        assert optimizer.optimal_qsrr_model is not None
        assert results.regressor_type == regressor_type
        assert isinstance(results.optimal_hyper_parameters, HyperParameterRegistry)
