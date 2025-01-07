import pytest
import json
from qsrr_ic.domain_models import (
    HyperParameter,
    HyperParameterRange,
    HyperParameterRegistry
)
from qsrr_ic.enums import HyperParameterName
from qsrr_ic.models.qsrr.enums import RegressorType
from qsrr_ic.config import (
    Config,
    DEFAULT_HYPER_PARAMETER_RANGES
)

default_hyper_parameters = DEFAULT_HYPER_PARAMETER_RANGES


class TestConfig:

    def test_init_with_defaults(self):
        config = Config(regressor_type=RegressorType.PLS)
        assert config.regressor_type == RegressorType.PLS
        assert config.hyper_parameter_ranges == default_hyper_parameters[RegressorType.PLS]

    def test_init_with_custom_hyper_parameters(self):
        custom_hps = HyperParameterRegistry()
        custom_hps.add(
            HyperParameterName.N_COMPONENTS,
            HyperParameterRange(HyperParameter(2), HyperParameter(8))
        )
        config = Config(regressor_type=RegressorType.PLS, hyper_parameter_ranges=custom_hps)
        assert config.regressor_type == RegressorType.PLS
        assert config.hyper_parameter_ranges == custom_hps

    def test_from_dict_with_defaults(self):
        config_dict = {
            "regressor_type": "PLS"
        }
        config = Config.from_dict(config_dict)
        assert config.regressor_type == RegressorType.PLS
        assert config.hyper_parameter_ranges == default_hyper_parameters[RegressorType.PLS]

    def test_from_dict_with_custom_hyper_parameters(self):
        config_dict = {
            "regressor_type": "PLS",
            "hyper_parameter_ranges": {
                "n_components": [2, 8]
            }
        }
        config = Config.from_dict(config_dict)
        assert config.regressor_type == RegressorType.PLS
        assert config.hyper_parameter_ranges.get(HyperParameterName.N_COMPONENTS).lower.value == 2
        assert config.hyper_parameter_ranges.get(HyperParameterName.N_COMPONENTS).upper.value == 8

    def test_from_dict_invalid_regressor_type(self):
        config_dict = {
            "regressor_type": "INVALID"
        }
        with pytest.raises(KeyError, match="Invalid regressor type"):
            Config.from_dict(config_dict)

    def test_to_dict(self):
        config = Config(regressor_type=RegressorType.PLS)
        config_dict = config.to_dict()
        assert config_dict["regressor_type"] == "PLS"
        assert "hyper_parameter_ranges" in config_dict

    def test_to_json(self, tmp_path):
        config = Config(regressor_type=RegressorType.PLS)
        json_path = tmp_path / "config.json"
        config.to_json(str(json_path))
        with open(json_path, "r") as f:
            loaded_dict = json.load(f)
        assert loaded_dict["regressor_type"] == "PLS"

    def test_from_json(self, tmp_path):
        config_dict = {
            "regressor_type": "PLS",
            "hyper_parameter_ranges": {
                "n_components":  [2, 8]
            }
        }
        json_path = tmp_path / "config.json"
        with open(json_path, "w") as f:
            json.dump(config_dict, f)

        config = Config.from_json(str(json_path))
        assert config.regressor_type == RegressorType.PLS
        assert config.hyper_parameter_ranges.get(HyperParameterName.N_COMPONENTS).lower.value == 2
        assert config.hyper_parameter_ranges.get(HyperParameterName.N_COMPONENTS).upper.value == 8
