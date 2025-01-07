import json
from typing import (
    Any,
    Dict,
    Optional
)

from qsrr_ic.domain_models import (
    HyperParameter,
    HyperParameterRange,
    HyperParameterRegistry
)
from qsrr_ic.enums import HyperParameterName
from qsrr_ic.models.qsrr.enums import RegressorType

default_pls_hps = HyperParameterRegistry()
default_pls_hps.add(HyperParameterName.N_COMPONENTS, HyperParameterRange(HyperParameter(1), HyperParameter(10)))

default_boosting_hps = HyperParameterRegistry()
default_boosting_hps.add(HyperParameterName.N_ESTIMATORS, HyperParameterRange(HyperParameter(10), HyperParameter(500)))
default_boosting_hps.add(HyperParameterName.LEARNING_RATE, HyperParameterRange(HyperParameter(0.1), HyperParameter(0.9)))
default_boosting_hps.add(HyperParameterName.MAX_DEPTH, HyperParameterRange(HyperParameter(1), HyperParameter(5)))

default_ada_hps = HyperParameterRegistry()
default_ada_hps.add(HyperParameterName.N_ESTIMATORS, HyperParameterRange(HyperParameter(300), HyperParameter(1000)))
default_ada_hps.add(HyperParameterName.LEARNING_RATE, HyperParameterRange(HyperParameter(0.1), HyperParameter(0.9)))

default_rfr_hps = HyperParameterRegistry()
default_rfr_hps.add(HyperParameterName.N_ESTIMATORS, HyperParameterRange(HyperParameter(10), HyperParameter(500)))
default_rfr_hps.add(HyperParameterName.MAX_DEPTH, HyperParameterRange(HyperParameter(8), HyperParameter(20)))
default_rfr_hps.add(HyperParameterName.MIN_SAMPLES_LEAF, HyperParameterRange(HyperParameter(1), HyperParameter(4)))

DEFAULT_HYPER_PARAMETER_RANGES: Dict[RegressorType, HyperParameterRegistry] = {
    RegressorType.PLS: default_pls_hps,
    RegressorType.xGB: default_boosting_hps,
    RegressorType.GBR: default_boosting_hps,
    RegressorType.ADA: default_ada_hps,
    RegressorType.RFR: default_rfr_hps,
}


class Config:
    def __init__(
        self,
        regressor_type: RegressorType,
        hyper_parameter_ranges: Optional[HyperParameterRegistry] = None,
    ):
        """
        Initialize the Config object.

        :param regressor_type: The type of the regressor.
        :param hyper_parameter_ranges: Optional custom hyperparameter ranges.
        """
        self.regressor_type: RegressorType = regressor_type
        self.hyper_parameter_ranges: HyperParameterRegistry = (
            hyper_parameter_ranges or DEFAULT_HYPER_PARAMETER_RANGES[regressor_type]
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create a Config object from a dictionary.

        :param config_dict: Dictionary containing configuration.
        :return: Config object.
        """
        regressor_type = config_dict.get("regressor_type")
        if not regressor_type:
            raise KeyError("Missing required key: 'regressor_type'.")
        try:
            regressor_type_enum = RegressorType[regressor_type]
        except KeyError as e:
            raise KeyError(f"Invalid regressor type: {regressor_type}.") from e

        hyper_parameter_ranges = config_dict.get("hyper_parameter_ranges")
        if hyper_parameter_ranges:
            hyper_parameter_ranges = HyperParameterRegistry.from_dict(hyper_parameter_ranges)

        return cls(
            regressor_type=regressor_type_enum,
            hyper_parameter_ranges=hyper_parameter_ranges,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Config object to a dictionary.

        :return: Dictionary representation of the Config object.
        """
        hyper_parameter_ranges_dict = None
        if self.hyper_parameter_ranges is not None:
            hyper_parameter_ranges_dict = self.hyper_parameter_ranges.to_dict()

        return {
            "regressor_type": self.regressor_type.name,
            "hyper_parameter_ranges": hyper_parameter_ranges_dict,
        }

    @classmethod
    def from_json(cls, filename: str) -> "Config":
        """
        Load a Config object from a JSON file.

        :param filename: Path to the JSON file.
        :return: Config object.
        """
        with open(filename, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_json(self, filename: str):
        """
        Save the Config object to a JSON file.

        :param filename: Path to the JSON file.
        """
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
