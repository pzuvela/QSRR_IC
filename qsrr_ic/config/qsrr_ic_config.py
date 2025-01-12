from typing import (
    Any,
    Dict,
    Optional
)

from qsrr_ic.domain_models import (
    CrossValidationSettings,
    GlobalSearchSettings,
    HyperParameter,
    HyperParameterRange,
    HyperParameterRegistry
)
from qsrr_ic.enums import (
    CrossValidationType,
    HyperParameterName,
    RegressorType,
    TrainingType
)
from qsrr_ic.models.iso2grad.domain_models import Iso2GradSettings


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


from qsrr_ic.base import BaseConfig
from qsrr_ic.config import DatasetConfig


class HyperParameterConfig(BaseConfig):
    def __init__(
        self,
        regressor_type: RegressorType,
        hyper_parameter_registry: Optional[HyperParameterRegistry] = None
    ):
        """
        Initialize the Config object.

        :param regressor_type: The type of the regressor.
        :param hyper_parameter_registry: Optional hyperparameter registry.
        """
        self.regressor_type: RegressorType = regressor_type
        self.hyper_parameter_registry: HyperParameterRegistry = (
            hyper_parameter_registry or DEFAULT_HYPER_PARAMETER_RANGES[regressor_type]
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "HyperParameterConfig":
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

        hyper_parameters = config_dict.get("hyper_parameters")
        if hyper_parameters:
            hyper_parameters = HyperParameterRegistry.from_dict(hyper_parameters)

        return cls(
            regressor_type=regressor_type_enum,
            hyper_parameter_registry=hyper_parameters
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Config object to a dictionary.

        :return: Dictionary representation of the Config object.
        """
        hyper_parameters = None
        if self.hyper_parameter_registry is not None:
            hyper_parameters = self.hyper_parameter_registry.to_dict()

        return {
            "regressor_type": self.regressor_type.name,
            "hyper_parameters": hyper_parameters
        }


class Iso2GradConfig(BaseConfig):
    def __init__(
        self,
        iso2grad_settings: Iso2GradSettings
    ):
        self.iso2grad_settings = iso2grad_settings

    @classmethod
    def from_dict(cls, iso2grad_config_dict: Dict[str, Any]) -> "Iso2GradConfig":
        return cls(
            iso2grad_settings=Iso2GradSettings(
                integration_step=iso2grad_config_dict.get("integration_step", 0.01),
                n_jobs=iso2grad_config_dict.get("n_jobs", -1),
                verbosity=iso2grad_config_dict.get("verbosity", 10),
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "integration_step": self.iso2grad_settings.integration_step,
            "n_jobs": self.iso2grad_settings.n_jobs,
            "verbosity": self.iso2grad_settings.verbosity
        }


class TrainTestSplitConfig(BaseConfig):
    def __init__(
        self,
        test_ratio: float,
        random_seed: int,
        shuffle: bool
    ):
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.shuffle = shuffle

    @classmethod
    def from_dict(cls, train_test_split_dict: Dict[str, Any]) -> 'TrainTestSplitConfig':
        test_ratio = train_test_split_dict.get("test_ratio", 0.3)
        random_seed = train_test_split_dict.get("random_seed", 0)
        shuffle = train_test_split_dict.get("shuffle", True)
        return cls(
            test_ratio=test_ratio,
            random_seed=random_seed,
            shuffle=shuffle
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class CrossValidationConfig(BaseConfig):
    def __init__(
        self,
        cv_settings: CrossValidationSettings
    ):
        self.cv_settings = cv_settings

    def to_dict(self):
        return {
            "cv_type": self.cv_settings.cv_type.name,
            "n_splits": self.cv_settings.n_splits
        }

    @classmethod
    def from_dict(cls, cv_settings_dict: Dict[str, Any]) -> "CrossValidationConfig":
        return cls(
            cv_settings=CrossValidationSettings(
                cv_type=CrossValidationType[cv_settings_dict.get("cv_type", "LeaveOneOut")],
                n_splits=cv_settings_dict.get("n_splits", None)
            )
        )


class GlobalSearchConfig(BaseConfig):
    def __init__(
        self,
        global_search_settings: GlobalSearchSettings
    ):
        self.global_search_settings = global_search_settings

    def to_dict(self):
        return {
            "population_size": self.global_search_settings.population_size,
            "mutation_rate": self.global_search_settings.mutation_rate,
            "n_jobs": self.global_search_settings.n_jobs
        }

    @classmethod
    def from_dict(cls, global_search_settings_dict: Dict[str, Any]) -> "GlobalSearchConfig":
        return cls(
            global_search_settings=GlobalSearchSettings(
                population_size=global_search_settings_dict.get("population_size", 20),
                mutation_rate=global_search_settings_dict.get("mutation_rate", (1.5, 1.9)),
                n_jobs=global_search_settings_dict.get("n_jobs", -1)
            )
        )


class QsrrIcConfig(BaseConfig):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        training_type: TrainingType,
        train_test_split_config: TrainTestSplitConfig,
        hyper_parameter_config: Dict[RegressorType, HyperParameterConfig],
        iso2grad_config: Optional[Iso2GradConfig],
        cross_validation_config: Optional[CrossValidationConfig],
        global_search_config: Optional[GlobalSearchConfig]
    ):
        self.dataset_config: DatasetConfig = dataset_config
        self.training_type: TrainingType = training_type
        self.train_test_split_config: TrainTestSplitConfig = train_test_split_config
        self.hyper_parameter_config = hyper_parameter_config
        self.iso2grad_config = iso2grad_config
        self.cross_validation_config = cross_validation_config
        self.global_search_config = global_search_config

    @classmethod
    def from_dict(cls, qsrr_ic_config_dict: Dict[str, Any]) -> 'QsrrIcConfig':

        dataset_config = qsrr_ic_config_dict.get("dataset")
        if dataset_config:
            dataset_config = DatasetConfig(**dataset_config)

        training_type = qsrr_ic_config_dict.get("training_type")
        if training_type is None:
            raise KeyError("TrainingType must be supplied!")

        training_type_enum = TrainingType(training_type)

        train_test_split_config = TrainTestSplitConfig.from_dict(
            qsrr_ic_config_dict.get("train_test_split", {})
        )

        hyper_parameters = qsrr_ic_config_dict.get("hyper_parameters")

        if hyper_parameters is None:
            raise KeyError("Hyper-Parameters of the regressors must be supplied ! ")

        if len(hyper_parameters) == 0:
            raise KeyError("Hyper-Parameters of the regressors must be supplied ! ")

        hyper_parameter_config = {
            RegressorType[regressor_type]:HyperParameterConfig.from_dict(
                {
                    "regressor_type": regressor_type,
                    "hyper_parameters": config
                }
            )
            for regressor_type, config in hyper_parameters.items()
        }

        iso2grad_config_dict: Optional[Dict[str, Any]] = qsrr_ic_config_dict.get(
            "iso2grad_parameters",
            None
        )

        iso2grad_config: Optional[Iso2GradConfig] = None

        if iso2grad_config_dict is not None:
            iso2grad_config = Iso2GradConfig.from_dict(iso2grad_config_dict)

        cross_validation_config_dict: Optional[Dict[str, Any]] = qsrr_ic_config_dict.get(
            "cross_validation",
            None
        )

        cross_validation_config: Optional[CrossValidationConfig] = None

        if cross_validation_config_dict is not None:
            cross_validation_config = CrossValidationConfig.from_dict(cross_validation_config_dict)

        global_search_config_dict: Optional[Dict[str, Any]] = qsrr_ic_config_dict.get(
            "global_search",
            None
        )

        global_search_config: Optional[GlobalSearchConfig] = None

        if global_search_config_dict is not None:
            global_search_config = GlobalSearchConfig.from_dict(global_search_config_dict)

        return cls(
            dataset_config=dataset_config,
            training_type=training_type_enum,
            train_test_split_config=train_test_split_config,
            hyper_parameter_config=hyper_parameter_config,
            iso2grad_config=iso2grad_config,
            cross_validation_config=cross_validation_config,
            global_search_config=global_search_config
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Config object to a dictionary.

        :return: Dictionary representation of the QsrrIcConfig object.
        """

        iso2grad_parameters = None

        if self.iso2grad_config is not None:
            iso2grad_parameters = self.iso2grad_config.to_dict()

        cross_validation = None

        if self.cross_validation_config is not None:
            cross_validation = self.cross_validation_config.to_dict()

        global_search = None

        if self.global_search_config is not None:
            global_search = self.global_search_config.to_dict()

        return {
            "dataset": self.dataset_config.to_dict(),
            "training_type": self.training_type.value,
            "train_test_split": self.train_test_split_config.to_dict(),
            "hyper_parameters": {
                regressor_type.name:config.to_dict()["hyper_parameters"] for regressor_type, config in self.hyper_parameter_config.items()
            },
            "iso2grad_parameters": iso2grad_parameters,
            "cross_validation": cross_validation,
            "global_search": global_search
        }
