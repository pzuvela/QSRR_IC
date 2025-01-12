import pytest

from qsrr_ic.config import (
    DatasetConfig,
    HyperParameterConfig,
    Iso2GradConfig,
    TrainTestSplitConfig,
    CrossValidationConfig,
    GlobalSearchConfig,
    QsrrIcConfig,
)
from qsrr_ic.config.qsrr_ic_config import DEFAULT_HYPER_PARAMETER_RANGES
from qsrr_ic.enums import CrossValidationType, RegressorType


# Fixture for DatasetConfig
@pytest.fixture
def valid_config_data_dataset():
    return {
        "molecular_descriptors_for_qsrr_training_path": "path/to/qsrr_training.csv",
        "isocratic_retention_path": "path/to/isocratic_retention.csv",
        "molecular_descriptors_for_iso2grad_path": "path/to/iso2grad.csv",
        "gradient_void_times_path": "path/to/gradient_void_times.csv",
        "gradient_profiles_path": "path/to/gradient_profiles.csv",
        "gradient_retention_path": "path/to/gradient_retention.csv",
    }


# Fixture for TrainTestSplitConfig
@pytest.fixture
def valid_config_data_train_test_split():
    return {
        "test_ratio": 0.3,
        "random_seed": 42,
        "shuffle": True,
    }


class TestDatasetConfig:
    @pytest.fixture
    def valid_config_data(self):
        return {
            "molecular_descriptors_for_qsrr_training_path": "path/to/qsrr_training.csv",
            "isocratic_retention_path": "path/to/isocratic_retention.csv",
            "molecular_descriptors_for_iso2grad_path": "path/to/iso2grad.csv",
            "gradient_void_times_path": "path/to/gradient_void_times.csv",
            "gradient_profiles_path": "path/to/gradient_profiles.csv",
            "gradient_retention_path": "path/to/gradient_retention.csv",
        }

    def test_to_dict(self, valid_config_data):
        config = DatasetConfig(**valid_config_data)
        assert config.to_dict() == valid_config_data

    def test_from_dict(self, valid_config_data):
        config = DatasetConfig.from_dict(valid_config_data)
        assert isinstance(config, DatasetConfig)
        assert config.to_dict() == valid_config_data

    def test_missing_required_field(self, valid_config_data):
        invalid_data = valid_config_data.copy()
        del invalid_data["isocratic_retention_path"]
        with pytest.raises(TypeError):
            DatasetConfig.from_dict(invalid_data)


class TestHyperParameterConfig:
    @pytest.fixture
    def valid_config_data(self):
        return {
            "regressor_type": "PLS",
            "hyper_parameters": None,
        }

    def test_to_dict(self, valid_config_data):
        config = HyperParameterConfig.from_dict(valid_config_data)
        assert config.to_dict()["regressor_type"] == "PLS"
        assert config.to_dict()["hyper_parameters"] == DEFAULT_HYPER_PARAMETER_RANGES[RegressorType.PLS].to_dict()

    def test_from_dict(self, valid_config_data):
        config = HyperParameterConfig.from_dict(valid_config_data)
        assert isinstance(config, HyperParameterConfig)

    def test_invalid_regressor_type(self, valid_config_data):
        invalid_data = valid_config_data.copy()
        invalid_data["regressor_type"] = "INVALID_TYPE"
        with pytest.raises(KeyError):
            HyperParameterConfig.from_dict(invalid_data)


class TestIso2GradConfig:
    @pytest.fixture
    def valid_config_data(self):
        return {
            "integration_step": 0.01,
            "n_jobs": -1,
            "verbosity": 10,
        }

    def test_to_dict(self, valid_config_data):
        config = Iso2GradConfig.from_dict(valid_config_data)
        assert config.to_dict() == valid_config_data

    def test_from_dict(self, valid_config_data):
        config = Iso2GradConfig.from_dict(valid_config_data)
        assert isinstance(config, Iso2GradConfig)

    def test_missing_field(self, valid_config_data):
        invalid_data = valid_config_data.copy()
        del invalid_data["integration_step"]
        config = Iso2GradConfig.from_dict(invalid_data)
        assert config.iso2grad_settings.integration_step == 0.01


class TestTrainTestSplitConfig:
    @pytest.fixture
    def valid_config_data(self):
        return {
            "test_ratio": 0.3,
            "random_seed": 42,
            "shuffle": True,
        }

    def test_to_dict(self, valid_config_data):
        config = TrainTestSplitConfig.from_dict(valid_config_data)
        assert config.to_dict() == valid_config_data

    def test_from_dict(self, valid_config_data):
        config = TrainTestSplitConfig.from_dict(valid_config_data)
        assert isinstance(config, TrainTestSplitConfig)

    def test_missing_field_defaults(self):
        config = TrainTestSplitConfig.from_dict({})
        assert config.test_ratio == 0.3
        assert config.random_seed == 0
        assert config.shuffle is True


class TestCrossValidationConfig:
    @pytest.fixture
    def valid_config_data(self):
        return {
            "cv_type": "KFold",
            "n_splits": 5,
        }

    def test_to_dict(self, valid_config_data):
        config = CrossValidationConfig.from_dict(valid_config_data)
        assert config.to_dict() == valid_config_data

    def test_from_dict(self, valid_config_data):
        config = CrossValidationConfig.from_dict(valid_config_data)
        assert isinstance(config, CrossValidationConfig)

    def test_default_values(self):
        config = CrossValidationConfig.from_dict({"cv_type": "LeaveOneOut"})
        assert config.cv_settings.cv_type == CrossValidationType.LeaveOneOut
        assert config.cv_settings.n_splits is None


class TestGlobalSearchConfig:
    @pytest.fixture
    def valid_config_data(self):
        return {
            "population_size": 20,
            "mutation_rate": (1.5, 1.9),
            "n_jobs": -1,
        }

    def test_to_dict(self, valid_config_data):
        config = GlobalSearchConfig.from_dict(valid_config_data)
        assert config.to_dict() == valid_config_data

    def test_from_dict(self, valid_config_data):
        config = GlobalSearchConfig.from_dict(valid_config_data)
        assert isinstance(config, GlobalSearchConfig)


class TestQsrrIcConfig:
    @pytest.fixture
    def valid_config_data(self, valid_config_data_train_test_split, valid_config_data_dataset):
        return {
            "dataset": valid_config_data_dataset,
            "training_type": "single_train",
            "train_test_split": valid_config_data_train_test_split,
            "hyper_parameters": {
                "PLS": {"n_components": 2},
            },
            "iso2grad_parameters": {"integration_step": 0.01, "n_jobs": -1, "verbosity": 10},
            "cross_validation": {"cv_type": "KFold", "n_splits": 5},
            "global_search": {"population_size": 20, "mutation_rate": (1.5, 1.9), "n_jobs": -1},
        }

    def test_to_dict(self, valid_config_data):
        config = QsrrIcConfig.from_dict(valid_config_data)
        assert config.to_dict() == valid_config_data

    def test_from_dict(self, valid_config_data):
        config = QsrrIcConfig.from_dict(valid_config_data)
        assert isinstance(config, QsrrIcConfig)

    def test_missing_required_field(self, valid_config_data):
        invalid_data = valid_config_data.copy()
        del invalid_data["training_type"]
        with pytest.raises(KeyError):
            QsrrIcConfig.from_dict(invalid_data)
