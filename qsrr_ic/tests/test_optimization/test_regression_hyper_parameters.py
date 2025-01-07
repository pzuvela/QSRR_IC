import pytest
from typing import Dict, Union, List

from qsrr_ic.enums import HyperParameterName
from qsrr_ic.domain_models import (
    HyperParameter,
    HyperParameterRange,
    HyperParameterRegistry
)


class TestHyperParameter:

    def test_initialization(self):
        hp = HyperParameter(5)
        assert hp.value == 5
        assert hp.type() == int

        hp = HyperParameter(5.0)
        assert hp.value == 5.0
        assert hp.type() == float

    def test_invalid_initialization(self):
        with pytest.raises(ValueError, match="must be a float or int"):
            HyperParameter("invalid")

    def test_immutability(self):
        hp = HyperParameter(5)
        with pytest.raises(ValueError, match="cannot be changed after instantiation"):
            hp.value = 6

    def test_equality(self):
        hp1 = HyperParameter(5)
        hp2 = HyperParameter(5)
        hp3 = HyperParameter(6)
        hp4 = HyperParameter(5.0)

        assert hp1 == hp2
        assert hp1 != hp3
        assert hp1 != hp4  # Different types should not be equal
        assert hp1 != "not a hyperparameter"

    def test_repr(self):
        hp = HyperParameter(5)
        assert repr(hp) == "HyperParameter(value=5)"


class TestHyperParameterRange:
    @pytest.fixture
    def valid_range(self):
        return HyperParameterRange(
            HyperParameter(1),
            HyperParameter(5)
        )

    def test_initialization(self, valid_range):
        assert valid_range.lower.value == 1
        assert valid_range.upper.value == 5
        assert valid_range.type() == int

    def test_invalid_type_combination(self):
        with pytest.raises(ValueError, match="must be of the same type"):
            HyperParameterRange(
                HyperParameter(1),
                HyperParameter(5.0)
            )

    def test_invalid_bounds(self):
        with pytest.raises(ValueError, match="cannot be greater than upper bound"):
            HyperParameterRange(
                HyperParameter(5),
                HyperParameter(1)
            )

    def test_immutability(self, valid_range):
        with pytest.raises(ValueError, match="cannot be changed after instantiation"):
            valid_range.lower = HyperParameter(2)

        with pytest.raises(ValueError, match="cannot be changed after instantiation"):
            valid_range.upper = HyperParameter(6)

    def test_get_static_value(self):
        static_range = HyperParameterRange(
            HyperParameter(5),
            HyperParameter(5)
        )
        assert static_range.get_static_value().value == 5

        dynamic_range = HyperParameterRange(
            HyperParameter(1),
            HyperParameter(5)
        )
        assert dynamic_range.get_static_value() is None

    def test_values_property(self, valid_range):
        assert valid_range.values == [1, 5]

    def test_values_immutability(self, valid_range):
        with pytest.raises(ValueError, match="cannot be changed after instantiation"):
            valid_range.values = [2, 6]

    def test_repr(self, valid_range):
        expected = "HyperParameterRange(lower=HyperParameter(value=1), upper=HyperParameter(value=5))"
        assert repr(valid_range) == expected


class TestHyperParameterRegistry:
    @pytest.fixture
    def empty_registry(self):
        return HyperParameterRegistry()

    @pytest.fixture
    def sample_registry(self, empty_registry):
        registry = empty_registry
        registry.add(
            HyperParameterName.N_ESTIMATORS,
            HyperParameter(100)
        )
        registry.add(
            HyperParameterName.LEARNING_RATE,
            HyperParameterRange(
                HyperParameter(0.01),
                HyperParameter(0.1)
            )
        )
        return registry

    def test_initialization(self, empty_registry):
        assert empty_registry.names() == []

    def test_add_single_parameter(self, empty_registry):
        empty_registry.add(
            HyperParameterName.N_ESTIMATORS,
            HyperParameter(100)
        )
        assert len(empty_registry.names()) == 1
        assert empty_registry.get(HyperParameterName.N_ESTIMATORS).value == 100

    def test_add_parameter_range(self, empty_registry):
        empty_registry.add(
            HyperParameterName.LEARNING_RATE,
            HyperParameterRange(
                HyperParameter(0.01),
                HyperParameter(0.1)
            )
        )
        param = empty_registry.get(HyperParameterName.LEARNING_RATE)
        assert isinstance(param, HyperParameterRange)
        assert param.lower.value == 0.01
        assert param.upper.value == 0.1

    def test_add_duplicate(self, sample_registry):
        with pytest.raises(ValueError, match="already exists in the collection"):
            sample_registry.add(
                HyperParameterName.N_ESTIMATORS,
                HyperParameter(200)
            )

    def test_add_wrong_type(self, empty_registry):
        with pytest.raises(ValueError, match="incorrect type"):
            empty_registry.add(
                HyperParameterName.N_ESTIMATORS,
                HyperParameter(1.5)  # Should be int
            )

    def test_get_nonexistent(self, empty_registry):
        with pytest.raises(KeyError, match="not found in the collection"):
            empty_registry.get(HyperParameterName.N_ESTIMATORS)

    def test_remove(self, sample_registry):
        sample_registry.remove(HyperParameterName.N_ESTIMATORS)
        assert HyperParameterName.N_ESTIMATORS not in sample_registry.names()

    def test_remove_nonexistent(self, empty_registry):
        with pytest.raises(KeyError, match="not found in the collection"):
            empty_registry.remove(HyperParameterName.N_ESTIMATORS)

    def test_to_dict(self, sample_registry):
        result = sample_registry.to_dict()
        assert result[HyperParameterName.N_ESTIMATORS.value] == 100
        assert result[HyperParameterName.LEARNING_RATE.value] == [0.01, 0.1]

    def test_from_dict_single_values(self):
        input_dict: Dict[HyperParameterName, Union[Union[List[float], float], Union[List[int], int]]] = {
            HyperParameterName.N_ESTIMATORS: 100,
            HyperParameterName.LEARNING_RATE: 0.01
        }
        registry = HyperParameterRegistry.from_dict(input_dict)
        assert isinstance(registry.get(HyperParameterName.N_ESTIMATORS), HyperParameter)
        assert isinstance(registry.get(HyperParameterName.LEARNING_RATE), HyperParameter)

    def test_from_dict_range_values(self):
        input_dict: Dict[HyperParameterName, Union[Union[List[float], float], Union[List[int], int]]] = {
            HyperParameterName.N_ESTIMATORS: [50, 150],
            HyperParameterName.LEARNING_RATE: [0.01, 0.1]
        }
        registry = HyperParameterRegistry.from_dict(input_dict)
        assert isinstance(registry.get(HyperParameterName.N_ESTIMATORS), HyperParameterRange)
        assert isinstance(registry.get(HyperParameterName.LEARNING_RATE), HyperParameterRange)

    def test_from_dict_invalid_range_length(self):
        input_dict: Dict[HyperParameterName, Union[Union[List[float], float], Union[List[int], int]]] = {
            HyperParameterName.N_ESTIMATORS: [50, 100, 150]
        }
        with pytest.raises(ValueError, match="must have exactly 2 elements"):
            HyperParameterRegistry.from_dict(input_dict)

    def test_from_dict_unknown_parameter(self):
        unknown_param = "unknown_param"
        with pytest.raises(ValueError) as e:
            HyperParameterRegistry.from_dict({unknown_param: 100})  # type: ignore
        assert f"'{unknown_param}' is not a valid HyperParameterName" == e.value.args[0]

    def test_repr(self, sample_registry):
        repr_str = repr(sample_registry)
        assert "HyperParameterRegistry" in repr_str
        assert "n_estimators" in repr_str
        assert "learning_rate" in repr_str
