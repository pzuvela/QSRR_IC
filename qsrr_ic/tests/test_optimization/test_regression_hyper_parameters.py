import pytest

from qsrr_ic.optimization.domain_models import (
    HyperParameter,
    HyperParameterRange,
    HyperParameterRegistry
)


class TestHyperParameter:
    
    def test_initialization_valid(self):
        # Test valid initialization of HyperParameter
        param = HyperParameter(0.1)
        assert param.value == 0.1

    def test_initialization_invalid(self):
        # Test invalid initialization (non-numeric values should raise ValueError)
        with pytest.raises(ValueError):
            HyperParameter("invalid_value")

    def test_immutable(self):
        # Test that the value of HyperParameter is immutable
        param = HyperParameter(0.1)
        with pytest.raises(ValueError):
            param.value = 0.2


class TestHyperParameterRange:

    def test_initialization_valid(self):
        # Test valid initialization of HyperParameterRange
        lower = HyperParameter(0.01)
        upper = HyperParameter(0.1)
        param_range = HyperParameterRange(lower, upper)
        assert param_range.lower.value == 0.01
        assert param_range.upper.value == 0.1

    def test_get_static_value(self):
        # Test that get_static_value returns lower bound if bounds are equal
        lower = HyperParameter(0.05)
        upper = HyperParameter(0.05)
        param_range = HyperParameterRange(lower, upper)
        static_value = param_range.get_static_value()
        assert static_value == lower
        assert static_value.value == 0.05


class TestHyperParameterRegistry:

    def test_add_and_get(self):
        # Test adding and retrieving HyperParameters and HyperParameterRanges
        param1 = HyperParameter(0.1)
        param_range = HyperParameterRange(HyperParameter(0.01), HyperParameter(0.1))

        registry = HyperParameterRegistry()
        registry.add("param1", param1)
        registry.add("param_range", param_range)

        retrieved_param = registry.get("param1")
        retrieved_range = registry.get("param_range")

        assert retrieved_param.value == 0.1
        assert retrieved_range.lower.value == 0.01
        assert retrieved_range.upper.value == 0.1

    def test_register_duplicate(self):
        # Test registering a duplicate entry (should raise an error)
        param1 = HyperParameter(0.1)
        registry = HyperParameterRegistry()
        registry.add("param1", param1)

        with pytest.raises(ValueError):
            registry.add("param1", param1)

    def test_unregister(self):
        # Test unregistering an existing entry
        param1 = HyperParameter(0.1)
        registry = HyperParameterRegistry()
        registry.add("param1", param1)

        registry.remove("param1")

        with pytest.raises(KeyError):
            registry.get("param1")

    def test_get_non_existent(self):

        # Test retrieving a non-existent entry (should raise KeyError)
        registry = HyperParameterRegistry()

        with pytest.raises(KeyError):
            registry.get("non_existent")

if __name__ == "__main__":
    pytest.main()
