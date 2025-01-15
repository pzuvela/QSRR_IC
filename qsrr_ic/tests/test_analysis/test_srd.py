import numpy as np

import pytest

from qsrr_ic.analysis.srd.domain_models import SrdSettings
from qsrr_ic.analysis.srd.enums import GoldenReference
from qsrr_ic.analysis.srd import SumOfRankingDifferences

@pytest.fixture
def mock_inputs():
    # Mock a small ndarray of inputs for testing purposes
    return np.array([[0.1, 0.4, 0.3], [0.2, 0.3, 0.6], [0.5, 0.2, 0.1]])

@pytest.fixture
def srd_instance(mock_inputs):
    # Create an instance of the SRD class
    return SumOfRankingDifferences(inputs=mock_inputs)

@pytest.fixture
def golden_reference():
    # Create a simple golden reference ndarray
    return np.array([0.1, 0.3, 0.5])

@pytest.fixture
def golden_reference_enum():
    # Using GoldenReference enum
    return GoldenReference.Mean


class TestSumOfRankingDifferences:

    def test_golden_reference_setting(self, srd_instance, golden_reference):
        # Test the setter and getter of golden_reference property
        with pytest.raises(ValueError):
            srd_instance.golden_reference = golden_reference


    def test_golden_ranking(self, srd_instance, golden_reference):
        expected_ranking = np.argsort(golden_reference)
        assert np.array_equal(srd_instance.golden_ranking, expected_ranking)

    def test_srd_computation(self, srd_instance):
        # Test SRD computation
        assert srd_instance.srds is not None
        assert srd_instance.srds.shape[0] == srd_instance.inputs.shape[1]  # Should be the number of columns

    def test_normalized_srd(self, srd_instance):
        # Test normalized SRD computation
        normalized_srds = srd_instance.normalized_srds
        assert normalized_srds is not None
        assert np.all(normalized_srds <= 100)  # It should be normalized to a percentage

    def test_ideal_ranking(self, srd_instance):
        # Test the ideal ranking is being set correctly
        ideal_ranking = srd_instance.ideal_ranking
        expected_ideal_ranking = np.arange(srd_instance.inputs.shape[0]).reshape(-1, 1)
        assert np.array_equal(ideal_ranking, expected_ideal_ranking)

    def test_srd_max_property(self, srd_instance):
        # Test the srd_max property calculation
        srd_max = srd_instance.srd_max
        assert srd_max > 0  # It should return a positive value
        assert isinstance(srd_max, float)

    def test_invalid_golden_reference_type(self):
        # Test invalid golden reference type raises an error
        with pytest.raises(TypeError):
            SumOfRankingDifferences(inputs=np.array([[0.1, 0.4], [0.3, 0.2]]), golden_reference="invalid")

    def test_srd_settings_initialization(self):
        # Test that settings are initialized correctly (using a mock)
        settings = SrdSettings(n_iterations=500, exact=True)
        assert settings.n_iterations == 500
        assert settings.exact is True
