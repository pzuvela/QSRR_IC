import pytest

import numpy as np

from scipy.stats import norm


from qsrr_ic.analysis.srd._srd_visualizer import (  # noqa (importing from private API for tests)
    SRDModel,
    SRDViewModel,
    SrdVisualizerData
)


class TestSRDModel:

    @pytest.fixture
    def mock_data(self):
        np.random.seed(42)
        normalized_srds = np.random.uniform(0, 100, size=20)
        normalized_random_srds = np.random.normal(50, 10, size=1000)
        return normalized_srds, normalized_random_srds

    def test_model_initialization(self, mock_data):
        normalized_srds, normalized_random_srds = mock_data
        model = SRDModel(normalized_srds, normalized_random_srds)

        assert np.array_equal(model.normalized_srds, normalized_srds)
        assert np.array_equal(model.normalized_random_srds, normalized_random_srds)

    def test_distribution_fitting(self, mock_data):
        _, normalized_random_srds = mock_data
        model = SRDModel([], normalized_random_srds)

        assert model.mu == pytest.approx(np.mean(normalized_random_srds), rel=1e-3)
        assert model.std == pytest.approx(np.std(normalized_random_srds, ddof=1), rel=1e-3)

    def test_pdf_calculation(self, mock_data):
        _, normalized_random_srds = mock_data
        model = SRDModel([], normalized_random_srds)
        x_values = np.linspace(0, 100, 50)
        pdf_values = model.calculate_pdf(x_values)

        expected_pdf = norm.pdf(x_values, loc=model.mu, scale=model.std)
        assert np.allclose(pdf_values, expected_pdf, atol=1e-6)

class TestSRDViewModel:
    @pytest.fixture
    def mock_data(self):
        np.random.seed(42)
        normalized_srds = np.random.uniform(0, 100, size=20)
        normalized_random_srds = np.random.normal(50, 10, size=1000)
        return normalized_srds, normalized_random_srds

    def test_prepare_data_for_plotting(self, mock_data):
        normalized_srds, normalized_random_srds = mock_data
        model = SRDModel(normalized_srds, normalized_random_srds)
        view_model = SRDViewModel(model)

        data = view_model.prepare_data_for_plotting()

        assert isinstance(data, SrdVisualizerData)
        assert np.array_equal(data.normalized_srds, normalized_srds)
        assert np.array_equal(data.normalized_random_srds, normalized_random_srds)
        assert data.mu == pytest.approx(model.mu, rel=1e-3)
        assert data.std == pytest.approx(model.std, rel=1e-3)

        # Check x_values and pdf_values
        assert len(data.x_values) == 100  # Based on SrdVisualizerConstants.N_POINTS
        assert len(data.pdf_values) == 100
        expected_pdf = norm.pdf(data.x_values, loc=model.mu, scale=model.std)
        assert np.allclose(data.pdf_values, expected_pdf, atol=1e-6)
