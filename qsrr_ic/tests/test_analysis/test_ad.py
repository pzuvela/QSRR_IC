import pytest
import numpy as np
from qsrr_ic.models.qsrr.domain_models import QsrrData
from qsrr_ic.analysis.ad import (
    ApplicabilityDomain,
    ApplicabilityDomainPlot
)
from qsrr_ic.analysis.ad.domain_models import ApplicabilityDomainData


@pytest.fixture
def mock_data():
    """
    Mock data for testing ApplicabilityDomain.
    """
    x_train = np.array([[1, 2], [3, 4], [5, 6]])  # Descriptors for training
    y_train = np.array([10, 20, 30])  # Retention values for training
    x_test = np.array([[2, 3], [4, 5]])  # Descriptors for testing
    y_test = np.array([15, 25])  # Retention values for testing

    predictions_train = np.array([11, 19, 31])  # Predictions for training
    predictions_test = np.array([14, 26])  # Predictions for testing

    data_train = QsrrData(x=x_train, y=y_train)
    data_test = QsrrData(x=x_test, y=y_test)
    predictions_train = QsrrData(x=x_train, y=predictions_train)
    predictions_test = QsrrData(x=x_test, y=predictions_test)

    ad_data = ApplicabilityDomainData(
        data_train=data_train,
        data_test=data_test,
        predictions_train=predictions_train,
        predictions_test=predictions_test,
    )
    return ad_data


class TestApplicabilityDomain:
    """
    Test suite for ApplicabilityDomain and related functionalities.
    """

    def test_hat_matrix_train(self, mock_data):
        """
        Test computation of the training hat matrix.
        """
        ad = ApplicabilityDomain(data=mock_data)
        hat_matrix_train = ad.hat_matrix_train
        assert hat_matrix_train.shape == (3,)
        assert np.all(hat_matrix_train > 0)

    def test_hat_matrix_test(self, mock_data):
        """
        Test computation of the testing hat matrix.
        """
        ad = ApplicabilityDomain(data=mock_data)
        hat_matrix_test = ad.hat_matrix_test
        assert hat_matrix_test.shape == (2,)
        assert np.all(hat_matrix_test > 0)

    def test_residuals_train(self, mock_data):
        """
        Test computation of the training residuals.
        """
        ad = ApplicabilityDomain(data=mock_data)
        residuals_train = ad.residuals_train
        expected_residuals = np.array([1, -1, 1])  # predictions - actual
        np.testing.assert_array_equal(residuals_train, expected_residuals)

    def test_residuals_test(self, mock_data):
        """
        Test computation of the testing residuals.
        """
        ad = ApplicabilityDomain(data=mock_data)
        residuals_test = ad.residuals_test
        expected_residuals = np.array([-1, 1])  # predictions - actual
        np.testing.assert_array_equal(residuals_test, expected_residuals)

    def test_standardized_residuals_train(self, mock_data):
        """
        Test computation of standardized residuals for training data.
        """
        ad = ApplicabilityDomain(data=mock_data)
        std_residuals_train = ad.std_residuals_train
        residuals_train = ad.residuals_train
        std_dev = np.std(residuals_train, ddof=1)
        expected_std_residuals = residuals_train / std_dev
        np.testing.assert_array_almost_equal(std_residuals_train, expected_std_residuals)

    def test_standardized_residuals_test(self, mock_data):
        """
        Test computation of standardized residuals for testing data.
        """
        ad = ApplicabilityDomain(data=mock_data)
        std_residuals_test = ad.std_residuals_test
        residuals_test = ad.residuals_test
        std_dev = np.std(ad.residuals_train, ddof=1)  # Standard deviation of training residuals
        expected_std_residuals_test = residuals_test / std_dev
        np.testing.assert_array_almost_equal(std_residuals_test, expected_std_residuals_test)

    def test_critical_hat_value(self, mock_data):
        """
        Test computation of the critical hat value.
        """
        ad = ApplicabilityDomain(data=mock_data)
        critical_hat = ad.critical_hat
        x_train = mock_data.data_train.x
        m, n = x_train.shape
        k = m + 1  # Parameters + intercept
        expected_critical_hat = 3 * (k / n)
        assert pytest.approx(critical_hat) == expected_critical_hat

    def test_plot_williams(self, mock_data):
        """
        Test plotting of the Williams plot.
        """
        ad = ApplicabilityDomain(data=mock_data)
        plotter = ApplicabilityDomainPlot(ad=ad)

        # Ensure that the plot runs without errors
        try:
            plotter.plot_williams(n_sigma=3, fig_size=(8, 6))
        except Exception as e:
            pytest.fail(f"plot_williams raised an exception: {e}")
