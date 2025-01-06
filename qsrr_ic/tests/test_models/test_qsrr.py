from unittest.mock import MagicMock

import pytest
import numpy as np

from qsrr_ic.models.qsrr.domain_models import QsrrData, QsrrResults
from qsrr_ic.models.qsrr.enums import RegressorType
from qsrr_ic.models.qsrr import QsrrModel


@pytest.fixture
def mock_data():
    x_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    x_test = np.random.rand(50, 10)
    y_test = np.random.rand(50)

    qsrr_train_data = QsrrData(x=x_train, y=y_train)
    qsrr_test_data = QsrrData(x=x_test, y=y_test)

    return qsrr_train_data, qsrr_test_data


class TestQsrrModel:

    @pytest.mark.parametrize("regressor_type", list(RegressorType))
    def test_initialization(self, mock_data, regressor_type):
        qsrr_train_data, qsrr_test_data = mock_data
        model = QsrrModel(
            regressor_type=regressor_type,
            qsrr_train_data=qsrr_train_data,
            qsrr_test_data=qsrr_test_data
        )

        # Test initialization
        assert model.regressor_type == regressor_type
        assert model.qsrr_train_data == qsrr_train_data
        assert model.qsrr_test_data == qsrr_test_data
        assert not model.is_fitted
        assert model.train_results is None
        assert model.test_results is None

    def test_invalid_train_data(self):
        with pytest.raises(TypeError):
            QsrrModel(
                regressor_type=RegressorType.xGB,
                qsrr_train_data=None,
                qsrr_test_data=None
            )

    @pytest.mark.parametrize("regressor_type", list(RegressorType))
    def test_fit_model(self, mock_data, regressor_type):
        qsrr_train_data, _ = mock_data
        model = QsrrModel(
            regressor_type=regressor_type,
            qsrr_train_data=qsrr_train_data,
            qsrr_test_data=None
        )

        # Test fitting the model
        model.fit()
        assert model.is_fitted
        assert model._QsrrModel__train_results is None  # Reset train results after fitting

    @pytest.mark.parametrize("regressor_type", list(RegressorType))
    def test_train_results_property(self, mock_data, regressor_type):
        qsrr_train_data, _ = mock_data
        model = QsrrModel(
            regressor_type=regressor_type,
            qsrr_train_data=qsrr_train_data,
            qsrr_test_data=None
        )

        # Test train results after fitting
        model.fit()
        train_results = model.train_results
        assert isinstance(train_results, QsrrResults)
        assert train_results.qsrr_predictions is not None

    @pytest.mark.parametrize("regressor_type", list(RegressorType))
    def test_test_results_property(self, mock_data, regressor_type):
        qsrr_train_data, qsrr_test_data = mock_data
        model = QsrrModel(
            regressor_type=regressor_type,
            qsrr_train_data=qsrr_train_data,
            qsrr_test_data=qsrr_test_data
        )

        # Test test results after fitting
        model.fit()
        test_results = model.test_results
        assert isinstance(test_results, QsrrResults)
        assert test_results.qsrr_predictions is not None

    @pytest.mark.parametrize("regressor_type", list(RegressorType))
    def test_pls_r2_property(self, mock_data, regressor_type):
        qsrr_train_data, _ = mock_data
        model = QsrrModel(
            regressor_type=RegressorType.PLS,
            qsrr_train_data=qsrr_train_data,
            qsrr_test_data=None
        )

        # Test PLS R² calculation
        model.fit()

        # Mock model's x_scores_ attribute for PLS variance calculation
        model.model.x_scores_ = np.random.rand(100, 10)

        pls_r2 = model.pls_r2
        assert isinstance(pls_r2, np.ndarray)
        assert pls_r2.shape[0] == 2

    @pytest.mark.parametrize("regressor_type", list(RegressorType))
    def test_to_pickle(self, mock_data, regressor_type):
        qsrr_train_data, _ = mock_data
        model = QsrrModel(
            regressor_type=regressor_type,
            qsrr_train_data=qsrr_train_data,
            qsrr_test_data=None
        )

        # Test saving model to a pickle file
        model.fit()
        assert model.is_fitted

        # Mock joblib.dump to avoid actual file I/O during the test
        joblib_mock = MagicMock()
        with pytest.MonkeyPatch .context() as m:
            m.setattr('joblib.dump', joblib_mock)
            model.to_pickle('model.pkl')
            joblib_mock.assert_called_once_with(model.model, 'model.pkl')

    @pytest.mark.parametrize("regressor_type", list(RegressorType))
    def test_invalid_fit_and_pickle(self, mock_data, regressor_type):
        qsrr_train_data, _ = mock_data
        model = QsrrModel(
            regressor_type=regressor_type,
            qsrr_train_data=qsrr_train_data,
            qsrr_test_data=None
        )

        # Try saving an unfit model
        with pytest.raises(RuntimeError):
            model.to_pickle('model.pkl')  # Should fail as the model is not fitted
