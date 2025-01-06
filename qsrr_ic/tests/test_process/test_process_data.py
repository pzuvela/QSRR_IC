
from unittest.mock import MagicMock

import numpy as np

import pandas as pd

import pytest

from qsrr_ic.constants import QsrrIcConstants
from qsrr_ic.datasets.qsrr_ic_dataset import QsrrIcDataset
from qsrr_ic.domain_models import QsrrIcData
from qsrr_ic.process.process_data import ProcessData


@pytest.fixture
def mock_qsrr_ic_dataset():

    # Mocking the QsrrIcDataset object
    mock_dataset = MagicMock(QsrrIcDataset)

    # Mocking DataFrame returns for each property
    mock_dataset.molecular_descriptors_for_qsrr_training_df = pd.DataFrame({
        QsrrIcConstants.ANALYTE: ["Analyte1", "Analyte2", "Analyte3"],
        'col1': [1, 2, 3],
        'col2': [4, 5, 6]
    })
    mock_dataset.isocratic_retention_df = pd.DataFrame({
        QsrrIcConstants.LOGK: [1.0, 2.0, 3.0]
    })
    mock_dataset.molecular_descriptors_for_iso2grad_df = pd.DataFrame({
        QsrrIcConstants.ANALYTE: ["Analyte1", "Analyte2", "Analyte3"],
        'col1': [7, 8, 9],
        'col2': [10, 11, 12]
    })
    mock_dataset.gradient_void_times_df = pd.DataFrame({
        QsrrIcConstants.GRADIENT_PROFILE: ["Profile1", "Profile2", "Profile3"],
        'col1': [13, 14, 15],
        'col2': [16, 17, 18]
    })
    mock_dataset.gradient_profiles_df = pd.DataFrame({
        'col1': [19, 20, 21],
        'col2': [22, 23, 24]
    })
    mock_dataset.gradient_retention_df = pd.DataFrame({
        QsrrIcConstants.TG: [25.0, 26.0, 27.0]
    })

    return mock_dataset


class TestProcessData:

    def test_process_molecular_descriptors_for_qsrr_training_data(self, mock_qsrr_ic_dataset):
        # Creating the ProcessData instance
        process_data = ProcessData(mock_qsrr_ic_dataset)

        # Process the molecular descriptors for qsrr training data
        processed_data = process_data._process_molecular_descriptors_for_qsrr_training_data()

        # Verify the processed data (remove the "ANALYTE" column if present)
        expected_result = mock_qsrr_ic_dataset.molecular_descriptors_for_qsrr_training_df.drop(
            columns=[QsrrIcConstants.ANALYTE],
            errors='ignore'
        ).values

        assert np.array_equal(processed_data, expected_result)

    def test_process_molecular_descriptors_for_iso2grad_data(self, mock_qsrr_ic_dataset):
        # Creating the ProcessData instance
        process_data = ProcessData(mock_qsrr_ic_dataset)

        # Process the molecular descriptors for iso2grad data
        processed_data = process_data._process_molecular_descriptors_for_iso2grad_data()

        # Verify the processed data (remove the "ANALYTE" column if present)
        expected_result = mock_qsrr_ic_dataset.molecular_descriptors_for_iso2grad_df.drop(
            columns=[QsrrIcConstants.ANALYTE], errors='ignore').values

        assert np.array_equal(processed_data, expected_result)

    def test_process_gradient_void_times(self, mock_qsrr_ic_dataset):
        # Creating the ProcessData instance
        process_data = ProcessData(mock_qsrr_ic_dataset)

        # Process the gradient void times
        processed_data = process_data._process_gradient_void_times()

        # Verify the processed data (remove the "GRADIENT_PROFILE" column if present)
        expected_result = mock_qsrr_ic_dataset.gradient_void_times_df.drop(
            columns=[QsrrIcConstants.GRADIENT_PROFILE], errors='ignore').values

        assert np.array_equal(processed_data, expected_result)

    def test_process_data(self, mock_qsrr_ic_dataset):
        # Creating the ProcessData instance
        process_data = ProcessData(mock_qsrr_ic_dataset)

        # Process the entire dataset
        processed_data = process_data.process()

        # Verify the output is a QsrrIcData object
        assert isinstance(processed_data, QsrrIcData)

        # Check if the processed data is as expected
        assert np.array_equal(processed_data.molecular_descriptors_for_qsrr_training.shape, (3, 2))
        assert np.array_equal(processed_data.isocratic_retention.shape, (3, 1))
        assert np.array_equal(processed_data.molecular_descriptors_for_iso2grad.shape, (3, 2))
        assert np.array_equal(processed_data.gradient_void_times.shape, (3, 2))
        assert np.array_equal(processed_data.gradient_profiles.shape, (3, 2))
        assert np.array_equal(processed_data.gradient_retention.shape, (3, 1))
