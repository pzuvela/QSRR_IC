import os
import sys
from typing import Any

import pytest

import numpy as np

import pandas as pd

from qsrr_ic.enums import RegressorType
from qsrr_ic.load import (
    QsrrIcDataset,
    QsrrIcData
)
from qsrr_ic.models.iso2grad import Iso2Grad
from qsrr_ic.models.iso2grad.domain_models import (
    Iso2GradData,
    Iso2GradSettings
)
from qsrr_ic.models.qsrr import QsrrModel
from qsrr_ic.models.qsrr.domain_models import QsrrData
from qsrr_ic.process import ProcessData
from qsrr_ic.tests.constants import TestPaths


datasets_path: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "datasets"
)

if datasets_path not in sys.path:
    sys.path.append(datasets_path)

from datasets import load_qsrr_ic_dataset

@pytest.fixture
def qsrr_model(data: QsrrIcData):
    qsrr_model = QsrrModel(
        regressor_type=RegressorType.PLS,
        qsrr_train_data=QsrrData(
            x=data.molecular_descriptors_for_qsrr_training,
            y=data.isocratic_retention
        )
    )
    qsrr_model.fit()
    return qsrr_model

@pytest.fixture
def data():
    dataset: QsrrIcDataset = load_qsrr_ic_dataset()
    data: QsrrIcData = ProcessData(dataset).process()
    return data


@pytest.fixture
def iso2grad(data: QsrrIcData, qsrr_model: Any):

    # Instantiate the data & settings required by the Iso2grad model
    iso2grad_data = Iso2GradData(
        isocratic_model_predictors=data.molecular_descriptors_for_iso2grad,
        gradient_void_times=data.gradient_void_times,
        gradient_retention_profiles=data.gradient_profiles
    )
    iso2grad_settings = Iso2GradSettings(
        integration_step=0.01,
        n_jobs=-1,
        verbosity=10
    )

    # Run the model
    iso2grad_model = Iso2Grad(
        qsrr_model=qsrr_model,
        iso2grad_data=iso2grad_data,
        iso2grad_settings=iso2grad_settings
    )

    return iso2grad_model


class TestIso2Grad:

    def test_calculate_concentrations(self, iso2grad: Iso2Grad):

        # Test concentration calculation for gradient steps
        tg1, tg2 = 0.0, 0.1
        ti1, ti2 = 1.0, 3.0
        conc1, conc2 = 2.0, 4.0
        slope = 0.5

        conc_grad1, conc_grad2 = iso2grad._calculate_concentrations(tg1, tg2, ti1, conc1, slope, ti2, conc2)

        # Assert the calculated concentrations are correct
        assert conc_grad1 == 1.5
        assert conc_grad2 == 1.55

    def test_calculate_average_k(self, iso2grad: Iso2Grad, qsrr_model: Any):
        # Test the calculation of the average retention factor k
        conc_grad1, conc_grad2 = 2.0, 3.0
        analyte_idx = 0

        k_avg = iso2grad._calculate_average_k(conc_grad1, conc_grad2, analyte_idx)

        # Assert that the average k is correct
        assert np.isclose(
            k_avg,
            3.91286262,
            atol=1e-5
        ).all()

    def test_integrate_retention_time(self, iso2grad: Iso2Grad):
        # Test retention time integration
        tg1, tg2 = 0.0, 0.1
        k_avg = 2.0
        i_prev = 0.0

        i_next = iso2grad._integrate_retention_time(tg1, tg2, k_avg, i_prev)

        # Assert the integration step is correct
        assert i_next == 0.05

    def test_process_analyte(self, iso2grad: Iso2Grad):
        # Test the processing of a single analyte (calculating retention time)
        profile_idx = 0
        analyte_idx = 0

        tr_g = iso2grad._process_analyte(profile_idx, analyte_idx)

        # Assert that the retention time is calculated correctly
        assert tr_g > 0  # We expect a positive retention time

    def test_fit_profile(self, iso2grad: Iso2Grad):
        # Test the fit profile function
        profile_idx = 0

        retention_times = iso2grad._fit_profile(profile_idx)

        # Assert that retention times for all analytes are returned
        assert retention_times.shape == (29,)  # Adjust based on mock data size
        assert retention_times[0] > 0  # Ensure the retention time is positive

    def test_iso2grad_train(self, iso2grad: Iso2Grad):

        iso2grad.fit()

        # Validate results
        golden_results = pd.read_csv(
            TestPaths.ISO2GRAD_TEST_RESULTS_PATH,
            float_precision="round_trip"
        ).values

        assert np.isclose(
            iso2grad.results.gradient_retention_times,
            golden_results,
            atol=1e-5
        ).all()


if __name__ == "__main__":
    pytest.main()
