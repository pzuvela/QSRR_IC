import pytest

import numpy as np
from numpy import ndarray

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from qsrr_ic.datasets import qsrr_ic_dataset
from qsrr_ic.datasets.qsrr_ic_dataset import QsrrIcDataset
from qsrr_ic.domain_models import QsrrIcData
from qsrr_ic.models.iso2grad import Iso2Grad
from qsrr_ic.models.iso2grad.domain_models import (
    Iso2GradData,
    Iso2GradSettings
)
from qsrr_ic.process import ProcessData
from qsrr_ic.tests.constants import TestPaths


class TestIso2Grad:

    def test_iso2grad_train(self):

        # Load and prepare data

        dataset: QsrrIcDataset = qsrr_ic_dataset.load_dataset()
        data: QsrrIcData = ProcessData(dataset).process()

        scaler: StandardScaler = StandardScaler()

        molecular_descriptors_scaled: ndarray = scaler.fit_transform(
            data.molecular_descriptors_for_qsrr_training
        )

        # Fit a QSRR model (MLR model as example)

        qsrr_model = LinearRegression()
        qsrr_model.fit(
            molecular_descriptors_scaled,
            data.isocratic_retention
        )

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
            scaler=scaler,
            iso2grad_data=iso2grad_data,
            iso2grad_settings=iso2grad_settings
        )
        iso2grad_model.fit()

        # Validate results
        golden_results = np.loadtxt(TestPaths.ISO2GRAD_TEST_RESULTS_PATH, delimiter=",")
        assert (iso2grad_model.results.gradient_retention_times == golden_results).all()


if __name__ == "__main__":
    pytest.main()
