from typing import (
    Any,
    Optional
)

import numpy as np
from numpy import ndarray

from joblib import (
    delayed,
    Parallel
)

from qsrr_ic.domain_models import Iso2GradSettings
from qsrr_ic.models.iso2grad.domain_models import (
    Iso2GradData,
    Iso2GradResults
)
from qsrr_ic.models.qsrr.domain_models import (
    QsrrData,
    QsrrMetrics
)


class Iso2Grad:
    def __init__(
        self,
        qsrr_model: Any,
        iso2grad_data: Iso2GradData,
        iso2grad_settings: Iso2GradSettings
    ):
        """
        Class to fit the Iso2Grad model

        Parameters
        ----------
        qsrr_model: Any (sklearn object), Isocratic QSRR model used for predictions
        iso2grad_data: Iso2GradData, data for fitting the iso2grad model
        iso2grad_settings: Iso2GradSettings, settings for fitting the iso2grad model
        """
        self.qsrr_model = qsrr_model
        self.iso2grad_data = iso2grad_data
        self.iso2grad_settings = iso2grad_settings
        self.results: Optional[Iso2GradResults] = None

    def _predict_k(self, concentration: float, predictors: ndarray) -> float:
        """
        Predicts the retention factor (k) using the QSRR model.

        Parameters
        ----------
        concentration: float, Gradient concentration at a specific time.
        predictors: ndarray, Predictors for the analyte.

        Returns
        -------
        float, Predicted retention factor
        """
        input_data = np.hstack((concentration, predictors)).reshape(1, -1)
        scaled_input = self.qsrr_model.scaler.transform(input_data)
        log_k = self.qsrr_model.model.predict(scaled_input)
        return 10 ** log_k

    @staticmethod
    def _calculate_concentrations(tg1: float, tg2: float, ti1: float, conc1: float, slope: float, ti2: float, conc2: float) -> tuple:
        """
        Calculate the gradient concentrations for the two retention times.

        Parameters
        ----------
        tg1, tg2: float, Retention times for the two gradient steps.
        ti1, ti2: float, Gradient segment start and end times.
        conc1, conc2: float, Gradient concentrations at the start and end times.
        slope: float, Slope of the gradient.

        Returns
        -------
        tuple, Calculated concentrations at tg1 and tg2.
        """
        if tg2 < ti2:
            conc_grad1 = slope * (tg1 - ti1) + conc1
            conc_grad2 = slope * (tg2 - ti1) + conc1
        else:
            conc_grad1 = slope * (tg1 - ti1) + conc1
            conc_grad2 = conc2
        return conc_grad1, conc_grad2

    def _calculate_average_k(self, conc_grad1: float, conc_grad2: float, analyte_idx: int) -> float:
        """
        Calculate the average retention factor (k) between two concentrations.

        Parameters
        ----------
        conc_grad1, conc_grad2: float, The two gradient concentrations.
        analyte_idx: int, Index of the analyte.

        Returns
        -------
        float, Average retention factor (k).
        """
        analyte_predictors = self.iso2grad_data.isocratic_model_predictors[analyte_idx]
        k1 = self._predict_k(concentration=conc_grad1, predictors=analyte_predictors)
        k2 = self._predict_k(concentration=conc_grad2, predictors=analyte_predictors)
        return (k2 + k1) / 2

    @staticmethod
    def _integrate_retention_time(tg1: float, tg2: float, k_avg: float, i_prev: float) -> float:
        """
        Integrate retention time for a given time step.

        Parameters
        ----------
        tg1, tg2: float, Retention times for the current gradient segment.
        k_avg: float, Average retention factor (k).
        i_prev: float, Previous integration step value.

        Returns
        -------
        float, Updated integration value.
        """
        del_t_updt = tg2 - tg1
        return i_prev + (del_t_updt / k_avg)

    def _process_analyte(self, profile_idx: int, analyte_idx: int) -> float:
        """
        Process retention times for a single analyte.

        Parameters
        ----------
        profile_idx: int, Index of the gradient profile.
        analyte_idx: int, Index of the analyte.

        Returns
        -------
        float, Retention time for the analyte.
        """
        n_profile_cols = self.iso2grad_data.gradient_retention_profiles.shape[1]
        i_prev = 0.0
        tr_g = 0.0

        # Loop through the gradient segments
        for p in range(0, n_profile_cols - 2, 3):

            ti1, conc1, slope = self.iso2grad_data.gradient_retention_profiles[profile_idx, p:p+3]
            ti2, conc2 = self.iso2grad_data.gradient_retention_profiles[profile_idx, p+3:p+5]

            for tg in np.arange(ti1, ti2, self.iso2grad_settings.integration_step):

                print(f"Computing for tg = {tg}")

                tg1, tg2 = tg, tg + self.iso2grad_settings.integration_step

                # Calculate gradient concentrations at tg1 and tg2
                conc_grad1, conc_grad2 = self._calculate_concentrations(tg1, tg2, ti1, conc1, slope, ti2, conc2)

                # Calculate the average retention factor (k)
                k_avg = self._calculate_average_k(conc_grad1, conc_grad2, analyte_idx)

                # Integrate retention time
                i_next = self._integrate_retention_time(tg1, tg2, k_avg, i_prev)

                # Check if the analyte passes the void time
                if i_prev < self.iso2grad_data.gradient_void_times[profile_idx, analyte_idx] < i_next:
                    tr_g = self.iso2grad_data.gradient_void_times[profile_idx, analyte_idx] + tg1 + \
                           (self.iso2grad_data.gradient_void_times[profile_idx, analyte_idx] - i_prev) * k_avg
                    break

                i_prev = i_next

            if tr_g != 0.0:
                break

        return tr_g

    def _fit_profile(self, profile_idx: int) -> np.ndarray:
        """
        Process and calculate retention times for all analytes in a profile.

        Parameters
        ----------
        profile_idx: int, Index of the gradient profile.

        Returns
        -------
        np.ndarray, Retention times for all analytes in the profile.
        """
        n_analytes = self.iso2grad_data.gradient_void_times.shape[1]
        retention_times = np.zeros(n_analytes)

        for analyte_idx in range(n_analytes):
            print(f"Fitting iso2grad for analyze {analyte_idx + 1}...")
            retention_times[analyte_idx] = self._process_analyte(profile_idx, analyte_idx)

        return retention_times

    def fit(self):
        """
        Fit the Iso2Grad model in parallel and return gradient retention times.
        """
        pool = Parallel(n_jobs=self.iso2grad_settings.n_jobs, verbose=self.iso2grad_settings.verbosity)
        gradient_retention_times = pool(
            delayed(self._fit_profile)(profile_idx)
            for profile_idx in range(self.iso2grad_data.gradient_retention_profiles.shape[0])
        )

        gradient_retention_times_ = np.hstack(gradient_retention_times)

        metrics = None

        if self.iso2grad_data.gradient_retention_times is not None:
            metrics = QsrrMetrics(
                qsrr_data=QsrrData(y=self.iso2grad_data.gradient_retention_times.ravel(), x=None),
                qsrr_predictions=QsrrData(y=gradient_retention_times_.ravel(), x=None)
            )

        # Stack the results and store in results
        self.results = Iso2GradResults(
            gradient_retention_times=gradient_retention_times_,
            metrics=metrics
        )
