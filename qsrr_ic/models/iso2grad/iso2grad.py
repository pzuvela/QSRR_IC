"""

ISO2GRAD model v1

Function to predict gradient retention times from an isocratic model

Reference: Bolanca, T. et al. Development of an ion chromatographic gradient retention model from isocratic elution
           experiments. J. Chromatogr. A 2006, 1121, 228-235. (doi:10.1016/j.chroma.2006.04.036)

"""

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

from qsrr_ic.models.iso2grad.domain_models import (
    Iso2GradData,
    Iso2GradResults,
    Iso2GradSettings
)


class Iso2Grad:

    def __init__(
        self,
        qsrr_model: Any,
        scaler: Any,
        iso2grad_data: Iso2GradData,
        iso2grad_settings: Iso2GradSettings
    ):

        """

        Class to fit the Iso2Grad model

        Parameters
        ----------
        qsrr_model: Any (sklearn object), Isocratic QSRR model used for predictions
        scaler: Any (sklearn Scaler object), Scaler used to descale the QSRR predictions
        iso2grad_data: Iso2GradData, data for fitting the iso2grad model
        iso2grad_settings: Iso2GradSettings, settings for fitting the iso2grad model
        """

        self.qsrr_model = qsrr_model
        self.scaler = scaler
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
        scaled_input = self.scaler.transform(input_data)
        log_k = self.qsrr_model.predict(scaled_input)
        return 10 ** log_k

    def _fit(self, profile_idx: int):

        n_profiles, n_analytes = self.iso2grad_data.gradient_void_times.shape
        n_profile_cols = self.iso2grad_data.gradient_retention_profiles.shape[1]

        tg1, tg2, i_prev, i_partial, k1_2 = np.zeros((5, 1))
        tr_g = np.zeros(n_analytes)

        # Loop through the analytes
        for b in range(n_analytes):

            # Initialize next integration step
            i_next = 0

            # Loop through the gradient segments
            for p in range(0, n_profile_cols - 2, 3):

                # First segment of the gradient
                ti1 = self.iso2grad_data.gradient_retention_profiles[profile_idx, p]
                conc1 = self.iso2grad_data.gradient_retention_profiles[profile_idx, p + 1]
                slope = self.iso2grad_data.gradient_retention_profiles[profile_idx, p + 2]

                # Second segment of the gradient
                ti2 = self.iso2grad_data.gradient_retention_profiles[profile_idx, p + 3]
                conc2 = self.iso2grad_data.gradient_retention_profiles[profile_idx, p + 4]

                # Loop through the retention times
                for tg in np.arange(ti1, ti2, self.iso2grad_settings.integration_step):

                    tg1 = tg
                    tg2 = tg1 + self.iso2grad_settings.integration_step

                    if tg2 < ti2:

                        conc_grad1 = slope * (tg1 - ti1) + conc1
                        conc_grad2 = slope * (tg2 - ti1) + conc1

                        # Update the integration step
                        del_t_updt = self.iso2grad_settings.integration_step

                    else:

                        conc_grad1 = slope * (tg1 - ti1) + conc1
                        conc_grad2 = conc2

                        # Update the integration step
                        del_t_updt = ti2 - tg1

                    # Re-define the next integration step as the previous one
                    i_prev = i_next

                    # Predict k from the machine learning model for the two gradient concentrations
                    analyte_predictors = self.iso2grad_data.isocratic_model_predictors[b]
                    k1 = self._predict_k(concentration=conc_grad1, predictors=analyte_predictors)
                    k2 = self._predict_k(concentration=conc_grad2, predictors=analyte_predictors)

                    # Average k between the two gradient concentrations
                    k1_2 = (k2 + k1) / 2

                    # Update integral
                    i_partial = del_t_updt / k1_2
                    i_next = i_prev + i_partial

                    if i_prev < self.iso2grad_data.gradient_void_times[profile_idx, b] < i_next:
                        break

                if i_prev < self.iso2grad_data.gradient_void_times[profile_idx, b] < i_next:
                    break

            # Calculate retention time for a specified gradient
            tr_g[b] = \
                self.iso2grad_data.gradient_void_times[profile_idx, b] + tg1 \
                + (self.iso2grad_data.gradient_void_times[profile_idx, b] - i_prev) * k1_2

        return tr_g

    def fit(self):

        pool = Parallel(
            n_jobs=self.iso2grad_settings.n_jobs,
            verbose=self.iso2grad_settings.verbosity
        )

        gradient_retention_times = pool(
            delayed(self._fit)(profile_idx)
            for profile_idx in range(self.iso2grad_data.gradient_retention_profiles.shape[0])
        )

        gradient_retention_times = np.vstack(gradient_retention_times)

        self.results = Iso2GradResults(gradient_retention_times)
