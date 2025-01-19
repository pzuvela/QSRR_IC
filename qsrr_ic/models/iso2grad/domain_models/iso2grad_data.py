from typing import Optional

from numpy import ndarray

from qsrr_ic.models.iso2grad.constants import Iso2GradConstants


class Iso2GradData:
    def __init__(
        self,
        isocratic_model_predictors: ndarray,
        gradient_void_times: ndarray,
        gradient_retention_profiles: ndarray,
        gradient_retention_times: Optional[ndarray] = None
    ):
        """

        Data class holding the data required to fit an iso2grad model

        Parameters
        ----------
        isocratic_model_predictors: ndarray
            Matrix of model predictors, [n_analytes, n_descriptors].
        gradient_void_times: ndarray
            Matrix of gradient void times, [n_profiles, n_analytes].
        gradient_retention_profiles: ndarray
            Matrix of gradient profiles, [n_profiles, n_cols].
            Example columns: [tG_1, c(KOH)_1, slope_1, tG_2, c(KOH)_2, slope_2, ..., tG_n, c(KOH)_n].
            Note: profiles are linear profiles!
        gradient_retention_times: Optional[ndarray], optional,
            Gradient retention times used to validate the iso2grad model & calculate metrics
        """

        self.isocratic_model_predictors = isocratic_model_predictors
        self.gradient_void_times = gradient_void_times
        self.gradient_retention_profiles = gradient_retention_profiles
        self.gradient_retention_times = gradient_retention_times

        self._validate_inputs()

        # Add a small constant to prevent division by zero
        self.gradient_retention_profiles += Iso2GradConstants.EPS

    def _validate_inputs(self) -> None:
        """

        Validates input shapes and consistency

        Returns
        -------
        None

        """
        if len(self.gradient_void_times.shape) != 2:
            raise ValueError("gradient_void_times must be a 2D array.")
        if len(self.isocratic_model_predictors.shape) != 2:
            raise ValueError("isocratic_model_predictors must be a 2D array.")
        if len(self.gradient_retention_profiles.shape) != 2:
            raise ValueError("gradient_retention_profiles must be a 2D array.")
        if self.gradient_void_times.shape[1] != self.isocratic_model_predictors.shape[0]:
            raise ValueError("Mismatch between gradient_void_times columns and isocratic_model_predictors rows.")
