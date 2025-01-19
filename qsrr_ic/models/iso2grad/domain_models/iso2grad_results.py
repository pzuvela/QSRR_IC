from typing import Optional

from numpy import ndarray

from qsrr_ic.models.qsrr.domain_models import QsrrMetrics


class Iso2GradResults:
    def __init__(
        self,
        gradient_retention_times: ndarray,
        metrics: Optional[QsrrMetrics] = None
    ):
        """
        Holds the results of the Iso2Grad model.

        Parameters
        ----------
        gradient_retention_times: ndarray,
            Matrix of gradient retention times, [n_profiles, n_analytes].
        metrics: Optional[QsrrMetrics], optional
            Metrics like R2, RMSE, ...
        """
        self.gradient_retention_times = gradient_retention_times
        self.metrics = metrics
