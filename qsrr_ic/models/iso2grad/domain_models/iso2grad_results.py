from numpy import ndarray


class Iso2GradResults:
    def __init__(
        self,
        gradient_retention_times: ndarray
    ):
        """
        Holds the results of the Iso2Grad model.

        Parameters
        ----------
        gradient_retention_times: ndarray
            Matrix of gradient retention times, [n_profiles, n_analytes].
        """
        self.gradient_retention_times = gradient_retention_times
