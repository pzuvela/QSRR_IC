class Iso2GradSettings:
    def __init__(
        self,
        integration_step: float = 0.01,
        n_jobs: int = -1,
        verbosity: int = 10
    ):
        """
        Settings for Iso2Grad model.

        Parameters
        ----------
        integration_step: float
            Step size for numerical integration. Default is 0.01.
        """
        if integration_step <= 0:
            raise ValueError("integration_step must be positive.")
        self.integration_step = integration_step

        if n_jobs < -1 or n_jobs == 0:
            raise ValueError("n_jobs can be either -1 or a positive number")
        self.n_jobs = n_jobs

        if verbosity < 0:
            raise ValueError("verbosity cannot be negative!")
        self.verbosity = verbosity
