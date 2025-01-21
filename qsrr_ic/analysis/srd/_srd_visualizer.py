from dataclasses import dataclass

import numpy as np
from numpy import ndarray

from scipy.stats import norm

from matplotlib import pyplot as plt

from qsrr_ic.analysis.srd import SumOfRankingDifferences
from qsrr_ic.error_handling import ErrorHandling


class SrdVisualizerConstants:
    N_POINTS: int = 100


@dataclass
class SrdVisualizerData:
    normalized_srds: ndarray
    normalized_random_srds: ndarray
    mu: float
    std: float
    x_values: ndarray
    pdf_values: ndarray


class SRDModel:
    def __init__(
        self,
        normalized_srds: ndarray,
        normalized_random_srds: ndarray
    ):
        self._normalized_srds: ndarray = normalized_srds
        self._normalized_random_srds: ndarray = normalized_random_srds
        self._mu = None
        self._std = None
        self._fit_done = False  # Flag to track if fit has been computed

    def _fit_distribution(self):
        """Performs the fitting and caches the results."""
        if not self._fit_done:
            self._mu, self._std = norm.fit(self.normalized_random_srds)
            self._fit_done = True

    @property
    def normalized_random_srds(self) -> ndarray:
        return self._normalized_random_srds

    @normalized_random_srds.setter
    def normalized_random_srds(self, value: ndarray) -> None:
        raise ErrorHandling.get_property_value_error("normalized_srds")

    @property
    def normalized_srds(self) -> ndarray:
        return self._normalized_srds

    @normalized_srds.setter
    def normalized_srds(self, value: ndarray) -> None:
        raise ErrorHandling.get_property_value_error("normalized_srds")

    @property
    def mu(self):
        if self._mu is None:
            self._fit_distribution()
        return self._mu

    @mu.setter
    def mu(self, value) -> None:
        raise ErrorHandling.get_property_value_error("mu")

    @property
    def std(self):
        if self._std is None:
            self._fit_distribution()
        return self._std

    @std.setter
    def std(self, value) -> None:
        raise ErrorHandling.get_property_value_error("std")

    def calculate_pdf(self, x_values):
        return norm.pdf(x_values, self.mu, self.std)


class SRDViewModel:
    def __init__(self, model):
        self.model = model

    def prepare_data_for_plotting(self) -> SrdVisualizerData:
        x_values = np.linspace(
            self.model.normalized_data.min(),
            self.model.normalized_data.max(),
            SrdVisualizerConstants.N_POINTS
        )
        pdf_values = self.model.calculate_pdf(x_values)
        return SrdVisualizerData(
            normalized_srds=self.model.normalized_srds,
            normalized_random_srds=self.model.normalized_random_srds,
            mu=self.model.mu,
            std=self.model.std,
            x_values=x_values,
            pdf_values=pdf_values
        )

class SRDView:

    @staticmethod
    def plot(data: SrdVisualizerData):

        fig, ax1 = plt.subplots(figsize=(8, 6))

        # Plot bar chart
        ax1.bar(
            data.normalized_srds,
            data.normalized_srds,
            color='blue',
            alpha=0.7,
            label="Normalized SRD / %"
        )
        ax1.set_xlabel("Normalized SRD / %", fontsize=12)
        ax1.set_ylabel("Normalized SRD / %", fontsize=12, color="blue")
        ax1.tick_params(axis='y', labelcolor="blue")

        # Add second y-axis for normal distribution
        ax2 = ax1.twinx()
        ax2.plot(data.x_values, data.pdf_values, color='red', linestyle='--')
        ax2.set_ylabel(
            "Relative Frequency of SRDs of Random Numbers / %",
            fontsize=12,
            color="red"
        )
        ax2.tick_params(axis='y', labelcolor="red")

        # Add vertical lines
        p_min = data.pdf_values.min()
        p_max = data.pdf_values.max() + 0.05 * data.pdf_values.max()
        ax2.vlines(data.mu - 2 * data.std, p_min, p_max, color="lightgrey", linestyle="-")
        ax2.vlines(data.mu, p_min, p_max, color="lightgrey", linestyle="--")
        ax2.vlines(data.mu + 2 * data.std, p_min, p_max, color="lightgrey", linestyle="-")
        ax2.set_ylim(p_min, p_max)

        ax1.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()
        plt.show()


class SrdVisualizer:

    def __init__(
        self,
        srd: SumOfRankingDifferences
    ):
        self._srd = srd

    def visualize(self):

        model = SRDModel(
            normalized_srds=self._srd.normalized_srds,
            normalized_random_srds=self._srd.normalized_random_srds
        )
        view_model = SRDViewModel(model)

        data_: SrdVisualizerData = view_model.prepare_data_for_plotting()

        SRDView.plot(data=data_)
