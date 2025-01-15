from typing import (
    Optional,
    Union
)

import numpy as np
from numpy import ndarray
from numpy.random import permutation

from itertools import permutations

from qsrr_ic.analysis.srd.domain_models import SrdSettings
from qsrr_ic.analysis.srd.enums import (
    GoldenReference,
    GOLDEN_REFERENCE_FUN_MAPPING
)


EXACT_ROW_LIMIT: int = 10


class SumOfRankingDifferences:
    """
    Class for performing Sum of Ranking Differences (SRD) analysis.

    This class implements the SRD method, which is a statistical approach for comparing models or methods
    based on their rankings. The SRD can be validated through comparison of ranks with random numbers (CRNN).

    References:
        1) Heberger, K. Sum of ranking differences compares methods or models fairly. TRAC 2010, 29, 101-109.
           (doi:10.1016/j.trac.2009.09.009)
        2) Heberger, K.; Kollar-Hunek, K. Sum of ranking differences for method discrimination and its validation:
           comparison of ranks with random numbers. J. Chemom. 2011, 25, 151-158.
           (doi:10.1002/cem.1320)
        3) Kalivas, J. H.; Heberger, K.; Andries, E. Sum of ranking differences (SRD) to ensemble multivariate
           calibration model merits for tuning parameter selection and comparing calibration methods, Anal. Chim. Acta
           2015, 869, 21-33.
           (doi:10.1016/j.aca.2014.12.056)

    """

    def __init__(
        self,
        inputs: ndarray,
        golden_reference: Union[ndarray, GoldenReference] = GoldenReference.Mean,
        settings: Optional[SrdSettings] = None
    ):
        """
        Initializes the SRD analysis class.

        Args:
            inputs (ndarray): Input data where columns represent models or methods and rows represent samples.
            golden_reference (Union[ndarray, GoldenReference]): The golden reference for SRD analysis.
                Defaults to GoldenReference.Mean.
            settings (Optional[SrdSettings]): Configuration settings for SRD analysis.
                If None, default settings will be used.

        Raises:
            TypeError: If the golden reference is neither an ndarray nor a GoldenReference.
        """
        self.inputs = inputs

        self.__n_rows = self.inputs.shape[0]

        self._validate_golden(golden_reference)
        self._set_golden(golden_reference)

        self.__settings = settings or self._initialize_settings()

        self.__golden_ranking: Optional[ndarray] = None
        self.__ideal_ranking: Optional[ndarray] = None
        self.__srd_max: Optional[int] = None
        self.__srds: Optional[ndarray] = None
        self.__normalized_srds: Optional[ndarray] = None
        self.__random_srds: Optional[ndarray] = None
        self.__normalized_random_srds: Optional[ndarray] = None

    @staticmethod
    def get_property_value_error(property_name: str) -> "ValueError":
        """
        Constructs a ValueError for read-only properties.

        Args:
            property_name (str): The name of the property.

        Returns:
            ValueError: Error indicating the property is read-only.
        """
        return ValueError(f"{property_name} is read-only and cannot be set directly.")

    @property
    def golden_reference(self) -> ndarray:
        """
        Retrieves the golden reference.

        Returns:
            ndarray: The golden reference used for SRD analysis.
        """
        return self.__golden_reference

    @golden_reference.setter
    def golden_reference(self, value: ndarray) -> None:
        raise self.get_property_value_error("Golden Reference")

    @property
    def golden_ranking(self) -> ndarray:
        """
        Retrieves the golden ranking (sorted indices of the golden reference).

        Returns:
            ndarray: The golden ranking.
        """
        if self.__golden_ranking is None:
            self.__golden_ranking = np.argsort(self.golden_reference)
        return self.__golden_ranking

    @golden_ranking.setter
    def golden_ranking(self, value: ndarray) -> None:
        raise self.get_property_value_error("Golden Ranking")

    @property
    def ideal_ranking(self) -> ndarray:
        """
        Retrieves the ideal ranking.

        Returns:
            ndarray: The ideal ranking as a column vector.
        """
        if self.__ideal_ranking is None:
            self.__ideal_ranking = np.arange(self.__n_rows).reshape(-1, 1)
        return self.__ideal_ranking

    @ideal_ranking.setter
    def ideal_ranking(self, value: ndarray) -> None:
        raise self.get_property_value_error("Ideal Ranking")

    @property
    def srd_max(self) -> float:
        """
        Retrieves the maximum possible SRD value.

        Returns:
            float: The maximum SRD value based on the number of rows in the input.
        """
        if self.__srd_max is None:
            if self.__n_rows % 2 == 1:
                k = (self.__n_rows - 1) / 2
                self.__srd_max = 2 * k * (k + 1)
            else:
                k = self.__n_rows / 2
                self.__srd_max = 2 * (k ** 2)
        return self.__srd_max

    @srd_max.setter
    def srd_max(self, value: float) -> None:
        raise self.get_property_value_error("SRD(max)")

    @property
    def srds(self) -> ndarray:
        """
        Computes or retrieves the SRD values.

        Returns:
            ndarray: The SRD values for each column in the input data.
        """
        if self.__srds is None:
            self.__srds = self._compute_srd()
        return self.__srds

    @srds.setter
    def srds(self, value: ndarray) -> None:
        raise self.get_property_value_error("SRD values")

    @property
    def normalized_srds(self) -> ndarray:
        """
        Computes or retrieves the normalized SRD values.

        Returns:
            ndarray: Normalized SRD values.
        """
        if self.__normalized_srds is None:
            self.__normalized_srds = self._normalize_srd(self.srds)
        return self.__normalized_srds

    @normalized_srds.setter
    def normalized_srds(self, value: ndarray) -> None:
        raise self.get_property_value_error("Normalized SRD values")

    @property
    def random_srds(self):
        """
        Validates SRD using CRNN and retrieves random SRD values.

        Returns:
            ndarray: Random SRD values.
        """
        if self.__random_srds is None:
            self.__random_srds = self._validate_using_crnn()
        return self.__random_srds

    @random_srds.setter
    def random_srds(self, value: ndarray) -> None:
        raise self.get_property_value_error("Random SRD values")

    @property
    def normalized_random_srds(self):
        """
        Computes or retrieves normalized random SRD values.

        Returns:
            ndarray: Normalized random SRD values.
        """
        if self.__normalized_random_srds is None:
            self.__normalized_random_srds = self._normalize_srd(self.random_srds)
        return self.__normalized_random_srds

    @normalized_random_srds.setter
    def normalized_random_srds(self, value: ndarray) -> None:
        raise self.get_property_value_error("Normalized Random SRD values")

    @staticmethod
    def _validate_golden(golden_reference):
        """
        Validates the type of the golden reference.

        Args:
            golden_reference: The golden reference to validate.

        Raises:
            TypeError: If the golden reference is not an ndarray or GoldenReference.
        """
        if not isinstance(golden_reference, (ndarray, GoldenReference)):
            raise TypeError("golden_reference must be of type ndarray or GoldenReference.")

    def _set_golden(self, golden_reference: Union[ndarray, GoldenReference]):
        """
        Sets the golden reference based on the input type.

        Args:
            golden_reference (Union[ndarray, GoldenReference]): The golden reference to set.
        """
        self.__golden_reference = golden_reference
        if isinstance(golden_reference, GoldenReference):
            self.__golden_reference: Optional[ndarray] = \
                GOLDEN_REFERENCE_FUN_MAPPING[golden_reference](self.inputs, axis=0)

    @staticmethod
    def _initialize_settings() -> SrdSettings:
        """
        Initializes default settings for SRD analysis.
        """
        return SrdSettings()

    def _compute_srd(self) -> ndarray:
        """
        Computes the SRD values using vectorized operations.

        Returns:
            ndarray: Computed SRD values for each column in the input data.
        """
        final_ranking = np.argsort(np.argsort(self.inputs, axis=0), axis=0)
        return np.sum(np.abs(final_ranking - self.ideal_ranking), axis=0)

    def _normalize_srd(self, srds: ndarray) -> ndarray:
        """
        Normalizes the SRD values to a percentage scale.

        Args:
            srds (ndarray): The SRD values to normalize.

        Returns:
            ndarray: Normalized SRD values.
        """
        return srds * (100 / self.srd_max)

    def _validate_using_crnn(self) -> ndarray:
        """
        Validates SRD using comparison of ranks with random numbers (CRNN).

        Returns:
            ndarray: Random SRD values generated for validation.
        """
        exact = self.__settings.exact and self.__n_rows <= EXACT_ROW_LIMIT

        if not exact:
            random_ranking = np.hstack(
                [permutation(self.__n_rows).reshape(-1, 1) for _ in range(self.__settings.n_iterations)]
            )
        else:
            random_ranking = np.array(list(permutations(range(self.__n_rows)))).T

        return np.sum(np.abs(random_ranking - self.ideal_ranking), axis=0)
