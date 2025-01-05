import os
from typing import Optional

import pandas as pd
from pandas import DataFrame


class QsrrIcDataset:

    def __init__(
        self,
        molecular_descriptors_for_qsrr_training_path: str,
        isocratic_retention_path: str,
        molecular_descriptors_for_iso2grad_path: str,
        gradient_void_times_path: str,
        gradient_profiles_path: str,
        gradient_retention_path: str
    ):
        self.__molecular_descriptors_for_qsrr_training_path = molecular_descriptors_for_qsrr_training_path
        self.__isocratic_retention_path = isocratic_retention_path
        self.__molecular_descriptors_for_iso2grad_path = molecular_descriptors_for_iso2grad_path
        self.__gradient_void_times_path = gradient_void_times_path
        self.__gradient_profiles_path = gradient_profiles_path
        self.__gradient_retention_path = gradient_retention_path

        self._validate()

        self.molecular_descriptors_for_qsrr_training_df: Optional[DataFrame] = None
        self.isocratic_retention_df: Optional[DataFrame] = None
        self.molecular_descriptors_for_iso2grad_df: Optional[DataFrame] = None
        self.gradient_void_times_df: Optional[DataFrame] = None
        self.gradient_profiles_df: Optional[DataFrame] = None
        self.gradient_retention_df: Optional[DataFrame] = None

    def _validate(self):

        missing_files = []

        # Validate the existence of all file paths
        for key, value in self.__dict__.items():
            if "_path" in key:
                if not os.path.exists(value):
                    missing_files.append(key)

        if missing_files:
            validations = f"The following files are missing: {', '.join(missing_files)}"
            raise FileNotFoundError(validations)

    def load(self):
        self.molecular_descriptors_for_qsrr_training_df: DataFrame = pd.read_csv(
            self.__molecular_descriptors_for_qsrr_training_path,
            float_precision="high"  # round_trip
        )
        self.isocratic_retention_df: DataFrame = pd.read_csv(
            self.__isocratic_retention_path,
            float_precision="high"  # round_trip
        )
        self.molecular_descriptors_for_iso2grad_df: DataFrame = pd.read_csv(
            self.__molecular_descriptors_for_iso2grad_path,
            float_precision="high"  # round_trip
        )
        self.gradient_void_times_df: DataFrame = pd.read_csv(
            self.__gradient_void_times_path,
            float_precision="high"  # round_trip
        )
        self.gradient_profiles_df: DataFrame = pd.read_csv(
            self.__gradient_profiles_path,
            float_precision="high"  # round_trip
        )
        self.gradient_retention_df: DataFrame = pd.read_csv(
            self.__gradient_retention_path,
            float_precision="high"  # round_trip
        )


def load_dataset() -> QsrrIcDataset:

    dataset_path = os.path.join(os.path.dirname(__file__), "qsrr_ic")

    dataset = QsrrIcDataset(
        molecular_descriptors_for_qsrr_training_path=os.path.join(
            dataset_path,
            "2025-01-05-molecular_descriptors_for_qsrr_training.csv"
        ),
        isocratic_retention_path=os.path.join(dataset_path, "2025-01-05-isocratic_retention.csv"),
        molecular_descriptors_for_iso2grad_path=os.path.join(
            dataset_path,
            "2025-01-05-molecular_descriptors_for_iso2grad.csv"
        ),
        gradient_void_times_path=os.path.join(dataset_path, "2025-01-05-gradient_void_times.csv"),
        gradient_profiles_path=os.path.join(dataset_path, "2025-01-05-gradient_profiles.csv"),
        gradient_retention_path=os.path.join(dataset_path, "2025-01-05-gradient_retention.csv"),
    )

    dataset.load()

    return dataset
