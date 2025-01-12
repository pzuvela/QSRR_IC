import os
from typing import Optional

import pandas as pd
from pandas import DataFrame


from qsrr_ic.config import DatasetConfig


class QsrrIcDataset:
    """
    Represents a dataset used for QSRR and Iso2Grad modeling, with functionality for loading and validating data files.
    """
    def __init__(self, dataset_paths: DatasetConfig):
        self._paths = dataset_paths
        self._validate_paths()

        # DataFrame placeholders
        self.molecular_descriptors_for_qsrr_training_df: Optional[DataFrame] = None
        self.isocratic_retention_df: Optional[DataFrame] = None
        self.molecular_descriptors_for_iso2grad_df: Optional[DataFrame] = None
        self.gradient_void_times_df: Optional[DataFrame] = None
        self.gradient_profiles_df: Optional[DataFrame] = None
        self.gradient_retention_df: Optional[DataFrame] = None

    def _validate_paths(self):
        """
        Validate that all file paths exist.
        """
        missing_files = [
            attr for attr, path in self._paths.__dict__.items() if not os.path.exists(path)
        ]
        if missing_files:
            missing = ", ".join(missing_files)
            raise FileNotFoundError(f"Missing dataset files: {missing}")

    @staticmethod
    def _load_file(file_path: str) -> DataFrame:
        """
        Load a CSV file into a DataFrame.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            DataFrame: Loaded DataFrame.
        """
        return pd.read_csv(file_path, float_precision="round_trip")

    def load(self):
        """
        Load all dataset files into DataFrames.
        """
        self.molecular_descriptors_for_qsrr_training_df = self._load_file(
            self._paths.molecular_descriptors_for_qsrr_training_path
        )
        self.isocratic_retention_df = self._load_file(
            self._paths.isocratic_retention_path
        )
        self.molecular_descriptors_for_iso2grad_df = self._load_file(
            self._paths.molecular_descriptors_for_iso2grad_path
        )
        self.gradient_void_times_df = self._load_file(
            self._paths.gradient_void_times_path
        )
        self.gradient_profiles_df = self._load_file(
            self._paths.gradient_profiles_path
        )
        self.gradient_retention_df = self._load_file(
            self._paths.gradient_retention_path
        )
        return self


def load_dataset(dataset_paths: DatasetConfig) -> QsrrIcDataset:
    """
    Load the QsrrIc dataset.

    Returns:
        QsrrIcDataset: Loaded dataset instance.
    """
    dataset = QsrrIcDataset(dataset_paths)
    dataset.load()
    return dataset
