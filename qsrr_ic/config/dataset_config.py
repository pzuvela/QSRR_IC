from typing import (
    Any,
    Dict
)

from qsrr_ic.base import BaseConfig


class DatasetConfig(BaseConfig):

    """
    A container for dataset file paths required by QsrrIcDataset.
    """

    def __init__(
        self,
        molecular_descriptors_for_qsrr_training_path: str,
        isocratic_retention_path: str,
        molecular_descriptors_for_iso2grad_path: str,
        gradient_void_times_path: str,
        gradient_profiles_path: str,
        gradient_retention_path: str
    ):
        self.molecular_descriptors_for_qsrr_training_path = molecular_descriptors_for_qsrr_training_path
        self.isocratic_retention_path = isocratic_retention_path
        self.molecular_descriptors_for_iso2grad_path = molecular_descriptors_for_iso2grad_path
        self.gradient_void_times_path = gradient_void_times_path
        self.gradient_profiles_path = gradient_profiles_path
        self.gradient_retention_path = gradient_retention_path

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, dataset_dict: Dict[str, Any]) -> 'DatasetConfig':
        return cls(**dataset_dict)
