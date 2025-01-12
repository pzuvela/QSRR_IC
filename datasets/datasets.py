import os

from qsrr_ic.config import DatasetConfig
from qsrr_ic.load import (
    load_dataset,
    QsrrIcDataset
)

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "qsrr_ic")

QSRR_IC_DATASET_PATHS = DatasetConfig(
    molecular_descriptors_for_qsrr_training_path=os.path.join(
        DATASETS_DIR,
        "2025-01-05-molecular_descriptors_for_qsrr_training.csv"
    ),
    isocratic_retention_path=os.path.join(
        DATASETS_DIR,
        "2025-01-05-isocratic_retention.csv"
    ),
    molecular_descriptors_for_iso2grad_path=os.path.join(
        DATASETS_DIR,
        "2025-01-05-molecular_descriptors_for_iso2grad.csv"
    ),
    gradient_void_times_path=os.path.join(
        DATASETS_DIR,
        "2025-01-05-gradient_void_times.csv"
    ),
    gradient_profiles_path=os.path.join(
        DATASETS_DIR,
        "2025-01-05-gradient_profiles.csv"
    ),
    gradient_retention_path=os.path.join(
        DATASETS_DIR,
        "2025-01-05-gradient_retention.csv"
    )
)

def load_qsrr_ic_dataset() -> QsrrIcDataset:
    return load_dataset(QSRR_IC_DATASET_PATHS)
