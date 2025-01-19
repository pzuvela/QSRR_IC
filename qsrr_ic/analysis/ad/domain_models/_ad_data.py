from dataclasses import dataclass
from typing import Optional

from qsrr_ic.models.qsrr.domain_models import QsrrData


@dataclass
class ApplicabilityDomainData:
    """
    Data class for encapsulating input data and related parameters for WilliamsModel.
    """

    data_train: QsrrData  # Training data (descriptors, retention)
    data_test: Optional[QsrrData]  # Test data (descriptors, retention), optional
    predictions_train: QsrrData  # Training predictions
    predictions_test: Optional[QsrrData]  # Test predictions
