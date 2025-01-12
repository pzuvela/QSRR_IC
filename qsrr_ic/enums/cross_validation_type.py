from enum import Enum

from sklearn.model_selection import (
    KFold,
    LeaveOneOut
)

class CrossValidationType(Enum):
    KFold = KFold
    LeaveOneOut = LeaveOneOut
