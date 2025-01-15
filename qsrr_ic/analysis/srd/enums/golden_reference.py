from enum import Enum
from typing import (
    Callable,
    Dict
)

import numpy as np

from scipy import stats


class GoldenReference(Enum):
    Mean = 1
    Median = 2
    Mode = 3
    Min = 4
    Max = 5


GOLDEN_REFERENCE_FUN_MAPPING: Dict[GoldenReference, Callable] = {
    GoldenReference.Mean: np.mean,
    GoldenReference.Median: np.median,
    GoldenReference.Mode: stats.mode,
    GoldenReference.Min: np.min,
    GoldenReference.Max: np.max
}
