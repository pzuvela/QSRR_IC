from enum import Enum
from typing import (
    Dict,
    Type
)


class HyperParameterName(Enum):
    N_ESTIMATORS = "n_estimators"
    LEARNING_RATE = "learning_rate"
    MAX_DEPTH = "max_depth"
    MIN_SAMPLES_LEAF = "min_samples_leaf"
    N_COMPONENTS = "n_components"


HYPER_PARAMETER_TYPE_MAPPING: Dict[HyperParameterName, Type] = {
    HyperParameterName.N_ESTIMATORS: int,
    HyperParameterName.LEARNING_RATE: float,
    HyperParameterName.MAX_DEPTH: int,
    HyperParameterName.MIN_SAMPLES_LEAF: int,
    HyperParameterName.N_COMPONENTS: int
}
