from enum import Enum
from typing import (
    Any,
    Dict
)

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from xgboost import XGBRegressor


class RegressorType(Enum):
    xGB = 1
    GBR = 2
    RFR = 3
    ADA = 4
    PLS = 5


REGRESSOR_MAPPING: Dict[RegressorType, Any] = {
    RegressorType.xGB: XGBRegressor,
    RegressorType.GBR: GradientBoostingRegressor,
    RegressorType.RFR: RandomForestRegressor,
    RegressorType.ADA: AdaBoostRegressor,
    RegressorType.PLS: PLSRegression
}

"""
VALID_HYPER_PARAMETER_NAME_MAPPING: Dict[RegressorType, Tuple[HyperParameterName, ...]] = {
    RegressorType.PLS:
        (HyperParameterName.N_COMPONENTS, ),
    RegressorType.xGB:
        (HyperParameterName.N_ESTIMATORS, HyperParameterName.MAX_DEPTH, HyperParameterName.LEARNING_RATE),
    RegressorType.GBR:
        (HyperParameterName.N_ESTIMATORS, HyperParameterName.MAX_DEPTH, HyperParameterName.LEARNING_RATE),
    RegressorType.ADA:
        (HyperParameterName.N_ESTIMATORS, HyperParameterName.LEARNING_RATE),
    RegressorType.RFR:
        (HyperParameterName.N_ESTIMATORS, HyperParameterName.MAX_DEPTH, HyperParameterName.MIN_SAMPLES_LEAF)
}
"""
