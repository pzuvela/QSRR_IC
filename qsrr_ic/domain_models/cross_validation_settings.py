from typing import Optional

from qsrr_ic.enums import CrossValidationType


class CrossValidationSettings:
    def __init__(
        self,
        cv_type: CrossValidationType = CrossValidationType.KFold,
        n_splits: Optional[int] = 3
    ):
        self.cv_type = cv_type

        if self.cv_type == CrossValidationType.KFold:
            if n_splits is None:
                raise ValueError("n_splits cannot be None for CV type KFold!")
            if n_splits < 0:
                raise ValueError("n_splits cannot be negative!")

        self.n_splits = n_splits

