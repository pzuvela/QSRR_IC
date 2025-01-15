from typing import (
    Any,
    Dict,
    Optional
)

VALIDATE_USING_CRNN: bool = True
N_ITERATIONS: int = 10000
EXACT: bool = False


class SrdSettings:
    def __init__(
        self,
        validate_using_crnn: bool = VALIDATE_USING_CRNN,
        n_iterations: int = N_ITERATIONS,
        exact: bool = EXACT
    ):

        self.validate_using_crnn = validate_using_crnn
        self.n_iterations = n_iterations
        self.exact = exact

    @classmethod
    def from_dict(cls, srd_settings_dict: Optional[Dict[str, Any]]) -> "SrdSettings":
        if srd_settings_dict is None:
            srd_settings_dict = {
                "validate_using_crnn": VALIDATE_USING_CRNN,
                "n_iterations": N_ITERATIONS,
                "exact": EXACT
            }
        return cls(
            srd_settings_dict.get("validate_using_crnn", VALIDATE_USING_CRNN),
            srd_settings_dict.get("n_iterations", N_ITERATIONS),
            srd_settings_dict.get("exact", EXACT)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validate_using_crnn": self.validate_using_crnn,
            "n_iterations": self.n_iterations,
            "exact": self.exact
        }
