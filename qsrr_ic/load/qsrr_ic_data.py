from numpy import ndarray


class QsrrIcData:
    def __init__(
        self,
        molecular_descriptors_for_qsrr_training: ndarray,
        isocratic_retention: ndarray,
        molecular_descriptors_for_iso2grad: ndarray,
        gradient_void_times: ndarray,
        gradient_profiles: ndarray,
        gradient_retention: ndarray
    ):
        self.molecular_descriptors_for_qsrr_training = molecular_descriptors_for_qsrr_training
        self.isocratic_retention = isocratic_retention
        self.molecular_descriptors_for_iso2grad = molecular_descriptors_for_iso2grad
        self.gradient_void_times = gradient_void_times
        self.gradient_profiles = gradient_profiles
        self.gradient_retention = gradient_retention
