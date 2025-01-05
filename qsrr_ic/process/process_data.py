from typing import Set

from numpy import ndarray
from pandas import DataFrame

from qsrr_ic.constants import QsrrIcConstants
from qsrr_ic.datasets.qsrr_ic_dataset import QsrrIcDataset
from qsrr_ic.domain_models import QsrrIcData


class ProcessData:

    def __init__(self, qsrr_ic_dataset: QsrrIcDataset):
        self.qsrr_ic_dataset: QsrrIcDataset = qsrr_ic_dataset

    @staticmethod
    def _process_data(data: DataFrame, columns_to_remove: Set[str]) -> ndarray:
        missing_columns = columns_to_remove - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # Dropping the columns to remove
        processed_data = data.drop(columns=columns_to_remove, errors="ignore")  # noqa (false alarm - errors="ignore")

        return processed_data.values

    def _process_molecular_descriptors_for_qsrr_training_data(self) -> ndarray:
        return self._process_data(
            self.qsrr_ic_dataset.molecular_descriptors_for_qsrr_training_df,
            {QsrrIcConstants.ANALYTE}
        )

    def _process_molecular_descriptors_for_iso2grad_data(self) -> ndarray:
        return self._process_data(
            self.qsrr_ic_dataset.molecular_descriptors_for_iso2grad_df,
            {QsrrIcConstants.ANALYTE}
        )

    def _process_gradient_void_times(self) -> ndarray:
        return self._process_data(
            self.qsrr_ic_dataset.gradient_void_times_df,
            {QsrrIcConstants.GRADIENT_PROFILE}
        )

    def process(self) -> QsrrIcData:
        return QsrrIcData(
            molecular_descriptors_for_qsrr_training=self._process_molecular_descriptors_for_qsrr_training_data(),
            isocratic_retention=self.qsrr_ic_dataset.isocratic_retention_df[QsrrIcConstants.LOGK].values.reshape(-1, 1),
            molecular_descriptors_for_iso2grad=self._process_molecular_descriptors_for_iso2grad_data(),
            gradient_void_times=self._process_gradient_void_times(),
            gradient_profiles=self.qsrr_ic_dataset.gradient_profiles_df.values,
            gradient_retention=self.qsrr_ic_dataset.gradient_retention_df[QsrrIcConstants.TG].values.reshape(-1, 1)
        )
