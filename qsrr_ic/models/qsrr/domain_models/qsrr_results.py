from .qsrr_data import QsrrData
from .qsrr_metrics import QsrrMetrics


class QsrrResults:
    def __init__(
        self,
        qsrr_predictions: QsrrData,
        qsrr_metrics: QsrrMetrics,
    ):
        self.qsrr_predictions = qsrr_predictions
        self.qsrr_metrics = qsrr_metrics
