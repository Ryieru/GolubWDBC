import pandas as pd

from src.types import Statistic
from src.evaluation.constants import (
	AUC_COLUMN, BALANCED_ACCURACY_COLUMN
)
from src.evaluation.metrics import Metrics


def get_metric_table(fold_metrics: dict[str, Metrics]) -> pd.DataFrame:
	fold_names = fold_metrics.keys()
	table_rows = []

	for metrics in fold_metrics.values():
		table_row = {
			AUC_COLUMN: metrics.auc,
			BALANCED_ACCURACY_COLUMN: metrics.balanced_accuracy
		}

		table_rows.append(table_row)

	metric_table = pd.DataFrame(table_rows, index=fold_names)
	means = metric_table.mean()
	stds = metric_table.std()

	metric_table.loc[Statistic.MEAN.value] = means
	metric_table.loc[Statistic.STD.value] = stds

	return metric_table
