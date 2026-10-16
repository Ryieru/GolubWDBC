import pandas as pd

from src.constants import TABLES_DIRECTORY
from src.comparison.constants import COMPARISON_FILE
from src.evaluation.constants import MODEL_COLUMN
from src.evaluation.fold import Fold
from src.evaluation.fold_result import FoldResult
from src.evaluation.metrics import Metrics
from src.logistic_regression.logistic_regression import LogisticRegression
from src.evaluation.metric_table import get_metric_table


class Comparison:

	def __init__(self, fold_results: list[FoldResult]):
		self.fold_results = fold_results
		self.fold_metrics = {}
		self.metric_table = pd.DataFrame()

	def _get_logistic_regression_metrics(self, fold: Fold) -> Metrics:
		validation_features = fold.validation_dataset.features
		validation_labels = fold.validation_dataset.labels

		logistic_regression = LogisticRegression(fold.training_dataset)
		malignant_probabilities = logistic_regression.compute_probabilities(validation_features)
		predictions = logistic_regression.predict_diagnosis(validation_features)
		logistic_regression_metrics = Metrics(validation_labels, malignant_probabilities, predictions)

		return logistic_regression_metrics

	def _get_fold_metrics(self) -> dict[str, Metrics]:
		fold_metrics = {}

		for fold_result in self.fold_results:
			fold = fold_result.fold
			logistic_regression_metrics = self._get_logistic_regression_metrics(fold)
			fold_metrics[fold.name] = logistic_regression_metrics

		return fold_metrics

	def _print_metric_table(self, metric_table: pd.DataFrame) -> None:
		print("\nLogistic regression metric table:")
		print(metric_table)

	def _save_metric_table(self, metric_table: pd.DataFrame) -> None:
		TABLES_DIRECTORY.mkdir(parents=True, exist_ok=True)
		metric_table.to_csv(COMPARISON_FILE, index_label=MODEL_COLUMN)
		print(f"Logistic regression metric table saved to '{COMPARISON_FILE}'")

	def run(self) -> None:
		print("\nStart comparing the weighted voting with the logistic regression")

		self.fold_metrics = self._get_fold_metrics()
		self.metric_table = get_metric_table(self.fold_metrics)
		self._print_metric_table(self.metric_table)
		self._save_metric_table(self.metric_table)

		print("\nEnd comparing the weighted voting with the logistic regression")
