import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from src.constants import RESULTS_DIRECTORY
from src.evaluation.constants import (
	FOLD_COUNT, REPEAT_COUNT, SEED,
	MODEL_COLUMN, CONFUSION_MATRIX_ROWS,
	CONFUSION_MATRIX_COLUMNS, WEIGHTED_VOTING_FILE,
	EVALUATION_FILE, SINGLE_EXAMPLE_CONFUSION_MATRIX_FILE
)
from src.dataset.dataset import Dataset
from src.evaluation.split import Split
from src.evaluation.fold import Fold
from src.evaluation.fold_result import FoldResult
from src.weighted_voting.weighted_voting import WeightedVoting
from src.exploration.constants import FEATURE_COLUMN
from src.evaluation.metric_table import get_metric_table


class Evaluation:

	def __init__(self, dataset: Dataset):
		self.dataset = dataset
		self.fold_results = []

	def _get_splits(self) -> list[Split]:
		cross_validation = RepeatedStratifiedKFold(n_splits=FOLD_COUNT, n_repeats=REPEAT_COUNT, random_state=SEED)
		index_pairs = cross_validation.split(self.dataset.features, self.dataset.labels)

		splits = []
		for training_indices, validation_indices in index_pairs:
			split = Split(training_indices, validation_indices)
			splits.append(split)

		return splits

	def _get_fold(self, split: Split, split_index: int) -> Fold:
		training_dataset = self.dataset.get_subset(split.training_indices)
		validation_dataset = self.dataset.get_subset(split.validation_indices)

		repeat_number = (split_index // FOLD_COUNT) + 1
		fold_number = (split_index % FOLD_COUNT) + 1
		fold = Fold(training_dataset, validation_dataset, repeat_number, fold_number)

		return fold

	def _get_folds(self) -> list[Fold]:
		splits = self._get_splits()

		folds = []
		for split_index, split in enumerate(splits):
			fold = self._get_fold(split, split_index)
			folds.append(fold)

		return folds

	def _get_fold_results(self) -> list[FoldResult]:
		folds = self._get_folds()

		fold_results = []
		for fold in folds:
			weighted_voting = WeightedVoting(fold.training_dataset)
			fold_result = FoldResult(fold, weighted_voting)
			fold_results.append(fold_result)

		return fold_results

	def _get_metric_table(self) -> pd.DataFrame:
		fold_metrics = {}

		for fold_result in self.fold_results:
			fold_metrics[fold_result.fold.name] = fold_result.metrics

		metric_table = get_metric_table(fold_metrics)

		return metric_table

	def _get_confusion_matrix_table(self, fold_result: FoldResult) -> pd.DataFrame:
		confusion_matrix_table = pd.DataFrame(fold_result.metrics.confusion_matrix, index=CONFUSION_MATRIX_ROWS, columns=CONFUSION_MATRIX_COLUMNS)

		return confusion_matrix_table

	def _print_metric_table(self, metric_table: pd.DataFrame) -> None:
		print("\nWeighted voting metric table:")
		print(metric_table)

	def _save_metric_table(self, metric_table: pd.DataFrame) -> None:
		RESULTS_DIRECTORY.mkdir(exist_ok=True)
		metric_table.to_csv(EVALUATION_FILE, index_label=MODEL_COLUMN)
		print(f"Weighted voting metric table saved to '{EVALUATION_FILE}'")

	def _print_single_example(self, example_result: FoldResult, confusion_matrix_table: pd.DataFrame) -> None:
		print("\nSingle Example (repeat 1, fold 1):")
		example_result.weighted_voting.print()

		print("\nConfusion matrix:")
		print(confusion_matrix_table)
		
		print(f"AUC: {example_result.metrics.auc}")
		print(f"balanced accuracy: {example_result.metrics.balanced_accuracy}")

	def _save_single_example(self, weighted_voting_table: pd.DataFrame, confusion_matrix_table: pd.DataFrame) -> None:
		RESULTS_DIRECTORY.mkdir(exist_ok=True)
		weighted_voting_table.to_csv(WEIGHTED_VOTING_FILE, index_label=FEATURE_COLUMN)
		confusion_matrix_table.to_csv(SINGLE_EXAMPLE_CONFUSION_MATRIX_FILE)
		print(f"Single example weighted voting table saved to '{WEIGHTED_VOTING_FILE}'")
		print(f"Single example confusion matrix saved to '{SINGLE_EXAMPLE_CONFUSION_MATRIX_FILE}'")

	def run(self) -> None:
		print("\nStart evaluating the weighted voting")

		self.fold_results = self._get_fold_results()
		single_example_result = self.fold_results[0]
		weighted_voting_table = single_example_result.weighted_voting.get_table()
		confusion_matrix_table = self._get_confusion_matrix_table(single_example_result)
		self._print_single_example(single_example_result, confusion_matrix_table)
		self._save_single_example(weighted_voting_table, confusion_matrix_table)

		metric_table = self._get_metric_table()
		self._print_metric_table(metric_table)
		self._save_metric_table(metric_table)

		print("End evaluating the weighted voting")
