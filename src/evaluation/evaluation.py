import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from src.constants import RESULTS_DIRECTORY
from src.evaluation.constants import (
	FOLD_COUNT, REPEAT_COUNT, SEED,
	REPEAT_COLUMN, FOLD_COLUMN, AUC_COLUMN,
	BALANCED_ACCURACY_COLUMN, METRIC_COLUMNS,
	SUMMARY_STATISTICS, CONFUSION_MATRIX_ROWS,
	CONFUSION_MATRIX_COLUMNS, FOLD_TABLE_FILE,
	SUMMARY_TABLE_FILE, SINGLE_EXAMPLE_CONFUSION_MATRIX_FILE
)
from src.dataset.dataset import Dataset
from src.evaluation.split import Split
from src.evaluation.fold import Fold
from src.evaluation.fold_result import FoldResult
from src.weighted_voting.weighted_voting import WeightedVoting


class Evaluation:

	def __init__(self, dataset: Dataset):
		self.dataset = dataset

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

	def _get_fold_table(self, fold_results: list[FoldResult]) -> pd.DataFrame:
		table_rows = []

		for fold_result in fold_results:
			table_row = {
				REPEAT_COLUMN: fold_result.fold.repeat_number,
				FOLD_COLUMN: fold_result.fold.fold_number,
				AUC_COLUMN: fold_result.auc,
				BALANCED_ACCURACY_COLUMN: fold_result.balanced_accuracy
			}

			table_rows.append(table_row)

		fold_table = pd.DataFrame(table_rows)

		return fold_table

	def _get_summary_table(self, fold_table: pd.DataFrame) -> pd.DataFrame:
		metric_table = fold_table[METRIC_COLUMNS]
		summary_table = metric_table.agg(SUMMARY_STATISTICS)

		return summary_table

	def _get_confusion_matrix_table(self, fold_result: FoldResult) -> pd.DataFrame:
		confusion_matrix_table = pd.DataFrame(fold_result.confusion_matrix, index=CONFUSION_MATRIX_ROWS, columns=CONFUSION_MATRIX_COLUMNS)

		return confusion_matrix_table

	def _print_single_example(self, example_result: FoldResult, confusion_matrix_table: pd.DataFrame) -> None:
		print("\nSingle Example (repeat 1, fold 1):")
		example_result.weighted_voting.print()

		print("\nConfusion matrix:")
		print(confusion_matrix_table)
		
		print(f"AUC: {example_result.auc}, balanced accuracy: {example_result.balanced_accuracy}")

	def _save_single_example(self, confusion_matrix_table: pd.DataFrame) -> None:
		RESULTS_DIRECTORY.mkdir(exist_ok=True)
		confusion_matrix_table.to_csv(SINGLE_EXAMPLE_CONFUSION_MATRIX_FILE)
		print(f"Example confusion matrix saved to '{SINGLE_EXAMPLE_CONFUSION_MATRIX_FILE}'")

	def _print_tables(self, fold_table: pd.DataFrame, summary_table: pd.DataFrame) -> None:
		print("\nFold table:")
		print(fold_table)

		print("\nSummary:")
		print(summary_table)

	def _save_tables(self, fold_table: pd.DataFrame, summary_table: pd.DataFrame) -> None:
		RESULTS_DIRECTORY.mkdir(exist_ok=True)
		fold_table.to_csv(FOLD_TABLE_FILE, index=False)
		summary_table.to_csv(SUMMARY_TABLE_FILE)
		print(f"Fold table saved to '{FOLD_TABLE_FILE}'")
		print(f"Summary table saved to '{SUMMARY_TABLE_FILE}'")

	def run(self) -> None:
		print("\nStart evaluating the weighted voting")

		fold_results = self._get_fold_results()
		single_example_result = fold_results[0]
		confusion_matrix_table = self._get_confusion_matrix_table(single_example_result)
		self._print_single_example(single_example_result, confusion_matrix_table)
		self._save_single_example(confusion_matrix_table)

		fold_table = self._get_fold_table(fold_results)
		summary_table = self._get_summary_table(fold_table)
		self._print_tables(fold_table, summary_table)
		self._save_tables(fold_table, summary_table)

		print("End evaluating the weighted voting")
