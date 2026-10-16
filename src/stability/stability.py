import pandas as pd

from src.constants import TABLES_DIRECTORY
from src.stability.constants import (
	FREQUENCY_TABLE_FILE, JACCARD_TABLE_FILE,
	FEATURE_COLUMN, SELECTION_COUNT_COLUMN,
	SELECTION_FREQUENCY_COLUMN, MEAN_JACCARD_COLUMN,
)
from src.dataset.dataset import Dataset
from src.evaluation.fold_result import FoldResult
from src.stability.feature_selection import FeatureSelection


class Stability:

	def __init__(self, dataset: Dataset, fold_results: list[FoldResult]):
		self.dataset = dataset
		self.feature_selections = self._get_feature_selections(fold_results)
		self.frequency_table = pd.DataFrame()
		self.mean_jaccard_similarity = 0.0

	def _get_feature_selections(self, fold_results: list[FoldResult]) -> list[FeatureSelection]:
		feature_selections = []

		for fold_result in fold_results:
			selected_features = fold_result.weighted_voting.selected_features
			feature_selection = FeatureSelection(selected_features)
			feature_selections.append(feature_selection)

		return feature_selections

	def _get_selection_count(self, feature_name: str) -> int:
		selection_count = 0

		for feature_selection in self.feature_selections:
			if feature_selection.contains(feature_name):
				selection_count += 1

		return selection_count

	def _get_frequency_table(self) -> pd.DataFrame:
		feature_names = self.dataset.features.columns
		selection_total = len(self.feature_selections)
		table_rows = []

		for feature_name in feature_names:
			selection_count = self._get_selection_count(feature_name)
			selection_frequency = selection_count / selection_total
			table_row = {
				SELECTION_COUNT_COLUMN: selection_count,
				SELECTION_FREQUENCY_COLUMN: selection_frequency
			}

			table_rows.append(table_row)

		frequency_table = pd.DataFrame(table_rows, index=feature_names)

		return frequency_table

	def _get_mean_jaccard_similarity(self) -> float:
		selection_total = len(self.feature_selections)
		jaccard_similarities = []

		for i in range(selection_total):
			for j in range(i + 1, selection_total):
				first_selection = self.feature_selections[i]
				second_selection = self.feature_selections[j]

				jaccard_similarity = first_selection.get_jaccard_similarity(second_selection)
				jaccard_similarities.append(jaccard_similarity)

		mean_jaccard_similarity = sum(jaccard_similarities) / len(jaccard_similarities)

		return mean_jaccard_similarity

	def get_model_jaccard_similarities(self) -> list[float]:
		model_jaccard_similarities = []

		for i, model_selection in enumerate(self.feature_selections):
			jaccard_similarities = []
			for j, other_selection in enumerate(self.feature_selections):
				if i == j:
					continue

				jaccard_similarity = model_selection.get_jaccard_similarity(other_selection)
				jaccard_similarities.append(jaccard_similarity)

			model_jaccard_similarity = sum(jaccard_similarities) / len(jaccard_similarities)
			model_jaccard_similarities.append(model_jaccard_similarity)

		return model_jaccard_similarities

	def get_identical_selection_counts(self) -> list[int]:
		identical_selection_counts = []

		for model_selection in self.feature_selections:
			identical_selection_count = 0
			for other_selection in self.feature_selections:
				jaccard_similarity = model_selection.get_jaccard_similarity(other_selection)
				if jaccard_similarity == 1:
					identical_selection_count += 1

			identical_selection_counts.append(identical_selection_count)

		return identical_selection_counts

	def _get_jaccard_table(self, mean_jaccard_similarity: float) -> pd.DataFrame:
		jaccard_row = {
			MEAN_JACCARD_COLUMN: [mean_jaccard_similarity]
		}

		jaccard_table = pd.DataFrame(jaccard_row)

		return jaccard_table

	def _print_tables(self, frequency_table: pd.DataFrame, jaccard_table: pd.DataFrame) -> None:
		print("\nSelection frequency table:")
		print(frequency_table)

		print("\nJaccard table:")
		print(jaccard_table)

	def _save_tables(self, frequency_table: pd.DataFrame, jaccard_table: pd.DataFrame) -> None:
		TABLES_DIRECTORY.mkdir(parents=True, exist_ok=True)
		frequency_table.to_csv(FREQUENCY_TABLE_FILE, index_label=FEATURE_COLUMN)
		jaccard_table.to_csv(JACCARD_TABLE_FILE, index=False)

		print(f"Selection frequency table saved to '{FREQUENCY_TABLE_FILE}'")
		print(f"Jaccard table saved to '{JACCARD_TABLE_FILE}'")

	def run(self) -> None:
		print("\nStart evaluating the feature selection stability")

		self.frequency_table = self._get_frequency_table()
		self.mean_jaccard_similarity = self._get_mean_jaccard_similarity()
		jaccard_table = self._get_jaccard_table(self.mean_jaccard_similarity)

		self._print_tables(self.frequency_table, jaccard_table)
		self._save_tables(self.frequency_table, jaccard_table)
		
		print("\nEnd evaluating the feature selection stability")
