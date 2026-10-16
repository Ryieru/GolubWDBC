import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.constants import TABLES_DIRECTORY
from src.exploration.constants import (
	EXPLORATION_FILE, FEATURE_COLUMN,
	AUC_COLUMN
)
from src.dataset.dataset import Dataset
from src.types import Diagnosis
from src.dataset.diagnosis_statistics import DiagnosisStatistics


class Exploration:

	def __init__(self, dataset: Dataset):
		self.dataset = dataset
		self.exploration_table = pd.DataFrame()

	def _get_feature_aucs(self) -> pd.Series:
		aucs = {}

		for feature_name in self.dataset.features.columns:
			feature_values = self.dataset.features[feature_name]
			feature_auc = roc_auc_score(self.dataset.labels, feature_values)
			aucs[feature_name] = feature_auc

		feature_aucs = pd.Series(aucs)

		return feature_aucs

	def _get_exploration_table(self) -> pd.DataFrame:
		benign_statistics = DiagnosisStatistics(self.dataset, Diagnosis.BENIGN)
		malignant_statistics = DiagnosisStatistics(self.dataset, Diagnosis.MALIGNANT)

		benign_columns = benign_statistics.get_columns()
		malignant_columns = malignant_statistics.get_columns()
		statistic_columns = benign_columns + malignant_columns

		table_columns = {}
		for column in statistic_columns:
			table_columns[column.name] = column.values

		feature_aucs = self._get_feature_aucs()
		table_columns[AUC_COLUMN] = feature_aucs

		exploration_table = pd.DataFrame(table_columns)

		return exploration_table

	def _save_exploration_table(self, exploration_table: pd.DataFrame) -> None:
		TABLES_DIRECTORY.mkdir(parents=True, exist_ok=True)
		exploration_table.to_csv(EXPLORATION_FILE, index_label=FEATURE_COLUMN)
		print(f"Exploration table saved to '{EXPLORATION_FILE}'")

	def _print_dataset_size(self) -> None:
		patient_count, feature_count = self.dataset.features.shape
		print(f"Patients: {patient_count}, features: {feature_count}")

		diagnosis_counts = np.bincount(self.dataset.labels)
		for diagnosis in Diagnosis:
			diagnosis_count = diagnosis_counts[diagnosis.value]
			print(f"{diagnosis.name}: {diagnosis_count}")

	def _print_exploration_table(self, exploration_table: pd.DataFrame) -> None:
		print("\nExploration table:")
		print(exploration_table)

	def run(self) -> None:
		print("\nStart exploring the dataset")

		self._print_dataset_size()

		self.exploration_table = self._get_exploration_table()
		self._print_exploration_table(self.exploration_table)
		self._save_exploration_table(self.exploration_table)

		print("End exploring the dataset")
