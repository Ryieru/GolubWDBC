import pandas as pd

from src.dataset.dataset import Dataset
from src.types import (
	Diagnosis, Statistic
)


class DiagnosisStatistics:

	class Column:

		def __init__(self, name: str, values: pd.Series):
			self.name = name
			self.values = values

	def __init__(self, dataset: Dataset, diagnosis: Diagnosis):
		self.diagnosis = diagnosis

		diagnosis_features = self._get_diagnosis_features(dataset)
		self.means = diagnosis_features.mean()
		self.stds = diagnosis_features.std()

	def _get_diagnosis_features(self, dataset: Dataset) -> pd.DataFrame:
		is_in_diagnosis = dataset.labels == self.diagnosis.value
		diagnosis_features = dataset.features[is_in_diagnosis]

		return diagnosis_features

	def _get_column(self, statistic: Statistic, values: pd.Series) -> Column:
		name = f"{self.diagnosis.name}_{statistic.value}"
		column = DiagnosisStatistics.Column(name, values)

		return column

	def get_columns(self) -> list[Column]:
		mean_column = self._get_column(Statistic.MEAN, self.means)
		std_column = self._get_column(Statistic.STD, self.stds)
		columns = [mean_column, std_column]

		return columns
