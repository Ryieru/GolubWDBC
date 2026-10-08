import pandas as pd

from src.types import (
	Diagnosis, Statistic
)


class DiagnosisStatistics:

	class Column:

		def __init__(self, name: str, values: pd.Series):
			self.name = name
			self.values = values

	def __init__(self, diagnosis: Diagnosis, diagnosis_features: pd.DataFrame):
		self.diagnosis = diagnosis
		self.diagnosis_features = diagnosis_features
		self.statistic_values = {
			Statistic.MEAN: self.diagnosis_features.mean(),
			Statistic.STD: self.diagnosis_features.std()
		}

	def _get_column(self, statistic: Statistic) -> Column:
		name = f"{self.diagnosis.name}_{statistic.value}"
		values = self.statistic_values[statistic]
		column = DiagnosisStatistics.Column(name, values)

		return column

	def get_columns(self) -> list[Column]:
		columns = []
		
		for statistic in Statistic:
			column = self._get_column(statistic)
			columns.append(column)

		return columns
