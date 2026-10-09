import numpy as np
import pandas as pd

from src.dataset.dataset import Dataset
from src.types import Diagnosis
from src.dataset.diagnosis_statistics import DiagnosisStatistics
from src.weighted_voting.constants import (
	WEIGHT_COLUMN, MIDPOINT_COLUMN
)


class WeightedVoting:

	def __init__(self, dataset: Dataset):
		malignant_statistics = DiagnosisStatistics(dataset, Diagnosis.MALIGNANT)
		benign_statistics = DiagnosisStatistics(dataset, Diagnosis.BENIGN)

		weights = self._get_weights(malignant_statistics, benign_statistics)
		midpoints = self._get_midpoints(malignant_statistics, benign_statistics)

		self.selected_features = self._get_selected_features(weights)
		self.weights = weights[self.selected_features]
		self.midpoints = midpoints[self.selected_features]

	def _get_weights(self, malignant_statistics: DiagnosisStatistics, benign_statistics: DiagnosisStatistics) -> pd.Series:
		mean_differences = malignant_statistics.means - benign_statistics.means
		std_sums = malignant_statistics.stds + benign_statistics.stds
		weights = mean_differences / std_sums

		return weights

	def _get_midpoints(self, malignant_statistics: DiagnosisStatistics, benign_statistics: DiagnosisStatistics) -> pd.Series:
		malignant_means = malignant_statistics.means
		benign_means = benign_statistics.means
		midpoints = (malignant_means + benign_means) / 2

		return midpoints

	def _get_selected_features(self, weights: pd.Series) -> pd.Index:
		absolute_weights = weights.abs()
		largest_weights = absolute_weights.nlargest(10)
		selected_features = largest_weights.index

		return selected_features

	def print(self) -> None:
		weighted_voting_table = pd.DataFrame({
			WEIGHT_COLUMN: self.weights,
			MIDPOINT_COLUMN: self.midpoints
		})

		print("\nWeighted voting table:")
		print(weighted_voting_table)
