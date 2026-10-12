import numpy as np
import pandas as pd


class Dataset:

	def __init__(self, features: pd.DataFrame, labels: np.ndarray):
		self.features = features
		self.labels = labels

	def get_subset(self, indices: np.ndarray) -> "Dataset":
		subset_features = self.features.iloc[indices]
		subset_labels = self.labels[indices]
		subset = Dataset(subset_features, subset_labels)

		return subset
