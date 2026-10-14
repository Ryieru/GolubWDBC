import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.preprocessing import StandardScaler

from src.dataset.dataset import Dataset
from src.types import Diagnosis


class LogisticRegression:

	def __init__(self, dataset: Dataset):
		self.scaler = StandardScaler()
		self.model = SklearnLogisticRegression(solver="liblinear", C=1.0, max_iter=1000)

		standardized_features = self.scaler.fit_transform(dataset.features)
		self.model.fit(standardized_features, dataset.labels)

	def compute_probabilities(self, features: pd.DataFrame) -> np.ndarray:
		standardized_features = self.scaler.transform(features)
		diagnosis_probabilities = self.model.predict_proba(standardized_features)
		malignant_probabilities = diagnosis_probabilities[:, Diagnosis.MALIGNANT.value]

		return malignant_probabilities

	def predict_diagnosis(self, features: pd.DataFrame) -> np.ndarray:
		malignant_probabilities = self.compute_probabilities(features)
		is_malignant = malignant_probabilities >= 0.5
		predictions = np.where(is_malignant, Diagnosis.MALIGNANT.value, Diagnosis.BENIGN.value)

		return predictions
	