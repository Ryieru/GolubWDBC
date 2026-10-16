import numpy as np
from sklearn.metrics import (
	roc_auc_score, roc_curve,
	balanced_accuracy_score, confusion_matrix
)


class Metrics:

	def __init__(self, labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray):
		self.auc = roc_auc_score(labels, scores)
		self.false_positive_rates, self.true_positive_rates, _ = roc_curve(labels, scores)
		self.balanced_accuracy = balanced_accuracy_score(labels, predictions)
		self.confusion_matrix = confusion_matrix(labels, predictions)
