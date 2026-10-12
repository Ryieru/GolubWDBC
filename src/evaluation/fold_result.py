from sklearn.metrics import (
	roc_auc_score, balanced_accuracy_score,
	confusion_matrix
)

from src.evaluation.fold import Fold
from src.weighted_voting.weighted_voting import WeightedVoting


class FoldResult:

	def __init__(self, fold: Fold, weighted_voting: WeightedVoting):
		validation_features = fold.validation_dataset.features
		validation_labels = fold.validation_dataset.labels

		votes = weighted_voting.compute_votes(validation_features)
		predictions = weighted_voting.predict_diagnosis(validation_features)

		self.fold = fold
		self.weighted_voting = weighted_voting
		self.auc = roc_auc_score(validation_labels, votes)
		self.balanced_accuracy = balanced_accuracy_score(validation_labels, predictions)
		self.confusion_matrix = confusion_matrix(validation_labels, predictions)
