from src.evaluation.fold import Fold
from src.weighted_voting.weighted_voting import WeightedVoting
from src.evaluation.metrics import Metrics


class FoldResult:

	def __init__(self, fold: Fold, weighted_voting: WeightedVoting):
		validation_features = fold.validation_dataset.features
		validation_labels = fold.validation_dataset.labels

		votes = weighted_voting.compute_votes(validation_features)
		predictions = weighted_voting.predict_diagnosis(validation_features)

		self.fold = fold
		self.weighted_voting = weighted_voting
		self.metrics = Metrics(validation_labels, votes, predictions)
