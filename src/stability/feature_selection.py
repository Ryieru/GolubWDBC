import pandas as pd


class FeatureSelection:

	def __init__(self, selected_features: pd.Index):
		self.feature_names = set(selected_features)

	def contains(self, feature_name: str) -> bool:
		is_selected = feature_name in self.feature_names

		return is_selected

	def get_jaccard_similarity(self, other_selection: "FeatureSelection") -> float:
		feature_names_intersection = self.feature_names.intersection(other_selection.feature_names)
		feature_names_union = self.feature_names.union(other_selection.feature_names)
		jaccard_similarity = len(feature_names_intersection) / len(feature_names_union)

		return jaccard_similarity
