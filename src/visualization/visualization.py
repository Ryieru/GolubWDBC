import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.types import Diagnosis, FeatureGroup, Statistic
from src.exploration.exploration import Exploration
from src.exploration.constants import AUC_COLUMN as FEATURE_AUC_COLUMN
from src.evaluation.evaluation import Evaluation
from src.evaluation.constants import (
	AUC_COLUMN, BALANCED_ACCURACY_COLUMN
)
from src.stability.stability import Stability
from src.stability.constants import SELECTION_COUNT_COLUMN
from src.comparison.comparison import Comparison
from src.evaluation.metrics import Metrics
from src.visualization.figures import (
	create_figure, style_axes, save_figure
)
from src.visualization.constants import (
	DIAGNOSIS_COUNTS_FIGURE_FILE, FEATURE_AUCS_FIGURE_FILE,
	FEATURE_WEIGHTS_FIGURE_FILE,
	VOTE_CONTRIBUTIONS_FIGURE_FILE,
	SINGLE_EXAMPLE_CONFUSION_MATRIX_FIGURE_FILE,
	SINGLE_EXAMPLE_VOTES_FIGURE_FILE,
	SELECTION_FREQUENCIES_FIGURE_FILE,
	MODEL_STABILITY_FIGURE_FILE,
	FOLD_METRICS_FIGURE_FILE,
	ROC_CURVES_FIGURE_FILE
)


class Visualization:

	def __init__(self, exploration: Exploration, evaluation: Evaluation, stability: Stability, comparison: Comparison):
		self.exploration = exploration
		self.evaluation = evaluation
		self.stability = stability
		self.comparison = comparison

	def _get_feature_group(self, feature_name: str) -> FeatureGroup:
		property_groups = {
			"radius": FeatureGroup.SIZE,
			"perimeter": FeatureGroup.SIZE,
			"area": FeatureGroup.SIZE,
			"smoothness": FeatureGroup.CONTOUR_SHAPE,
			"compactness": FeatureGroup.CONTOUR_SHAPE,
			"concavity": FeatureGroup.CONTOUR_SHAPE,
			"concave_points": FeatureGroup.CONTOUR_SHAPE,
			"symmetry": FeatureGroup.CONTOUR_SHAPE,
			"fractal_dimension": FeatureGroup.CONTOUR_SHAPE,
			"texture": FeatureGroup.TEXTURE
		}

		property_name = feature_name.removesuffix("_error")
		property_name = property_name.removeprefix("mean_")
		property_name = property_name.removeprefix("worst_")
		feature_group = property_groups[property_name]

		return feature_group

	def _get_group_color(self, feature_group: FeatureGroup) -> str:
		group_colors = {
			FeatureGroup.SIZE: "#2a78d6",
			FeatureGroup.CONTOUR_SHAPE: "#eb6834",
			FeatureGroup.TEXTURE: "#1baf7a"
		}

		group_color = group_colors[feature_group]

		return group_color

	def _get_feature_colors(self, feature_names: pd.Index) -> list[str]:
		feature_colors = []

		for feature_name in feature_names:
			feature_group = self._get_feature_group(feature_name)
			feature_color = self._get_group_color(feature_group)
			feature_colors.append(feature_color)

		return feature_colors

	def _add_feature_group_legend(self, axes: Axes, feature_names: pd.Index) -> None:
		feature_groups = set()
		for feature_name in feature_names:
			feature_group = self._get_feature_group(feature_name)
			feature_groups.add(feature_group)

		legend_handles, _ = axes.get_legend_handles_labels()
		for feature_group in FeatureGroup:
			if feature_group not in feature_groups:
				continue

			group_color = self._get_group_color(feature_group)
			legend_handle = Patch(facecolor=group_color, edgecolor="#333333", label=feature_group.value)
			legend_handles.append(legend_handle)

		axes.legend(handles=legend_handles, loc="lower right")

	def _plot_feature_bars(self, feature_values: pd.Series, figure_height: float, value_label: str) -> tuple[Figure, Axes]:
		sorted_values = feature_values.sort_values(kind="stable")
		feature_names = sorted_values.index
		feature_colors = self._get_feature_colors(feature_names)
		bar_lengths = sorted_values.to_numpy()

		figure = create_figure(figure_height)
		axes = figure.subplots()
		axes.barh(feature_names, bar_lengths, height=0.7, color=feature_colors, edgecolor="#333333", linewidth=0.6)
		axes.set_xlabel(value_label)
		axes.margins(y=0.01)

		return figure, axes

	def _plot_diagnosis_counts(self) -> Figure:
		diagnosis_colors = {
			Diagnosis.BENIGN: "#2a78d6",
			Diagnosis.MALIGNANT: "#eb6834"
		}
		diagnosis_labels = {
			Diagnosis.BENIGN: "Benign",
			Diagnosis.MALIGNANT: "Malignant"
		}

		diagnosis_counts = np.bincount(self.exploration.dataset.labels)
		patient_count = diagnosis_counts.sum()
		bar_names = []
		bar_colors = []
		bar_labels = []

		for diagnosis in Diagnosis:
			diagnosis_count = diagnosis_counts[diagnosis.value]
			diagnosis_share = diagnosis_count / patient_count
			bar_names.append(diagnosis_labels[diagnosis])
			bar_colors.append(diagnosis_colors[diagnosis])
			bar_labels.append(f"{diagnosis_count} ({diagnosis_share:.1%})")

		figure = create_figure(3.2)
		axes = figure.subplots()
		diagnosis_bars = axes.bar(bar_names, diagnosis_counts, width=0.6, color=bar_colors, edgecolor="#333333", linewidth=0.6)
		axes.bar_label(diagnosis_bars, labels=bar_labels, padding=3)
		axes.margins(y=0.12)
		axes.set_ylabel("Patients")
		style_axes(axes, f"Diagnosis of the {patient_count} patients")

		return figure

	def _plot_feature_aucs(self) -> Figure:
		feature_aucs = self.exploration.exploration_table[FEATURE_AUC_COLUMN]
		figure, axes = self._plot_feature_bars(feature_aucs, 6.4, "Raw feature AUC (positive class: malignant)")
		axes.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="AUC = 0.5")
		style_axes(axes, "Link of each feature to the diagnosis")
		self._add_feature_group_legend(axes, feature_aucs.index)

		return figure

	def _plot_feature_weights(self) -> Figure:
		weighted_voting = self.evaluation.fold_results[0].weighted_voting
		absolute_weights = weighted_voting.feature_weights.abs()
		sorted_weights = absolute_weights.sort_values(kind="stable")
		feature_names = sorted_weights.index
		weight_lengths = sorted_weights.to_numpy()
		selection_threshold = absolute_weights[weighted_voting.selected_features].min()

		bar_colors = []
		for feature_name in feature_names:
			is_selected = feature_name in weighted_voting.selected_features
			bar_color = "#2a78d6" if is_selected else "#c8c8c8"
			bar_colors.append(bar_color)

		figure = create_figure(6.4)
		axes = figure.subplots()
		axes.barh(feature_names, weight_lengths, height=0.7, color=bar_colors, edgecolor="#333333", linewidth=0.6)
		axes.axvline(selection_threshold, color="black", linestyle="--", linewidth=1.2, label="Weight of the tenth feature")
		axes.set_xlabel("Absolute weight")
		axes.margins(y=0.01)
		style_axes(axes, "Weights of the 30 features, single example")

		legend_handles, _ = axes.get_legend_handles_labels()
		legend_handles.append(Patch(facecolor="#2a78d6", edgecolor="#333333", label="Selected (10 largest weights)"))
		legend_handles.append(Patch(facecolor="#c8c8c8", edgecolor="#333333", label="Not selected"))
		axes.legend(handles=legend_handles, loc="lower right")

		return figure

	def _get_vote_contributions(self) -> pd.Series:
		single_example_result = self.evaluation.fold_results[0]
		validation_features = single_example_result.fold.validation_dataset.features
		feature_votes = single_example_result.weighted_voting.compute_feature_votes(validation_features)
		absolute_votes = feature_votes.abs()
		vote_contributions = absolute_votes.mean()

		return vote_contributions

	def _plot_vote_contributions(self) -> Figure:
		vote_contributions = self._get_vote_contributions()
		figure, axes = self._plot_feature_bars(vote_contributions, 3.4, "Mean absolute contribution to the vote")
		contribution_bars = axes.containers[0]
		axes.bar_label(contribution_bars, fmt="{:.3g}", padding=3)
		axes.set_xlim(right=680)
		style_axes(axes, "Contribution to V(x), single example")
		self._add_feature_group_legend(axes, vote_contributions.index)

		return figure

	def _plot_single_example_confusion_matrix(self) -> Figure:
		diagnosis_labels = ["Benign", "Malignant"]

		single_example_result = self.evaluation.fold_results[0]
		confusion_matrix = single_example_result.metrics.confusion_matrix
		largest_count = confusion_matrix.max()
		cell_positions = np.arange(len(diagnosis_labels))

		figure = create_figure(3.4)
		axes = figure.subplots()
		axes.imshow(confusion_matrix, cmap="Blues", vmin=0, vmax=largest_count)

		for i in cell_positions:
			for j in cell_positions:
				cell_count = confusion_matrix[i, j]
				is_dark_cell = cell_count > largest_count / 2
				text_color = "white" if is_dark_cell else "black"
				axes.text(j, i, cell_count, ha="center", va="center", color=text_color, fontsize=14, fontweight="bold")

		axes.set_xticks(cell_positions, labels=diagnosis_labels)
		axes.set_yticks(cell_positions, labels=diagnosis_labels)
		axes.set_xlabel("Predicted diagnosis")
		axes.set_ylabel("True diagnosis")
		axes.set_title("Confusion matrix of the 114 validation patients, single example")

		return figure

	def _plot_single_example_votes(self) -> Figure:
		diagnosis_colors = {
			Diagnosis.BENIGN: "#2a78d6",
			Diagnosis.MALIGNANT: "#eb6834"
		}
		diagnosis_labels = {
			Diagnosis.BENIGN: "True benign",
			Diagnosis.MALIGNANT: "True malignant"
		}

		single_example_result = self.evaluation.fold_results[0]
		validation_dataset = single_example_result.fold.validation_dataset
		votes = single_example_result.weighted_voting.compute_votes(validation_dataset.features)
		vote_order = np.argsort(votes, kind="stable")
		sorted_votes = votes[vote_order]
		sorted_labels = validation_dataset.labels[vote_order]
		patient_positions = np.arange(sorted_votes.size)

		figure = create_figure(3.4)
		axes = figure.subplots()

		for diagnosis in Diagnosis:
			is_in_diagnosis = sorted_labels == diagnosis.value
			diagnosis_positions = patient_positions[is_in_diagnosis]
			diagnosis_votes = sorted_votes[is_in_diagnosis]
			diagnosis_color = diagnosis_colors[diagnosis]
			diagnosis_label = diagnosis_labels[diagnosis]
			axes.bar(diagnosis_positions, diagnosis_votes, width=1, color=diagnosis_color, edgecolor="#333333", linewidth=0.3, label=diagnosis_label)

		axes.axhline(0, color="black", linestyle="--", linewidth=1.2, label="Threshold V(x) = 0")
		axes.set_xlabel("Validation patients, sorted by vote")
		axes.set_ylabel("Vote V(x)")
		axes.margins(x=0.01)
		style_axes(axes, "Votes of the 114 validation patients, single example")
		axes.legend(loc="upper left")

		return figure

	def _plot_selection_frequencies(self) -> Figure:
		selection_counts = self.stability.frequency_table[SELECTION_COUNT_COLUMN]
		figure, axes = self._plot_feature_bars(selection_counts, 6.4, "Number of models selecting the feature (out of 15)")
		style_axes(axes, "Feature selection over the 15 models")
		self._add_feature_group_legend(axes, selection_counts.index)

		return figure

	def _get_model_set_colors(self) -> tuple[list[str], int]:
		identical_selection_counts = self.stability.get_identical_selection_counts()
		most_frequent_count = max(identical_selection_counts)
		model_colors = []

		for identical_selection_count in identical_selection_counts:
			is_most_frequent_set = identical_selection_count == most_frequent_count
			model_color = "#2a78d6" if is_most_frequent_set else "#eb6834"
			model_colors.append(model_color)

		return model_colors, most_frequent_count

	def _plot_model_stability(self) -> Figure:
		model_jaccard_similarities = self.stability.get_model_jaccard_similarities()
		balanced_accuracies = self._get_fold_values(self.evaluation.metric_table, BALANCED_ACCURACY_COLUMN)
		model_colors, most_frequent_count = self._get_model_set_colors()
		model_names = balanced_accuracies.index
		model_count = len(model_names)
		model_positions = np.arange(model_count)
		other_count = model_count - most_frequent_count

		figure = create_figure(5.0)
		jaccard_axes, balanced_accuracy_axes = figure.subplots(2, 1, sharex=True)

		jaccard_axes.scatter(model_positions, model_jaccard_similarities, s=50, color=model_colors, edgecolors="#333333", zorder=3)
		jaccard_axes.axhline(self.stability.mean_jaccard_similarity, color="black", linestyle="--", linewidth=1.2)
		jaccard_axes.set_ylabel("Mean Jaccard")
		style_axes(jaccard_axes, "Mean Jaccard similarity with the 14 other models")

		balanced_accuracy_axes.scatter(model_positions, balanced_accuracies, s=50, color=model_colors, edgecolors="#333333", zorder=3)
		balanced_accuracy_axes.set_xticks(model_positions, labels=model_names, rotation=90)
		balanced_accuracy_axes.set_xlabel("Model (repeat, fold)")
		balanced_accuracy_axes.set_ylabel("Balanced accuracy")
		style_axes(balanced_accuracy_axes, "Balanced accuracy of the weighted voting")

		mean_label = f"Mean pairwise Jaccard = {self.stability.mean_jaccard_similarity:.3f}"
		legend_handles = [
			Line2D([], [], marker="o", linestyle="", markersize=7, color="#2a78d6", markeredgecolor="#333333", label=f"Most frequent feature set ({most_frequent_count} models)"),
			Line2D([], [], marker="o", linestyle="", markersize=7, color="#eb6834", markeredgecolor="#333333", label=f"Other feature set ({other_count} models)"),
			Line2D([], [], color="black", linestyle="--", linewidth=1.2, label=mean_label)
		]
		figure.legend(handles=legend_handles, loc="outside upper center", ncols=2)

		return figure

	def _get_fold_values(self, metric_table: pd.DataFrame, metric_column: str) -> pd.Series:
		summary_rows = [Statistic.MEAN.value, Statistic.STD.value]
		fold_table = metric_table.drop(index=summary_rows)
		fold_values = fold_table[metric_column]

		return fold_values

	def _plot_metric_panel(self, axes: Axes, metric_column: str, metric_label: str) -> None:
		weighted_voting_values = self._get_fold_values(self.evaluation.metric_table, metric_column)
		logistic_regression_values = self._get_fold_values(self.comparison.metric_table, metric_column)
		fold_names = weighted_voting_values.index
		fold_count = len(fold_names)
		fold_positions = np.arange(fold_count)

		axes.plot(fold_positions, weighted_voting_values, marker="o", color="#2a78d6", label="Weighted voting")
		axes.plot(fold_positions, logistic_regression_values, marker="s", color="#eb6834", label="Logistic regression")
		axes.set_xticks(fold_positions, labels=fold_names, rotation=90)
		axes.set_ylabel(metric_label)
		style_axes(axes, metric_label)

	def _plot_fold_metrics(self) -> Figure:
		figure = create_figure(5.4)
		auc_axes, balanced_accuracy_axes = figure.subplots(2, 1, sharex=True)

		self._plot_metric_panel(auc_axes, AUC_COLUMN, "AUC")
		self._plot_metric_panel(balanced_accuracy_axes, BALANCED_ACCURACY_COLUMN, "Balanced accuracy")
		balanced_accuracy_axes.set_xlabel("Model (repeat, fold)")

		legend_handles, legend_labels = auc_axes.get_legend_handles_labels()
		figure.legend(legend_handles, legend_labels, loc="outside upper center", ncols=2)

		return figure

	def _plot_roc_panel(self, axes: Axes, fold_metrics: list[Metrics], method_color: str, method_label: str) -> None:
		curve_label = "One curve per model"

		for metrics in fold_metrics:
			axes.plot(metrics.false_positive_rates, metrics.true_positive_rates, color=method_color, alpha=0.5, linewidth=1.2, label=curve_label)
			curve_label = None

		axes.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Random classifier")
		axes.set_xlabel("False positive rate")
		axes.set_ylabel("True positive rate")
		style_axes(axes, method_label)
		axes.legend(loc="lower right")

	def _plot_roc_curves(self) -> Figure:
		weighted_voting_metrics = []
		for fold_result in self.evaluation.fold_results:
			weighted_voting_metrics.append(fold_result.metrics)

		logistic_regression_metrics = list(self.comparison.fold_metrics.values())

		figure = create_figure(3.6)
		weighted_voting_axes, logistic_regression_axes = figure.subplots(1, 2, sharey=True)

		self._plot_roc_panel(weighted_voting_axes, weighted_voting_metrics, "#2a78d6", "Weighted voting")
		self._plot_roc_panel(logistic_regression_axes, logistic_regression_metrics, "#eb6834", "Logistic regression")
		logistic_regression_axes.set_ylabel("")
		figure.suptitle("ROC curves of the 15 models")

		return figure

	def run(self) -> None:
		print("\nStart drawing the figures")

		diagnosis_counts_figure = self._plot_diagnosis_counts()
		save_figure(diagnosis_counts_figure, DIAGNOSIS_COUNTS_FIGURE_FILE)

		feature_aucs_figure = self._plot_feature_aucs()
		save_figure(feature_aucs_figure, FEATURE_AUCS_FIGURE_FILE)

		feature_weights_figure = self._plot_feature_weights()
		save_figure(feature_weights_figure, FEATURE_WEIGHTS_FIGURE_FILE)

		vote_contributions_figure = self._plot_vote_contributions()
		save_figure(vote_contributions_figure, VOTE_CONTRIBUTIONS_FIGURE_FILE)

		single_example_confusion_matrix_figure = self._plot_single_example_confusion_matrix()
		save_figure(single_example_confusion_matrix_figure, SINGLE_EXAMPLE_CONFUSION_MATRIX_FIGURE_FILE)

		single_example_votes_figure = self._plot_single_example_votes()
		save_figure(single_example_votes_figure, SINGLE_EXAMPLE_VOTES_FIGURE_FILE)

		selection_frequencies_figure = self._plot_selection_frequencies()
		save_figure(selection_frequencies_figure, SELECTION_FREQUENCIES_FIGURE_FILE)

		model_stability_figure = self._plot_model_stability()
		save_figure(model_stability_figure, MODEL_STABILITY_FIGURE_FILE)

		fold_metrics_figure = self._plot_fold_metrics()
		save_figure(fold_metrics_figure, FOLD_METRICS_FIGURE_FILE)

		roc_curves_figure = self._plot_roc_curves()
		save_figure(roc_curves_figure, ROC_CURVES_FIGURE_FILE)

		print("End drawing the figures")
