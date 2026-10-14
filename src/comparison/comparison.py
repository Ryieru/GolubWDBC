import pandas as pd

from src.constants import RESULTS_DIRECTORY
from src.comparison.constants import (
    WEIGHTED_VOTING_AUC_COLUMN, LOGISTIC_REGRESSION_AUC_COLUMN,
    WEIGHTED_VOTING_BALANCED_ACCURACY_COLUMN, LOGISTIC_REGRESSION_BALANCED_ACCURACY_COLUMN,
    FOLD_TABLE_FILE, SUMMARY_TABLE_FILE, METRIC_COLUMNS
)
from src.evaluation.constants import (
    REPEAT_COLUMN, FOLD_COLUMN, SUMMARY_STATISTICS
)
from src.evaluation.fold import Fold
from src.evaluation.fold_result import FoldResult
from src.evaluation.metrics import Metrics
from src.logistic_regression.logistic_regression import LogisticRegression


class Comparison:

    def __init__(self, fold_results: list[FoldResult]):
        self.fold_results = fold_results

    def _get_logistic_regression_metrics(self, fold: Fold) -> Metrics:
        validation_features = fold.validation_dataset.features
        validation_labels = fold.validation_dataset.labels

        logistic_regression = LogisticRegression(fold.training_dataset)
        malignant_probabilities = logistic_regression.compute_probabilities(validation_features)
        predictions = logistic_regression.predict_diagnosis(validation_features)
        logistic_regression_metrics = Metrics(validation_labels, malignant_probabilities, predictions)

        return logistic_regression_metrics

    def _get_fold_table(self) -> pd.DataFrame:
        table_rows = []

        for fold_result in self.fold_results:
            weighted_voting_metrics = fold_result.metrics
            logistic_regression_metrics = self._get_logistic_regression_metrics(fold_result.fold)
            table_row = {
                REPEAT_COLUMN: fold_result.fold.repeat_number,
                FOLD_COLUMN: fold_result.fold.fold_number,
                WEIGHTED_VOTING_AUC_COLUMN: weighted_voting_metrics.auc,
                LOGISTIC_REGRESSION_AUC_COLUMN: logistic_regression_metrics.auc,
                WEIGHTED_VOTING_BALANCED_ACCURACY_COLUMN: weighted_voting_metrics.balanced_accuracy,
                LOGISTIC_REGRESSION_BALANCED_ACCURACY_COLUMN: logistic_regression_metrics.balanced_accuracy
            }

            table_rows.append(table_row)

        fold_table = pd.DataFrame(table_rows)

        return fold_table

    def _get_summary_table(self, fold_table: pd.DataFrame) -> pd.DataFrame:
        metric_table = fold_table[METRIC_COLUMNS]
        summary_table = metric_table.aggregate(SUMMARY_STATISTICS)

        return summary_table

    def _print_tables(self, fold_table: pd.DataFrame, summary_table: pd.DataFrame) -> None:
        print("\nComparison fold table:")
        print(fold_table)

        print("\nComparison summary table:")
        print(summary_table)

    def _save_tables(self, fold_table: pd.DataFrame, summary_table: pd.DataFrame) -> None:
        RESULTS_DIRECTORY.mkdir(exist_ok=True)
        fold_table.to_csv(FOLD_TABLE_FILE, index=False)
        summary_table.to_csv(SUMMARY_TABLE_FILE)

        print(f"Comparison fold table saved to '{FOLD_TABLE_FILE}'")
        print(f"Comparison summary table saved to '{SUMMARY_TABLE_FILE}'")

    def run(self) -> None:
        print("\nStart comparing the weighted voting with the logistic regression")

        fold_table = self._get_fold_table()
        summary_table = self._get_summary_table(fold_table)

        self._print_tables(fold_table, summary_table)
        self._save_tables(fold_table, summary_table)
        
        print("\nEnd comparing the weighted voting with the logistic regression")
