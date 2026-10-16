# GolubWDBC

Golub et al. (1999) classified leukemia samples from gene-expression measurements using a small set of weighted features. This repository applies the published weighted-voting method to a different disease and measurement system: numerical properties extracted from breast fine-needle-aspirate images, from the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

> Can the Golub weighted-voting method classify malignant breast samples accurately and robustly?

## Requirements

- uv 0.12.23
- Python 3.13.13
- numpy 2.5.3
- pandas 3.0.6
- scikit-learn 1.9.1
- matplotlib 3.11.2

## Run the pipeline

```
uv run main.py
```

## Project structure

```
GolubWDBC/
|-- main.py (application entry point, runs the whole pipeline)
|-- data/ (frozen WDBC dataset, its manifest and its data dictionary)
|-- src/
|   |-- dataset/ (loading of the dataset and per-diagnosis statistics)
|   |-- exploration/ (part 1, size of the dataset and link of each feature to the diagnosis)
|   |-- weighted_voting/ (part 2, weighted-voting model)
|   |-- evaluation/ (part 3, repeated stratified cross-validation and metrics)
|   |-- stability/ (part 4, selection frequencies and Jaccard similarity)
|   |-- logistic_regression/ (logistic regression model with standardization)
|   |-- comparison/ (part 5, logistic regression on the same 15 splits)
|   `-- visualization/ (figures of parts 1 to 5)
|-- results/ (outputs of the pipeline)
|   |-- tables/
|   `-- figures/
`-- report/ (PDF of the report)
```

## Outputs

The pipeline writes its tables in `results/tables/` and its figures in `results/figures/`. The number at the start of each file is the part of the assignment it belongs to.

| Tables | Content |
|---|---|
| `tables/1_exploration.csv` | Mean and standard deviation of each feature per diagnosis, and raw feature AUC |
| `tables/2_weighted_voting.csv` | Weights and midpoints of the ten features selected by the example model (R1F1) |
| `tables/3_evaluation.csv` | AUC and balanced accuracy of the weighted voting for each of the 15 models, with mean and standard deviation |
| `tables/3_single_example_confusion_matrix.csv` | Confusion matrix of the example model |
| `tables/4_stability.csv` | Selection count and frequency of each feature over the 15 models |
| `tables/4_jaccard.csv` | Mean pairwise Jaccard similarity of the 15 selected sets |
| `tables/5_comparison.csv` | AUC and balanced accuracy of the logistic regression for each of the 15 models, with mean and standard deviation |

| Figures | Content |
|---|---|
| `figures/1_diagnosis_counts.png` | Number of benign and malignant patients |
| `figures/1_feature_aucs.png` | Raw AUC of the 30 features, grouped by nucleus property |
| `figures/2_vote_contributions.png` | Mean contribution of the ten selected features to the vote of the example model |
| `figures/2_feature_weights.png` | Absolute weight of the 30 features in the example model |
| `figures/3_single_example_confusion_matrix.png` | Confusion matrix of the example model |
| `figures/3_single_example_votes.png` | Votes of the 114 validation patients of the example model, sorted |
| `figures/4_selection_frequencies.png` | Number of models selecting each feature |
| `figures/4_model_stability.png` | Mean Jaccard similarity and balanced accuracy of each model |
| `figures/5_fold_metrics.png` | AUC and balanced accuracy of both methods on the 15 splits |
| `figures/5_roc_curves.png` | ROC curves of the 15 models of both methods |

## References

- Golub TR et al. (1999). Molecular classification of cancer: class discovery and class prediction by gene expression monitoring. *Science*, 286(5439), 531-537. https://doi.org/10.1126/science.286.5439.531
- Wolberg W, Mangasarian O, Street N, Street W. Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
- Street WN, Wolberg WH, Mangasarian OL (1993). Nuclear feature extraction for breast tumor diagnosis. *IS&T/SPIE Symposium on Electronic Imaging*, 1905, 861-870. https://doi.org/10.1117/12.148698
- Wolberg WH, Street WN, Mangasarian OL (1994). Machine learning techniques to diagnose breast cancer from image-processed nuclear features of fine needle aspirates. *Cancer Letters*, 77(2-3), 163-171. https://doi.org/10.1016/0304-3835(94)90099-X
- Pedregosa F et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830. https://jmlr.org/papers/v12/pedregosa11a.html
