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
