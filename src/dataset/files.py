import pandas as pd

from src.dataset.constants import (
	DATASET_FILE, METADATA_COLUMNS,
	DIAGNOSIS_COLUMN, DIAGNOSIS_CODES
)
from src.dataset.dataset import Dataset


def load_dataset() -> Dataset:
	print("Start loading the dataset")

	table = pd.read_csv(DATASET_FILE)
	features = table.drop(columns=METADATA_COLUMNS)
	diagnosis = table[DIAGNOSIS_COLUMN]
	codes = diagnosis.map(DIAGNOSIS_CODES)
	labels = codes.to_numpy()
	dataset = Dataset(features, labels)

	print("End loading the dataset")

	return dataset
