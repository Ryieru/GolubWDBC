from src.dataset.dataset import Dataset


class Fold:

	def __init__(self, training_dataset: Dataset, validation_dataset: Dataset, repeat_number: int, fold_number: int):
		self.training_dataset = training_dataset
		self.validation_dataset = validation_dataset
		self.repeat_number = repeat_number
		self.fold_number = fold_number
