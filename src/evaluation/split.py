import numpy as np


class Split:

    def __init__(self, training_indices: np.ndarray, validation_indices: np.ndarray):
        self.training_indices = training_indices
        self.validation_indices = validation_indices
