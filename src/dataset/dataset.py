import numpy as np
import pandas as pd


class Dataset:

	def __init__(self, features: pd.DataFrame, labels: np.ndarray):
		self.features = features
		self.labels = labels
