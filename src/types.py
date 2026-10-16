from enum import Enum


class Diagnosis(Enum):
	BENIGN = 0
	MALIGNANT = 1


class Statistic(Enum):
	MEAN = "MEAN"
	STD = "STD"


class FeatureGroup(Enum):
	SIZE = "Size"
	CONTOUR_SHAPE = "Contour shape"
	TEXTURE = "Texture"
