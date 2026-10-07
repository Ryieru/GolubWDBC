from src.constants import PROJECT_ROOT
from src.types import Diagnosis


DATASET_FILE = PROJECT_ROOT / "data" / "wdbc.csv"

SAMPLE_ID_COLUMN = "sample_id"
DIAGNOSIS_COLUMN = "diagnosis"
METADATA_COLUMNS = [SAMPLE_ID_COLUMN, DIAGNOSIS_COLUMN]

DIAGNOSIS_CODES = {
	"benign": Diagnosis.BENIGN.value,
	"malignant": Diagnosis.MALIGNANT.value
}
