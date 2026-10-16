from src.dataset.files import load_dataset
from src.exploration.exploration import Exploration
from src.evaluation.evaluation import Evaluation
from src.stability.stability import Stability
from src.comparison.comparison import Comparison
from src.visualization.visualization import Visualization


def main() -> None:

	# 0. Loading the dataset
	dataset = load_dataset()

	# 1. Data exploration
	exploration = Exploration(dataset)
	exploration.run()

	# 2. Implement weighted voting
	# 3. Evaluate the model
	evaluation = Evaluation(dataset)
	evaluation.run()

	# 4. Evaluate feature selection stability
	stability = Stability(dataset, evaluation.fold_results)
	stability.run()

	# 5. Compare
	comparison = Comparison(evaluation.fold_results)
	comparison.run()

	# Figures of parts 1 to 5
	visualization = Visualization(exploration, evaluation, stability, comparison)
	visualization.run()


if __name__ == "__main__":
	main()
