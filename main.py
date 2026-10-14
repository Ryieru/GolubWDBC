from src.dataset.files import load_dataset
from src.exploration.exploration import Exploration
from src.evaluation.evaluation import Evaluation
from src.stability.stability import Stability
from src.comparison.comparison import Comparison


def main() -> None:
	dataset = load_dataset()

	exploration = Exploration(dataset)
	exploration.run()

	evaluation = Evaluation(dataset)
	evaluation.run()

	stability = Stability(dataset, evaluation.fold_results)
	stability.run()

	comparison = Comparison(evaluation.fold_results)
	comparison.run()


if __name__ == "__main__":
	main()
