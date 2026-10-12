from src.dataset.files import load_dataset
from src.exploration.exploration import Exploration
from src.evaluation.evaluation import Evaluation


def main() -> None:
	dataset = load_dataset()

	exploration = Exploration(dataset)
	exploration.run()

	evaluation = Evaluation(dataset)
	evaluation.run()


if __name__ == "__main__":
	main()
