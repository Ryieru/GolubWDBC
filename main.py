from src.dataset.files import load_dataset
from src.exploration.exploration import Exploration


def main() -> None:
	dataset = load_dataset()

	exploration = Exploration(dataset)
	exploration.run()


if __name__ == "__main__":
	main()
