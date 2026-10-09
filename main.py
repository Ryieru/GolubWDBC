from src.dataset.files import load_dataset
from src.exploration.exploration import Exploration
from src.weighted_voting.weighted_voting import WeightedVoting


def main() -> None:
	dataset = load_dataset()

	exploration = Exploration(dataset)
	exploration.run()

	weighted_voting = WeightedVoting(dataset)
	weighted_voting.print()


if __name__ == "__main__":
	main()
