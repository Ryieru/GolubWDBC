from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.constants import FIGURES_DIRECTORY


def create_figure(figure_height: float) -> Figure:
	figure_size = (6.3, figure_height)
	figure = Figure(figsize=figure_size, layout="constrained")

	return figure


def style_axes(axes: Axes, title: str) -> None:
	axes.set_title(title)
	axes.grid(color="#dddddd", linewidth=0.6)
	axes.set_axisbelow(True)


def save_figure(figure: Figure, figure_file: Path) -> None:
	FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
	figure.savefig(figure_file, dpi=200)
	print(f"Figure saved to '{figure_file}'")
