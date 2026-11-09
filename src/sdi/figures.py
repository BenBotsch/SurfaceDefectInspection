"""Figures of the README, rendered from ``results/`` and the dataset.

Every figure is written twice, ``<name>.png`` for a light page and ``<name>-dark.png`` for a dark
one, so the README can pick the right one with ``<picture>``. The colours come from a categorical
palette whose first three slots were checked for colour-vision-deficiency separation in both
modes. Lines carry markers and direct labels, so no series is told apart by colour alone.
"""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from sdi.dataset import CLASSES, Sample
from sdi.metrics import Report, abbreviate
from sdi.preprocess import read_gray


@dataclass(frozen=True, slots=True)
class Theme:
    """Colours of one page mode."""

    name: str
    surface: str
    ink: str
    secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, str, str]
    ramp: tuple[str, ...]


LIGHT = Theme(
    name="light",
    surface="#ffffff",
    ink="#0b0b0b",
    secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    ramp=("#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
)
DARK = Theme(
    name="dark",
    surface="#0d1117",
    ink="#f0efec",
    secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70"),
    ramp=("#111a26", "#104281", "#184f95", "#1c5cab", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"),
)
THEMES = (LIGHT, DARK)

# Slot of each model in the palette. A model keeps its colour in every figure.
RESNET, BASELINE, THIRD = 0, 1, 2


@contextmanager
def styled(theme: Theme) -> Iterator[None]:
    """Matplotlib defaults for the theme: recessive axes, hairline grid, ink text."""
    with plt.rc_context(
        {
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "axes.edgecolor": theme.axis,
            "axes.labelcolor": theme.secondary,
            "axes.titlecolor": theme.ink,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": theme.grid,
            "grid.linewidth": 0.8,
            "xtick.color": theme.muted,
            "ytick.color": theme.muted,
            "xtick.labelcolor": theme.secondary,
            "ytick.labelcolor": theme.secondary,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "text.color": theme.ink,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": theme.secondary,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "lines.linewidth": 2,
            "lines.markersize": 5,
        }
    ):
        yield


def save(figure: Figure, directory: Path, name: str, theme: Theme) -> Path:
    """Write the figure as ``name.png`` (light) or ``name-dark.png`` and close it."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}{'' if theme is LIGHT else '-dark'}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15, metadata={"Software": None})
    plt.close(figure)
    return path


def render(
    directory: Path, name: str, draw: Callable[[Theme], Figure], themes: Sequence[Theme] = THEMES
) -> list[Path]:
    """Draw one figure in every theme and save it."""
    paths = []
    for theme in themes:
        with styled(theme):
            paths.append(save(draw(theme), directory, name, theme))
    return paths


def _ink_on(colour: str) -> str:
    r, g, b = (int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return "#0b0b0b" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else "#ffffff"


def _image_axes(ax: Any) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def samples_figure(samples: Sequence[Sample], per_class: int = 3) -> Callable[[Theme], Figure]:
    """A grid with one column per class and ``per_class`` images in each.

    The images are the first of each class in the order given, so the grid is the same on
    every run.
    """
    columns = [[s for s in samples if s.label == c][:per_class] for c in range(len(CLASSES))]

    def draw(theme: Theme) -> Figure:
        figure, axes = plt.subplots(
            per_class,
            len(CLASSES),
            figsize=(1.55 * len(CLASSES), 1.6 * per_class),
            squeeze=False,
        )
        for col, column in enumerate(columns):
            for row, sample in enumerate(column):
                ax = axes[row, col]
                ax.imshow(read_gray(sample.path), cmap="gray", vmin=0, vmax=255)
                _image_axes(ax)
            axes[0, col].set_title(
                CLASSES[col].replace("_", " "), loc="center", fontsize=9.5, color=theme.ink
            )
        figure.subplots_adjust(wspace=0.05, hspace=0.05)
        return figure

    return draw


def confusion_figure(reports: Sequence[Report]) -> Callable[[Theme], Figure]:
    """Row-normalised confusion matrices side by side, with the counts in the cells."""

    def draw(theme: Theme) -> Figure:
        cmap = LinearSegmentedColormap.from_list(f"seq-{theme.name}", theme.ramp)
        figure, axes = plt.subplots(1, len(reports), figsize=(4.2 * len(reports), 4.0))
        for ax, report in zip(np.atleast_1d(axes), reports, strict=True):
            counts = np.asarray(report.confusion)
            rates = counts / counts.sum(axis=1, keepdims=True)
            ax.imshow(rates, cmap=cmap, vmin=0, vmax=1)
            names = [abbreviate(c.name) for c in report.classes]
            ax.set_xticks(range(len(names)), names)
            ax.set_yticks(range(len(names)), names)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for (i, j), count in np.ndenumerate(counts):
                if count:
                    ax.text(
                        j,
                        i,
                        str(count),
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=_ink_on(_hex(cmap(rates[i, j]))),
                    )
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{report.model}  ·  accuracy {report.accuracy:.3f}")
        figure.tight_layout(w_pad=2.5)
        return figure

    return draw


def _hex(rgba: tuple[float, float, float, float]) -> str:
    return "#" + "".join(f"{round(v * 255):02x}" for v in rgba[:3])


def per_class_figure(baseline: Report, model: Report) -> Callable[[Theme], Figure]:
    """Per-class F1 of the baseline and the model as grouped horizontal bars."""

    def draw(theme: Theme) -> Figure:
        figure, ax = plt.subplots(figsize=(6.4, 3.6))
        names = [c.name.replace("_", " ") for c in model.classes]
        y = np.arange(len(names))
        height = 0.36
        for offset, report, slot in ((-1, baseline, BASELINE), (1, model, RESNET)):
            values = [c.f1 for c in report.classes]
            ax.barh(
                y + offset * (height / 2 + 0.01),
                values,
                height=height,
                color=theme.series[slot],
                label=report.model,
                edgecolor=theme.surface,
                linewidth=1,
            )
            for yi, value in zip(y, values, strict=True):
                ax.text(
                    value + 0.01,
                    yi + offset * (height / 2 + 0.01),
                    f"{value:.2f}",
                    va="center",
                    fontsize=7.5,
                    color=theme.secondary,
                )
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.08)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("F1 on the test split")
        ax.set_title("Per-class F1")
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, handlelength=1.2)
        figure.tight_layout()
        return figure

    return draw


def training_figure(histories: dict[int, list[dict[str, float]]]) -> Callable[[Theme], Figure]:
    """Training loss and validation macro F1 per epoch, one line per seed, in two panels."""
    seeds = sorted(histories)[:3]

    def draw(theme: Theme) -> Figure:
        figure, (loss_ax, f1_ax) = plt.subplots(1, 2, figsize=(8.4, 3.2))
        markers = ("o", "s", "^")
        for slot, seed in enumerate(seeds):
            epochs = [h["epoch"] for h in histories[seed]]
            colour = theme.series[slot]
            for ax, key in ((loss_ax, "loss"), (f1_ax, "val_macro_f1")):
                values = [h[key] for h in histories[seed]]
                # The seeds end on almost the same values, so the legend names them instead of
                # direct labels at the line ends. Distinct markers keep them apart without colour.
                ax.plot(epochs, values, color=colour, marker=markers[slot], label=f"seed {seed}")
        loss_ax.set_yscale("log")
        loss_ax.set_title("Training loss")
        loss_ax.set_ylabel("cross-entropy (log scale)")
        f1_ax.set_title("Validation macro F1")
        f1_ax.set_ylim(top=1.005)
        for ax in (loss_ax, f1_ax):
            ax.set_xlabel("epoch")
            ax.yaxis.grid(True)
            ax.set_axisbelow(True)
            ax.set_xlim(0.5, max(len(h) for h in histories.values()) + 0.5)
        f1_ax.legend(loc="lower right", handlelength=1.6)
        figure.tight_layout(w_pad=3)
        return figure

    return draw


@dataclass(frozen=True, slots=True)
class Prediction:
    """A test image with its true class, the predicted class and the probability of it."""

    path: Path
    label: str
    predicted: str
    confidence: float


def errors_figure(errors: Sequence[Prediction]) -> Callable[[Theme], Figure]:
    """The misclassified test images with their true and predicted classes."""

    def draw(theme: Theme) -> Figure:
        figure, axes = plt.subplots(1, len(errors), figsize=(2.3 * len(errors), 2.8))
        for ax, error in zip(np.atleast_1d(axes), errors, strict=True):
            ax.imshow(read_gray(error.path), cmap="gray", vmin=0, vmax=255)
            _image_axes(ax)
            ax.set_title(error.path.stem, loc="center", fontsize=9, color=theme.ink)
            ax.set_xlabel(
                f"true  {error.label}\npredicted  {error.predicted} ({error.confidence:.0%})",
                fontsize=8,
                color=theme.secondary,
                linespacing=1.5,
            )
        figure.tight_layout()
        return figure

    return draw


@dataclass(frozen=True, slots=True)
class Explanation:
    """An image with a Grad-CAM map and the caption of its column."""

    path: Path
    title: str
    caption: str
    cam: npt.NDArray[np.float32]


def gradcam_figure(explanations: Sequence[Explanation]) -> Callable[[Theme], Figure]:
    """Each image above its Grad-CAM overlay, one column per image.

    The map uses ``inferno``, a perceptually uniform sequential colormap, over the image in grey.
    """

    def draw(theme: Theme) -> Figure:
        n, gap, left, right, bottom_margin, top_margin = (
            len(explanations),
            0.06,
            0.01,
            0.99,
            0.13,
            0.9,
        )
        width = 1.45 * n
        # Height at which square images fill the grid without empty bands between the rows.
        cell = width * (right - left) / (n + (n - 1) * gap)
        height = cell * (2 + gap) / (top_margin - bottom_margin)
        figure, axes = plt.subplots(2, n, figsize=(width, height), squeeze=False)
        for col, item in enumerate(explanations):
            size = item.cam.shape[::-1]
            image = cv2.resize(read_gray(item.path), size, interpolation=cv2.INTER_LINEAR)
            top, bottom = axes[0, col], axes[1, col]
            top.imshow(image, cmap="gray", vmin=0, vmax=255)
            bottom.imshow(image, cmap="gray", vmin=0, vmax=255)
            bottom.imshow(item.cam, cmap="inferno", vmin=0, vmax=1, alpha=0.5)
            _image_axes(top)
            _image_axes(bottom)
            top.set_title(item.title, loc="center", fontsize=8.5, color=theme.ink)
            bottom.set_xlabel(item.caption, fontsize=7.5, color=theme.secondary, linespacing=1.4)
        figure.subplots_adjust(
            left=left, right=right, bottom=bottom_margin, top=top_margin, wspace=gap, hspace=gap
        )
        return figure

    return draw


def robustness_figure(data: dict[str, Any]) -> Callable[[Theme], Figure]:
    """Accuracy over corruption level, one panel per corruption, one line per model.

    ``data`` is ``results/robustness.json``. The INT8 line is dashed with its own marker, because
    it often lies on top of the float model.
    """
    models: list[str] = data["models"]
    slots = {"ResNet18": RESNET, "HOG + linear SVM": BASELINE}
    styles = {RESNET: ("o", "-"), BASELINE: ("s", "-"), THIRD: ("^", "--")}

    def draw(theme: Theme) -> Figure:
        panels = data["robustness"]
        figure, axes = plt.subplots(1, len(panels), figsize=(2.15 * len(panels), 3.0), sharey=True)
        for ax, panel in zip(np.atleast_1d(axes), panels, strict=True):
            levels = range(len(panel["strengths"]))
            for name in models:
                slot = slots.get(name, THIRD)
                marker, line = styles[slot]
                ax.plot(
                    levels,
                    panel["accuracy"][name],
                    color=theme.series[slot],
                    marker=marker,
                    linestyle=line,
                    label=name,
                )
            ax.axhline(1 / 6, color=theme.muted, linestyle=":", linewidth=1, zorder=0)
            ax.set_title(panel["name"], fontsize=10)
            ax.set_xticks(list(levels), [f"{s:g}" for s in panel["strengths"]])
            ax.set_xlabel(panel["parameter"])
            ax.set_ylim(0, 1.03)
            ax.yaxis.grid(True)
            ax.set_axisbelow(True)
        first = np.atleast_1d(axes)[0]
        first.set_ylabel("test accuracy")
        first.text(0, 1 / 6 + 0.02, "chance", fontsize=7.5, color=theme.muted)
        handles, labels = first.get_legend_handles_labels()
        figure.legend(
            handles, labels, loc="upper left", bbox_to_anchor=(0.0, 1.08), ncol=len(models)
        )
        figure.tight_layout(w_pad=1.2)
        return figure

    return draw
