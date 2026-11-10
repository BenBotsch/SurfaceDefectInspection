"""Figures of the README and of web pages, rendered from ``results/`` and the dataset.

The style, the colours and the export come from FigureKit (:mod:`figkit`), the figure style shared
by all of my repositories. Every figure is written for a light page as ``<name>.<ext>`` and for a
dark one as ``<name>-dark.<ext>``, on a transparent background. Charts are SVG and PNG
(:data:`CHART`), photographs WebP and PNG (:data:`PHOTO`). The categorical colours are used in a
fixed order, and lines carry markers and labels, so no series is told apart by colour alone.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import figkit
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from figkit import Theme
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from sdi.dataset import CLASSES, Sample
from sdi.metrics import Report, abbreviate
from sdi.preprocess import read_gray

Draw = Callable[[Theme], Figure]

CHART = figkit.WEB
PHOTO = figkit.PHOTO

# Slot of each model in the palette. A model keeps its colour in every figure.
RESNET, BASELINE, THIRD = 0, 1, 2

_SIZE = figkit.tokens()["font"]


def _image_axes(ax: Any) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def samples_figure(samples: Sequence[Sample], per_class: int = 3) -> Draw:
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
            # Two-word names take two lines, so that neighbouring titles do not meet.
            axes[0, col].set_title(
                CLASSES[col].replace("_", "\n"),
                loc="center",
                fontsize=_SIZE["label"],
                color=theme.ink,
                linespacing=1.2,
            )
        figure.subplots_adjust(wspace=0.05, hspace=0.05)
        return figure

    return draw


def confusion_figure(reports: Sequence[Report]) -> Draw:
    """Row-normalised confusion matrices side by side, with the counts in the cells."""

    def draw(theme: Theme) -> Figure:
        cmap = LinearSegmentedColormap.from_list(f"seq-{theme.name}", theme.ramp)
        figure, axes = plt.subplots(1, len(reports), figsize=(3.2 * len(reports), 3.3))
        for ax, report in zip(np.atleast_1d(axes), reports, strict=True):
            counts = np.asarray(report.confusion)
            rates = counts / counts.sum(axis=1, keepdims=True)
            ax.imshow(rates, cmap=cmap, vmin=0, vmax=1, interpolation="none")
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
                        fontsize=_SIZE["annotation"],
                        color=figkit.ink_on(figkit.hex_colour(cmap(rates[i, j]))),
                        fontfamily=figkit.mono(),
                    )
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            # Two lines, so that the titles of neighbouring panels do not meet on narrow pages.
            ax.set_title(f"{report.model}\naccuracy {report.accuracy:.3f}", linespacing=1.4)
        figure.set_layout_engine("tight", w_pad=2.5)
        return figure

    return draw


def per_class_figure(baseline: Report, model: Report) -> Draw:
    """Per-class F1 of the baseline and the model as grouped horizontal bars."""

    def draw(theme: Theme) -> Figure:
        figure, ax = plt.subplots(figsize=(6.4, 3.6))
        names = [c.name.replace("_", " ") for c in model.classes]
        y = np.arange(len(names))
        height = 0.36
        for offset, report, slot in ((-1, baseline, BASELINE), (1, model, RESNET)):
            values = [c.f1 for c in report.classes]
            # The offset leaves a thin gap between the two bars of a class.
            bars = ax.barh(
                y + offset * (height / 2 + 0.02),
                values,
                height=height,
                color=theme.series[slot],
                label=report.model,
            )
            figkit.bar_labels(ax, bars, theme)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.08)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.spines["bottom"].set_bounds(0, 1)
        figkit.grid(ax, "x")
        figkit.category_axis(ax, "y")
        ax.set_xlabel("F1 on the test split")
        ax.set_title("Per-class F1")
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, handlelength=1.2)
        figure.set_layout_engine("tight")
        return figure

    return draw


def training_figure(histories: dict[int, list[dict[str, float]]]) -> Draw:
    """Training loss and validation macro F1 per epoch, one line per seed, in two panels."""
    seeds = sorted(histories)[:3]

    def draw(theme: Theme) -> Figure:
        figure, (loss_ax, f1_ax) = plt.subplots(1, 2, figsize=(figkit.WEB_WIDTH, 2.9))
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
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        f1_ax.legend(loc="lower right", handlelength=1.6)
        figure.set_layout_engine("tight", w_pad=3)
        return figure

    return draw


@dataclass(frozen=True, slots=True)
class Prediction:
    """A test image with its true class, the predicted class and the probability of it."""

    path: Path
    label: str
    predicted: str
    confidence: float


def errors_figure(errors: Sequence[Prediction]) -> Draw:
    """The misclassified test images with their true and predicted classes."""

    def draw(theme: Theme) -> Figure:
        figure, axes = plt.subplots(1, len(errors), figsize=(2.3 * len(errors), 2.8))
        for ax, error in zip(np.atleast_1d(axes), errors, strict=True):
            ax.imshow(read_gray(error.path), cmap="gray", vmin=0, vmax=255)
            _image_axes(ax)
            ax.set_title(error.path.stem, loc="center", fontsize=_SIZE["label"], color=theme.ink)
            ax.set_xlabel(
                f"true  {error.label}\npredicted  {error.predicted} ({error.confidence:.0%})",
                fontsize=_SIZE["annotation"],
                color=theme.secondary,
                linespacing=1.5,
            )
        figure.set_layout_engine("tight")
        return figure

    return draw


@dataclass(frozen=True, slots=True)
class Explanation:
    """An image with a Grad-CAM map and the caption of its column."""

    path: Path
    title: str
    caption: str
    cam: npt.NDArray[np.float32]


def gradcam_figure(explanations: Sequence[Explanation], columns: int | None = None) -> Draw:
    """Each image above its Grad-CAM overlay, one column per image.

    With ``columns``, the images wrap into rows of that many, for narrow pages. The map uses
    ``inferno``, a perceptually uniform sequential colormap, over the image in grey.
    """
    n = len(explanations)
    per_row = min(columns or n, n)
    rows = math.ceil(n / per_row)

    def draw(theme: Theme) -> Figure:
        gap, left, right, bottom_margin, top_margin = 0.06, 0.01, 0.99, 0.13, 0.9
        width = 1.45 * per_row
        # Height at which square images fill the grid without empty bands between the rows.
        cell = width * (right - left) / (per_row + (per_row - 1) * gap)
        height = cell * (2 + gap) / (top_margin - bottom_margin)
        figure = plt.figure(figsize=(width, height * rows))
        bands = figure.subfigures(rows, 1, squeeze=False)
        for band_index, band in enumerate(bands.flat):
            axes = band.subplots(2, per_row, squeeze=False)
            items = explanations[band_index * per_row : (band_index + 1) * per_row]
            for col in range(per_row):
                top, bottom = axes[0, col], axes[1, col]
                _image_axes(top)
                _image_axes(bottom)
                if col >= len(items):
                    top.set_visible(False)
                    bottom.set_visible(False)
                    continue
                item = items[col]
                size = item.cam.shape[::-1]
                image = cv2.resize(read_gray(item.path), size, interpolation=cv2.INTER_LINEAR)
                top.imshow(image, cmap="gray", vmin=0, vmax=255)
                bottom.imshow(image, cmap="gray", vmin=0, vmax=255)
                bottom.imshow(item.cam, cmap="inferno", vmin=0, vmax=1, alpha=0.5)
                top.set_title(item.title, loc="center", fontsize=_SIZE["tick"], color=theme.ink)
                bottom.set_xlabel(
                    item.caption,
                    fontsize=_SIZE["annotation"],
                    color=theme.secondary,
                    linespacing=1.4,
                )
            band.set_facecolor(theme.surface)
            band.subplots_adjust(
                left=left, right=right, bottom=bottom_margin, top=top_margin, wspace=gap, hspace=gap
            )
        return figure

    return draw


def robustness_figure(data: dict[str, Any], columns: int | None = None) -> Draw:
    """Accuracy over corruption level, one panel per corruption, one line per model.

    ``data`` is ``results/robustness.json``. The INT8 line is dashed with its own marker, because
    it often lies on top of the float model. With ``columns``, the panels wrap into a grid, and
    the legend takes the first free cell if there is one.
    """
    models: list[str] = data["models"]
    slots = {"ResNet18": RESNET, "HOG + linear SVM": BASELINE}
    styles = {RESNET: ("o", "-"), BASELINE: ("s", "-"), THIRD: ("^", "--")}

    def draw(theme: Theme) -> Figure:
        panels = data["robustness"]
        per_row = min(columns or len(panels), len(panels))
        rows = math.ceil(len(panels) / per_row)
        figure, grid = plt.subplots(
            rows, per_row, figsize=(2.15 * per_row, 3.0 * rows), sharey=True, squeeze=False
        )
        cells = list(grid.flat)
        axes, spare = cells[: len(panels)], cells[len(panels) :]
        for ax, panel in zip(axes, panels, strict=True):
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
            ax.set_title(panel["name"], fontsize=_SIZE["label"] + 1)
            ax.set_xticks(list(levels), [f"{s:g}" for s in panel["strengths"]])
            ax.set_xlabel(panel["parameter"])
            ax.set_ylim(0, 1.03)
            ax.yaxis.grid(True)
            ax.set_axisbelow(True)
        for row in grid:
            row[0].set_ylabel("test accuracy")
        first = axes[0]
        first.text(0, 1 / 6 + 0.02, "chance", fontsize=_SIZE["annotation"], color=theme.secondary)
        handles, labels = first.get_legend_handles_labels()
        for ax in spare:
            ax.axis("off")
        if spare:
            spare[0].legend(handles, labels, loc="center", handlelength=2.4, labelspacing=1.0)
        else:
            figure.legend(
                handles, labels, loc="upper left", bbox_to_anchor=(0.0, 1.08), ncol=len(models)
            )
        figure.set_layout_engine("tight", w_pad=1.2, h_pad=1.5)
        return figure

    return draw
