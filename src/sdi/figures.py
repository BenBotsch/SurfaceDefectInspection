"""Figures of the README and of web pages, rendered from ``results/`` and the dataset.

The style, the colours and the export come from FigureKit (:mod:`figkit`), the figure style shared
by all of my repositories. Every figure is written for a light page as ``<name>.<ext>`` and for a
dark one as ``<name>-dark.<ext>``, on a transparent background. Charts are SVG and PNG
(:data:`CHART`), photographs WebP and PNG (:data:`PHOTO`). The categorical colours are used in a
fixed order, and lines carry markers and labels, so no series is told apart by colour alone.

The charts of the portfolio exist in English and German as well: each figure reads its texts from
:data:`TEXT` in the language that :func:`figkit.render` draws, and the ``*_captions`` functions
give the title, subtitle and alt text of each language for ``figures.json``.
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
from figkit import Caption, Theme, number
from figkit.language import speaking
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
# Width of a photo grid: the canvas minus its margins, so that the images fill the canvas.
_GRID_WIDTH = figkit.WEB_WIDTH - 2 * figkit.export.PAD

LANGUAGES = ("en", "de")
# The NEU-DET class names stay English in German, as in the German text of the portfolio.
CLASS_LABELS = {
    "en": {c: c.replace("_", " ") for c in CLASSES},
    "de": {
        "crazing": "Crazing",
        "inclusion": "Inclusion",
        "patches": "Patches",
        "pitted_surface": "Pitted Surface",
        "rolled-in_scale": "Rolled-in Scale",
        "scratches": "Scratches",
    },
}
# Names of models and corruptions that differ in German; the others keep their name.
NAMES = {
    "en": {},
    "de": {
        "HOG + linear SVM": "HOG + lineare SVM",
        "Gaussian noise": "Gaußsches Rauschen",
        "Gaussian blur": "Gaußsche Unschärfe",
        "Brightness": "Helligkeit",
        "Contrast": "Kontrast",
        "sigma": "Sigma",
        "sigma (px)": "Sigma (px)",
        "offset": "Versatz",
        "factor": "Faktor",
        "quality": "Qualität",
    },
}
# Every other text of the figures and of figures.json, per language.
TEXT: dict[str, dict[str, str]] = {
    "en": {
        "confusion.panel": "{model}\naccuracy {accuracy}",
        "confusion.predicted": "predicted",
        "confusion.true": "true",
        "confusion.title": "{model} misclassifies {errors} of {total} test images, "
        "the baseline {baseline_errors}",
        "confusion.subtitle": "Row-normalised confusion matrices on the test split, rows are the "
        "true class, columns the predicted one.",
        "confusion.alt": "Two row-normalised confusion matrices for the six classes: {baseline} "
        "with accuracy {baseline_accuracy} and many entries off the diagonal, {model} with "
        "accuracy {accuracy} and {errors} images off the diagonal.",
        "per_class.title": "{model} gains most on {top}",
        "per_class.subtitle": "F1 per class on the test split, largest gain on top.",
        "per_class.chart": "Per-class F1",
        "per_class.x": "F1 on the test split",
        "per_class.alt": "Dumbbell chart of the F1 per class. {baseline} ranges from {low} for "
        "{low_class} to {high} for {high_class}, {model} from {model_low} to {model_high}.",
        "training.loss": "Training loss",
        "training.loss_y": "cross-entropy (log scale)",
        "training.f1": "Validation macro F1",
        "training.epoch": "epoch",
        "training.note": "Line: mean of {seeds} seeds. Band: lowest to highest seed.",
        "training.title": "Every seed reaches its best validation F1 by epoch {epoch}",
        "training.subtitle": "Mean of {seeds} seeds as a line, lowest to highest seed as a band.",
        "training.alt": "Two line charts over {epochs} epochs for {seeds} seeds: the training "
        "loss on a log scale falls from {first_loss} to {last_loss}, the validation macro F1 "
        "reaches {best}.",
        "gradcam.wrong": "misclassified",
        "gradcam.title": "Where ResNet18 looks",
        "gradcam.subtitle": "Images: NEU surface defect database (K. Song and Y. Yan, 2013), "
        "cleaned release KeenForgeAI/NEU-DET-corrected, CC BY 4.0. Downsized, Grad-CAM overlays "
        "added.",
        "gradcam.alt": "{images} test images from NEU-DET, each above the same image with its "
        "Grad-CAM map: one correctly classified image per class with its probability and the "
        "{wrong} misclassified images with their true and predicted class.",
        "robustness.accuracy": "test accuracy",
        "robustness.chance": "chance",
        "robustness.title": "Under corruption, {model} falls from {clean} to as low as {low}",
        "robustness.subtitle": "Test accuracy over {levels} levels of {corruptions} corruptions, "
        "the first level is the unchanged image.",
        "robustness.alt": "{corruptions} line charts, one each for {names}, with the test "
        "accuracy of {models} over {levels} levels. Every curve ends lower than it starts, "
        "{model} from {clean} to as low as {low}.",
    },
    "de": {
        "confusion.panel": "{model}\nAccuracy {accuracy}",
        "confusion.predicted": "vorhergesagt",
        "confusion.true": "wahr",
        "confusion.title": "Das {model} klassifiziert {errors} von {total} Testbildern falsch, "
        "die Baseline {baseline_errors}",
        "confusion.subtitle": "Zeilennormierte Konfusionsmatrizen auf der Testaufteilung, Zeilen "
        "sind die wahre Klasse, Spalten die vorhergesagte.",
        "confusion.alt": "Zwei zeilennormierte Konfusionsmatrizen für die sechs Klassen: "
        "{baseline} mit Accuracy {baseline_accuracy} und vielen Einträgen neben der Diagonale, "
        "{model} mit Accuracy {accuracy} und {errors} Bildern neben der Diagonale.",
        "per_class.title": "Das {model} gewinnt am meisten bei {top}",
        "per_class.subtitle": "F1 je Klasse auf der Testaufteilung, der größte Gewinn oben.",
        "per_class.chart": "F1 je Klasse",
        "per_class.x": "F1 auf der Testaufteilung",
        "per_class.alt": "Hanteldiagramm mit dem F1 je Klasse. {baseline} reicht von {low} für "
        "{low_class} bis {high} für {high_class}, {model} von {model_low} bis {model_high}.",
        "training.loss": "Trainingsverlust",
        "training.loss_y": "Kreuzentropie (log. Skala)",
        "training.f1": "Validierungs-Macro-F1",
        "training.epoch": "Epoche",
        "training.note": "Linie: Mittel von {seeds} Seeds. Band: niedrigster bis höchster Seed.",
        "training.title": "Jeder Seed erreicht seinen besten Validierungs-F1 bis Epoche {epoch}",
        "training.subtitle": "Mittel von {seeds} Seeds als Linie, niedrigster bis höchster Seed "
        "als Band.",
        "training.alt": "Zwei Liniendiagramme über {epochs} Epochen für {seeds} Seeds: Der "
        "Trainingsverlust auf logarithmischer Skala fällt von {first_loss} auf {last_loss}, der "
        "Validierungs-Macro-F1 erreicht {best}.",
        "gradcam.wrong": "falsch erkannt",
        "gradcam.title": "Worauf das ResNet18 schaut",
        "gradcam.subtitle": "Bilder: NEU surface defect database (K. Song und Y. Yan, 2013), "
        "bereinigte Fassung KeenForgeAI/NEU-DET-corrected, CC BY 4.0. Verkleinert und mit "
        "Grad-CAM überlagert.",
        "gradcam.alt": "{images} Testbilder aus NEU-DET, jeweils über derselben Aufnahme mit "
        "ihrer Grad-CAM-Karte: je ein richtig erkanntes Bild pro Klasse mit seiner "
        "Wahrscheinlichkeit und die {wrong} falsch erkannten Bilder mit wahrer und "
        "vorhergesagter Klasse.",
        "robustness.accuracy": "Test-Accuracy",
        "robustness.chance": "Zufall",
        "robustness.title": "Unter Bildstörungen fällt das {model} von {clean} auf bis zu {low}",
        "robustness.subtitle": "Test-Accuracy über {levels} Stufen von {corruptions} "
        "Bildstörungen, die erste Stufe ist das unveränderte Bild.",
        "robustness.alt": "{corruptions} Liniendiagramme, je eines für {names}, mit der "
        "Test-Accuracy von {models} über {levels} Stufen. Jede Kurve endet tiefer, als sie "
        "beginnt, das {model} von {clean} auf bis zu {low}.",
    },
}


def text(key: str, **values: object) -> str:
    """Text ``key`` in the language being drawn, filled with ``values``."""
    return TEXT[figkit.language()][key].format(**values)


def name(value: str) -> str:
    """A model, corruption or parameter name in the language being drawn."""
    return NAMES[figkit.language()].get(value, value)


def class_label(value: str) -> str:
    """A class name in the language being drawn."""
    return CLASS_LABELS[figkit.language()].get(value, value)


def _join(items: Sequence[str]) -> str:
    """``a, b and c`` in the language being drawn."""
    word = {"en": "and", "de": "und"}[figkit.language()]
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} {word} {items[-1]}"


def _captions(key: str, values: Callable[[], dict[str, object]]) -> dict[str, Caption]:
    """Title, subtitle and alt text of ``key`` per language, ``values`` filled in each."""
    result = {}
    for code in LANGUAGES:
        with speaking(code):
            fill = values()
            result[code] = Caption(
                title=text(f"{key}.title", **fill),
                subtitle=text(f"{key}.subtitle", **fill),
                alt=text(f"{key}.alt", **fill),
            )
    return result


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
    gap, title = 0.05, 0.45
    # Square cells that fill the width; the titles get a fixed band above the grid.
    cell = _GRID_WIDTH / (len(CLASSES) + (len(CLASSES) - 1) * gap)
    grid_height = cell * (per_class + (per_class - 1) * gap)

    def draw(theme: Theme) -> Figure:
        figure, axes = plt.subplots(
            per_class,
            len(CLASSES),
            figsize=(_GRID_WIDTH, grid_height + title),
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
        figure.subplots_adjust(
            left=0,
            right=1,
            bottom=0,
            top=grid_height / (grid_height + title),
            wspace=gap,
            hspace=gap,
        )
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
            ax.set_xlabel(text("confusion.predicted"))
            ax.set_ylabel(text("confusion.true"))
            # Two lines, so that the titles of neighbouring panels do not meet on narrow pages.
            panel = text(
                "confusion.panel",
                model=name(report.model),
                accuracy=number(report.accuracy, ".3f"),
            )
            ax.set_title(panel, linespacing=1.4)
        figure.set_layout_engine("tight", w_pad=2.5)
        return figure

    return draw


def per_class_figure(baseline: Report, model: Report) -> Draw:
    """Per-class F1 of the baseline and the model as a dumbbell chart, largest gain on top.

    Each class is one row: a dot for each model and a line between them, so the gain reads as a
    length. The two models also differ in marker shape, so they are not told apart by colour alone.
    """
    before = {c.name: c.f1 for c in baseline.classes}
    rows = sorted(model.classes, key=lambda c: c.f1 - before[c.name], reverse=True)

    def draw(theme: Theme) -> Figure:
        figure, ax = plt.subplots(figsize=(figkit.WEB_WIDTH, 3.3))
        y = np.arange(len(rows))
        low = [before[c.name] for c in rows]
        high = [c.f1 for c in rows]
        ax.hlines(y, low, high, color=theme.grid, linewidth=3, zorder=1)
        for values, report, slot, marker in (
            (low, baseline, BASELINE, "s"),
            (high, model, RESNET, "o"),
        ):
            ax.plot(
                values,
                y,
                linestyle="none",
                marker=marker,
                markersize=8,
                color=theme.series[slot],
                label=name(report.model),
                zorder=2,
            )
        for row, (a, b) in enumerate(zip(low, high, strict=True)):
            # The smaller value is labelled on the left of its dot, the larger on the right.
            left, right = min(a, b), max(a, b)
            figkit.value(ax, left - 0.025, row, number(left, ".2f"), theme, ha="right", va="center")
            figkit.value(
                ax, right + 0.025, row, number(right, ".2f"), theme, ha="left", va="center"
            )
        ax.set_yticks(y, [class_label(c.name) for c in rows])
        ax.invert_yaxis()
        # The axis starts one tick of 0.2 below the lowest value, leaving room for its label.
        start = max(0.0, math.floor(min(low + high) * 5) / 5 - 0.2)
        ax.set_xlim(start, 1.1)
        ax.set_xticks(np.arange(start, 1.0001, 0.2))
        ax.spines["bottom"].set_bounds(start, 1)
        figkit.grid(ax, "x")
        figkit.category_axis(ax, "y")
        ax.set_xlabel(text("per_class.x"))
        figkit.title(ax, text("per_class.chart"), theme)
        figkit.legend_top(ax)
        figure.set_layout_engine("tight")
        return figure

    return draw


def training_figure(histories: dict[int, list[dict[str, float]]]) -> Draw:
    """Training loss and validation macro F1 per epoch in two panels.

    The seeds end on almost the same values, so one line shows their mean and a band the range
    from the lowest to the highest seed, instead of three lines on top of each other.
    """
    seeds = sorted(histories)
    epochs = min(len(histories[s]) for s in seeds)

    def series(key: str) -> npt.NDArray[np.float64]:
        return np.array([[h[key] for h in histories[s][:epochs]] for s in seeds], dtype=float)

    def draw(theme: Theme) -> Figure:
        figure, (loss_ax, f1_ax) = plt.subplots(1, 2, figsize=(figkit.WEB_WIDTH, 2.9))
        x = np.arange(1, epochs + 1)
        colour = theme.series[RESNET]
        for ax, key in ((loss_ax, "loss"), (f1_ax, "val_macro_f1")):
            values = series(key)
            # The mean of a log-scaled loss is the geometric mean, so the line sits inside the band.
            mean = np.exp(np.log(values).mean(axis=0)) if key == "loss" else values.mean(axis=0)
            ax.fill_between(
                x, values.min(axis=0), values.max(axis=0), color=colour, alpha=0.18, linewidth=0
            )
            ax.plot(x, mean, color=colour)
        loss_ax.set_yscale("log")
        loss_ax.set_title(text("training.loss"))
        loss_ax.set_ylabel(text("training.loss_y"))
        f1_ax.set_title(text("training.f1"))
        f1_ax.set_ylim(top=1.005)
        for ax in (loss_ax, f1_ax):
            ax.set_xlabel(text("training.epoch"))
            ax.yaxis.grid(True)
            ax.set_axisbelow(True)
            ax.set_xlim(1, epochs)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        figure.text(
            0.01,
            0.0,
            text("training.note", seeds=len(seeds)),
            fontsize=_SIZE["annotation"],
            color=theme.secondary,
            va="top",
        )
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
    """A test image with its Grad-CAM map for the predicted class.

    The figure writes the texts of its column in the language it draws: a correct image shows its
    class and ``probability``, a misclassified one its true and predicted class.
    """

    path: Path
    label: str
    predicted: str
    probability: float
    cam: npt.NDArray[np.float32]

    @property
    def wrong(self) -> bool:
        """Whether the model predicted another class than the true one."""
        return self.label != self.predicted

    def title(self) -> str:
        """Heading of the column."""
        return text("gradcam.wrong") if self.wrong else class_label(self.label)

    def caption(self) -> str:
        """Line below the overlay."""
        if self.wrong:
            return f"{class_label(self.label)}\n→ {class_label(self.predicted)}"
        return f"p = {number(self.probability, '.2f')}"


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
        width = _GRID_WIDTH
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
                top.set_title(item.title(), loc="center", fontsize=_SIZE["tick"], color=theme.ink)
                bottom.set_xlabel(
                    item.caption(),
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
            for model in models:
                slot = slots.get(model, THIRD)
                marker, line = styles[slot]
                ax.plot(
                    levels,
                    panel["accuracy"][model],
                    color=theme.series[slot],
                    marker=marker,
                    linestyle=line,
                    label=name(model),
                )
            ax.axhline(1 / 6, color=theme.muted, linestyle=":", linewidth=1, zorder=0)
            ax.set_title(name(panel["name"]), fontsize=_SIZE["label"] + 1)
            ax.set_xticks(list(levels), [number(s, "g") for s in panel["strengths"]])
            ax.set_xlabel(name(panel["parameter"]))
            ax.set_ylim(0, 1.03)
            ax.yaxis.grid(True)
            ax.set_axisbelow(True)
        for row in grid:
            row[0].set_ylabel(text("robustness.accuracy"))
        first = axes[0]
        first.text(
            0,
            1 / 6 + 0.02,
            text("robustness.chance"),
            fontsize=_SIZE["annotation"],
            color=theme.secondary,
        )
        handles, _ = first.get_legend_handles_labels()
        # The legend lists the models from the highest to the lowest mean accuracy, the order in
        # which their lines run through the panels. The handles follow the order of ``models``.
        mean = [np.mean([np.mean(p["accuracy"][m]) for p in panels]) for m in models]
        order = sorted(range(len(models)), key=lambda i: -mean[i])
        handles = [handles[i] for i in order]
        labels = [name(models[i]) for i in order]
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


def confusion_captions(baseline: Report, model: Report) -> dict[str, Caption]:
    """Texts of :func:`confusion_figure` per language."""

    def errors(report: Report) -> int:
        counts = np.asarray(report.confusion)
        return int(counts.sum() - np.trace(counts))

    return _captions(
        "confusion",
        lambda: {
            "model": name(model.model),
            "baseline": name(baseline.model),
            "errors": number(errors(model), "d"),
            "baseline_errors": number(errors(baseline), "d"),
            "total": number(int(np.asarray(model.confusion).sum()), ",d"),
            "accuracy": number(model.accuracy, ".3f"),
            "baseline_accuracy": number(baseline.accuracy, ".3f"),
        },
    )


def per_class_captions(baseline: Report, model: Report) -> dict[str, Caption]:
    """Texts of :func:`per_class_figure` per language."""
    before = {c.name: c.f1 for c in baseline.classes}
    top = max(model.classes, key=lambda c: c.f1 - before[c.name]).name
    low, high = min(before, key=before.__getitem__), max(before, key=before.__getitem__)
    after = [c.f1 for c in model.classes]
    return _captions(
        "per_class",
        lambda: {
            "model": name(model.model),
            "baseline": name(baseline.model),
            "top": class_label(top),
            "low": number(before[low], ".2f"),
            "low_class": class_label(low),
            "high": number(before[high], ".2f"),
            "high_class": class_label(high),
            "model_low": number(min(after), ".2f"),
            "model_high": number(max(after), ".2f"),
        },
    )


def training_captions(histories: dict[int, list[dict[str, float]]]) -> dict[str, Caption]:
    """Texts of :func:`training_figure` per language, from the same epochs as the figure."""
    seeds = sorted(histories)
    epochs = min(len(histories[s]) for s in seeds)
    f1 = np.array([[h["val_macro_f1"] for h in histories[s][:epochs]] for s in seeds])
    loss = np.array([[h["loss"] for h in histories[s][:epochs]] for s in seeds])
    mean_loss = np.exp(np.log(loss).mean(axis=0))
    # The first epoch at which each seed reaches its best F1; the title names the latest of them.
    epoch = int(max(np.argmax(row) for row in f1)) + 1
    return _captions(
        "training",
        lambda: {
            "epoch": epoch,
            "epochs": epochs,
            "seeds": len(seeds),
            "first_loss": number(mean_loss[0], ".2f"),
            "last_loss": number(mean_loss[-1], ".4f"),
            "best": number(f1.mean(axis=0).max(), ".3f"),
        },
    )


def gradcam_captions(images: int, wrong: int) -> dict[str, Caption]:
    """Texts of :func:`gradcam_figure` per language, for ``images`` of which ``wrong`` are wrong."""
    return _captions("gradcam", lambda: {"images": images, "wrong": wrong})


def robustness_captions(data: dict[str, Any], model: str = "ResNet18") -> dict[str, Caption]:
    """Texts of :func:`robustness_figure` per language, the title about ``model``."""
    panels = data["robustness"]
    curves = [p["accuracy"][model] for p in panels]

    def corruption(value: str) -> str:
        # In English the names head a panel but sit inside a sentence in the alt text.
        local = name(value)
        keep = figkit.language() == "de" or local.split()[0] in {"Gaussian", "JPEG"}
        return local if keep else local.lower()

    return _captions(
        "robustness",
        lambda: {
            "model": name(model),
            "clean": number(curves[0][0], ".3f"),
            "low": number(min(min(c) for c in curves), ".3f"),
            "levels": len(panels[0]["strengths"]),
            "corruptions": len(panels),
            "names": _join([corruption(p["name"]) for p in panels]),
            "models": _join([name(m) for m in data["models"]]),
        },
    )
