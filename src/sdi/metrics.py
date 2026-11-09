"""Classification metrics as plain data, and their Markdown rendering."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Abbreviations used for NEU-DET in the literature, for compact table and figure headers.
ABBREVIATIONS = {
    "crazing": "Cr",
    "inclusion": "In",
    "patches": "Pa",
    "pitted_surface": "PS",
    "rolled-in_scale": "RS",
    "scratches": "Sc",
}


def abbreviate(name: str) -> str:
    """The NEU-DET abbreviation of a class, or the name itself for other classes."""
    return ABBREVIATIONS.get(name, name)


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """Precision, recall and F1 of one class with its number of test images."""

    name: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class Report:
    """Metrics of one model on one split."""

    model: str
    split: str
    accuracy: float
    macro_f1: float
    classes: tuple[ClassMetrics, ...]
    confusion: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        """The report as JSON-ready data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Report":
        """A report from :meth:`to_dict` output; keys it does not know are ignored."""
        return cls(
            model=data["model"],
            split=data["split"],
            accuracy=data["accuracy"],
            macro_f1=data["macro_f1"],
            classes=tuple(ClassMetrics(**c) for c in data["classes"]),
            confusion=tuple(tuple(row) for row in data["confusion"]),
        )


def evaluate(
    model: str,
    split: str,
    names: Sequence[str],
    truth: Sequence[int],
    predicted: Sequence[int],
) -> Report:
    """Compute the report from true and predicted class indices."""
    labels = list(range(len(names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(truth, predicted, labels=labels)
    return Report(
        model=model,
        split=split,
        accuracy=float(np.mean(np.asarray(truth) == np.asarray(predicted))),
        macro_f1=float(np.mean(f1)),
        classes=tuple(
            ClassMetrics(n, float(p), float(r), float(f), int(s))
            for n, p, r, f, s in zip(names, precision, recall, f1, support, strict=True)
        ),
        confusion=tuple(tuple(int(v) for v in row) for row in matrix),
    )


def to_markdown(report: Report) -> str:
    """Per-class table and confusion matrix of a report."""
    lines = [
        f"**{report.model}** on {report.split}: accuracy {report.accuracy:.3f}, "
        f"macro F1 {report.macro_f1:.3f}",
        "",
        "| Class | Precision | Recall | F1 | Images |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {c.name} | {c.precision:.3f} | {c.recall:.3f} | {c.f1:.3f} | {c.support} |"
        for c in report.classes
    ]
    short = [abbreviate(c.name) for c in report.classes]
    lines += ["", "Confusion matrix, rows true, columns predicted:", ""]
    lines += [
        "| | " + " | ".join(short) + " |",
        "| --- " + "| ---: " * len(short) + "|",
    ]
    lines += [
        f"| {name} | " + " | ".join(str(v) for v in row) + " |"
        for name, row in zip(short, report.confusion, strict=True)
    ]
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class SeedSummary:
    """Accuracy and macro F1 of the same model trained with different seeds."""

    seeds: tuple[int, ...]
    accuracy: tuple[float, ...]
    macro_f1: tuple[float, ...]

    @property
    def accuracy_mean(self) -> float:
        """Mean accuracy over the seeds."""
        return float(np.mean(self.accuracy))

    @property
    def accuracy_std(self) -> float:
        """Sample standard deviation of the accuracy, 0 for a single seed."""
        return _std(self.accuracy)

    @property
    def macro_f1_mean(self) -> float:
        """Mean macro F1 over the seeds."""
        return float(np.mean(self.macro_f1))

    @property
    def macro_f1_std(self) -> float:
        """Sample standard deviation of the macro F1, 0 for a single seed."""
        return _std(self.macro_f1)

    def to_dict(self) -> dict[str, Any]:
        """The summary with its means and standard deviations as JSON-ready data."""
        return asdict(self) | {
            "accuracy_mean": self.accuracy_mean,
            "accuracy_std": self.accuracy_std,
            "macro_f1_mean": self.macro_f1_mean,
            "macro_f1_std": self.macro_f1_std,
        }


def _std(values: Sequence[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def summarize_seeds(seeds: Sequence[int], reports: Sequence[Report]) -> SeedSummary:
    """Collect the reports of one model trained with the given seeds."""
    if not reports or len(seeds) != len(reports):
        raise ValueError("need one report per seed and at least one seed")
    return SeedSummary(
        seeds=tuple(seeds),
        accuracy=tuple(r.accuracy for r in reports),
        macro_f1=tuple(r.macro_f1 for r in reports),
    )


def seeds_to_markdown(model: str, summary: SeedSummary) -> str:
    """Per-seed table with mean and sample standard deviation."""
    lines = [
        f"**{model}** on test over {len(summary.seeds)} seeds",
        "",
        "| Seed | Accuracy | Macro F1 |",
        "| ---: | ---: | ---: |",
    ]
    lines += [
        f"| {seed} | {acc:.3f} | {f1:.3f} |"
        for seed, acc, f1 in zip(summary.seeds, summary.accuracy, summary.macro_f1, strict=True)
    ]
    lines.append(
        f"| mean ± std | {summary.accuracy_mean:.3f} ± {summary.accuracy_std:.3f} "
        f"| {summary.macro_f1_mean:.3f} ± {summary.macro_f1_std:.3f} |"
    )
    return "\n".join(lines)
