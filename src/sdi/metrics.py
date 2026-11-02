"""Classification metrics as plain data, and their Markdown rendering."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


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
    short = [c.name.split("_")[0][:8] for c in report.classes]
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
