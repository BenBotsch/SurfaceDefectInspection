"""Latency of one image through each runtime, and the table that compares them."""

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Timing:
    """Median and 95th percentile of repeated calls, in milliseconds."""

    median_ms: float
    p95_ms: float


def time_call(
    call: Callable[[], object],
    iterations: int = 200,
    warmup: int = 20,
    clock: Callable[[], float] = time.perf_counter,
) -> Timing:
    """Time ``call`` after ``warmup`` untimed calls, which fill caches and pick kernels.

    Raises:
        ValueError: If ``iterations`` is below 1.
    """
    if iterations < 1:
        raise ValueError("need at least one timed iteration")
    for _ in range(warmup):
        call()
    took = []
    for _ in range(iterations):
        start = clock()
        call()
        took.append((clock() - start) * 1000)
    return Timing(float(np.median(took)), float(np.percentile(took, 95, method="nearest")))


@dataclass(frozen=True, slots=True)
class Row:
    """One runtime of the benchmark. Accuracy is None where it was not measured."""

    runtime: str
    device: str
    accuracy: float | None
    size_mb: float
    median_ms: float
    p95_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Plain data for JSON, rounded to the shown precision."""
        data = asdict(self)
        for key in ("size_mb", "median_ms", "p95_ms"):
            data[key] = round(data[key], 2)
        if data["accuracy"] is not None:
            data["accuracy"] = round(data["accuracy"], 4)
        return data


def benchmark_to_markdown(rows: Sequence[dict[str, Any]], note: str) -> str:
    """A Markdown table of benchmark rows as stored in ``results/benchmark.json``."""
    lines = [
        "| Runtime | Device | Test accuracy | Model size | Median latency | p95 latency |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        accuracy = "n/a" if r["accuracy"] is None else f"{r['accuracy']:.3f}"
        lines.append(
            f"| {r['runtime']} | {r['device']} | {accuracy} | {r['size_mb']:.1f} MB "
            f"| {r['median_ms']:.2f} ms | {r['p95_ms']:.2f} ms |"
        )
    return "\n".join(lines) + f"\n\n{note}"
