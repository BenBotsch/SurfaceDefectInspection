from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from sdi import figures
from sdi.cli import render
from sdi.preprocess import Image
from sdi.robustness import CORRUPTIONS, Corruption, evaluate_robustness, robustness_to_markdown


def image() -> Image:
    rng = np.random.default_rng(0)
    return rng.integers(30, 220, (200, 200), dtype=np.uint8)


@pytest.mark.parametrize("corruption", CORRUPTIONS, ids=lambda c: c.name)
def test_corruptions_keep_shape_and_type_and_level_zero_is_clean(corruption: Corruption) -> None:
    original = image()

    clean = corruption(original, 0, np.random.default_rng(0))
    strongest = corruption(original, len(corruption.strengths) - 1, np.random.default_rng(0))

    assert np.array_equal(clean, original)
    assert clean is not original
    assert strongest.shape == original.shape
    assert strongest.dtype == np.uint8
    assert not np.array_equal(strongest, original)


@pytest.mark.parametrize("corruption", CORRUPTIONS, ids=lambda c: c.name)
def test_corruptions_grow_stronger_with_the_level(corruption: Corruption) -> None:
    original = image().astype(np.float32)
    changes = [
        np.abs(corruption(image(), level, np.random.default_rng(0)) - original).mean()
        for level in range(len(corruption.strengths))
    ]

    assert changes == sorted(changes)


def test_level_outside_the_strengths_is_rejected() -> None:
    with pytest.raises(ValueError, match="level"):
        CORRUPTIONS[0](image(), len(CORRUPTIONS[0].strengths), np.random.default_rng(0))


def test_evaluation_is_deterministic_and_renders(tmp_path: Path) -> None:
    images = [image(), 255 - image()]
    labels = [0, 1]

    def bright_is_one(batch: Sequence[Image]) -> list[int]:
        return [int(i.mean() > 127) for i in batch]

    models = {"always zero": lambda batch: [0] * len(batch), "threshold": bright_is_one}
    first = evaluate_robustness(images, labels, models, CORRUPTIONS[:2])
    second = evaluate_robustness(images, labels, models, CORRUPTIONS[:2])

    assert first == second
    assert first["robustness"][0]["accuracy"]["always zero"] == [0.5] * 5
    assert first["robustness"][0]["accuracy"]["threshold"][0] == 1.0
    markdown = robustness_to_markdown(first)
    assert render(first) == markdown
    assert "| Gaussian noise | 0 | sigma 0 | 0.500 | 1.000 |" in markdown
    paths = figures.render(tmp_path, "robustness", figures.robustness_figure(first))
    assert [p.name for p in paths] == ["robustness.png", "robustness-dark.png"]
