"""Image corruptions of increasing strength, and the accuracy of models on corrupted images.

The corruptions stand in for what changes on a production line: sensor noise, defocus, lighting
and compression. Training only augments brightness by up to 15 grey levels and contrast by up
to 15 %, so most of these strengths are outside what the models have seen.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from sdi.preprocess import Image

Apply = Callable[[Image, float, np.random.Generator], Image]
Classify = Callable[[Sequence[Image]], Sequence[int]]


def _to_uint8(values: np.ndarray[Any, Any]) -> Image:
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def gaussian_noise(image: Image, sigma: float, rng: np.random.Generator) -> Image:
    """Add zero-mean Gaussian noise with standard deviation ``sigma`` in grey levels."""
    return _to_uint8(image + rng.normal(0, sigma, image.shape))


def gaussian_blur(image: Image, sigma: float, _: np.random.Generator) -> Image:
    """Blur with a Gaussian kernel of standard deviation ``sigma`` pixels."""
    return np.asarray(cv2.GaussianBlur(image, (0, 0), sigma), dtype=np.uint8)


def brightness(image: Image, offset: float, _: np.random.Generator) -> Image:
    """Add ``offset`` grey levels to every pixel."""
    return _to_uint8(image.astype(np.float32) + offset)


def contrast(image: Image, factor: float, _: np.random.Generator) -> Image:
    """Scale the deviation from the mean grey level by ``factor``."""
    mean = float(image.mean())
    return _to_uint8(mean + factor * (image.astype(np.float32) - mean))


def jpeg(image: Image, quality: float, _: np.random.Generator) -> Image:
    """Encode as JPEG at ``quality`` and decode again."""
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return np.asarray(cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class Corruption:
    """A corruption and its strengths. The first strength is the clean image."""

    name: str
    parameter: str
    strengths: tuple[float, ...]
    apply: Apply

    def __call__(self, image: Image, level: int, rng: np.random.Generator) -> Image:
        """The image corrupted at ``level``, an index into ``strengths``. Level 0 is a copy.

        Raises:
            ValueError: If the level is not an index into ``strengths``.
        """
        if not 0 <= level < len(self.strengths):
            raise ValueError(f"level must be in 0..{len(self.strengths) - 1}, got {level}")
        if level == 0:
            return image.copy()
        return self.apply(image, self.strengths[level], rng)


CORRUPTIONS = (
    Corruption("Gaussian noise", "sigma", (0, 5, 10, 20, 30), gaussian_noise),
    Corruption("Gaussian blur", "sigma (px)", (0, 0.5, 1, 2, 3), gaussian_blur),
    Corruption("Brightness", "offset", (0, 25, 50, 75, 100), brightness),
    Corruption("Contrast", "factor", (1, 0.75, 0.5, 0.35, 0.2), contrast),
    Corruption("JPEG", "quality", (100, 50, 25, 10, 5), jpeg),
)


def evaluate_robustness(
    images: Sequence[Image],
    labels: Sequence[int],
    models: Mapping[str, Classify],
    corruptions: Sequence[Corruption] = CORRUPTIONS,
    seed: int = 0,
) -> dict[str, Any]:
    """Accuracy of every model on every corruption and level.

    The random generator is reseeded per corruption and level, so each model sees the same
    corrupted images and a rerun gives the same numbers.

    Returns:
        Plain data for ``results/robustness.json``.
    """
    truth = np.asarray(labels)
    results = []
    for index, corruption in enumerate(corruptions):
        accuracy: dict[str, list[float]] = {name: [] for name in models}
        for level in range(len(corruption.strengths)):
            rng = np.random.default_rng([seed, index, level])
            corrupted = [corruption(image, level, rng) for image in images]
            for name, classify in models.items():
                predicted = np.asarray(classify(corrupted))
                accuracy[name].append(round(float(np.mean(predicted == truth)), 4))
        results.append(
            {
                "name": corruption.name,
                "parameter": corruption.parameter,
                "strengths": list(corruption.strengths),
                "accuracy": accuracy,
            }
        )
    return {"models": list(models), "images": len(images), "robustness": results}


def _strength(value: float) -> str:
    return f"{value:g}"


def robustness_to_markdown(data: dict[str, Any]) -> str:
    """A Markdown table with one row per corruption and level, one column per model."""
    models = data["models"]
    lines = [
        f"Test accuracy on corrupted copies of the {data['images']} test images. "
        "Level 0 is the clean image.",
        "",
        "| Corruption | Level | Strength | " + " | ".join(models) + " |",
        "| --- | ---: | ---: | " + " | ".join("---:" for _ in models) + " |",
    ]
    for c in data["robustness"]:
        for level, strength in enumerate(c["strengths"]):
            cells = " | ".join(f"{c['accuracy'][m][level]:.3f}" for m in models)
            lines.append(
                f"| {c['name']} | {level} | {c['parameter']} {_strength(strength)} | {cells} |"
            )
    return "\n".join(lines)
