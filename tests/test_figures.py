from pathlib import Path

import cv2
import figkit
import numpy as np
from figkit.check import check
from PIL import Image

from sdi import figures
from sdi.dataset import CLASSES, Sample
from sdi.metrics import evaluate


def _image(tmp_path: Path) -> Path:
    image = tmp_path / "crazing_1.png"
    cv2.imwrite(str(image), np.full((200, 200), 128, dtype=np.uint8))
    return image


def _explanations(tmp_path: Path, count: int) -> list[figures.Explanation]:
    cam = np.eye(8, dtype=np.float32)
    return [figures.Explanation(_image(tmp_path), "a", "p = 1", cam)] * count


def _robustness(panels: int) -> dict[str, object]:
    return {
        "models": ["ResNet18", "HOG + linear SVM", "ResNet18 INT8"],
        "robustness": [
            {
                "name": f"c{i}",
                "parameter": "p",
                "strengths": [0, 1],
                "accuracy": {
                    "ResNet18": [1.0, 0.5],
                    "HOG + linear SVM": [0.6, 0.3],
                    "ResNet18 INT8": [1.0, 0.4],
                },
            }
            for i in range(panels)
        ],
    }


def test_every_figure_is_written_transparent_for_light_and_dark_pages(tmp_path: Path) -> None:
    image = _image(tmp_path)
    samples = [Sample(image, label) for label in range(len(CLASSES))]
    labels = list(range(len(CLASSES)))
    report = evaluate("m", "test", CLASSES, labels, labels[::-1])
    history = [{"epoch": 1, "loss": 0.5, "val_macro_f1": 0.8}]
    jobs = {
        "samples": (figures.samples_figure(samples, per_class=1), figures.PHOTO),
        "confusion": (figures.confusion_figure([report, report]), figures.CHART),
        "per_class_f1": (figures.per_class_figure(report, report), figures.CHART),
        "training": (figures.training_figure({0: history, 1: history}), figures.CHART),
        "errors": (
            figures.errors_figure([figures.Prediction(image, "a", "b", 0.6)] * 2),
            figures.PHOTO,
        ),
        "gradcam": (figures.gradcam_figure(_explanations(tmp_path, 2)), figures.PHOTO),
        "robustness": (figures.robustness_figure(_robustness(5), columns=3), figures.CHART),
    }
    out = tmp_path / "out"

    for name, (draw, outputs) in jobs.items():
        paths = figkit.render(out, name, draw, outputs)

        formats = [o.format for o in outputs]
        expected = [f"{name}{s}.{f}" for s in ("", "-dark") for f in formats]
        assert [p.name for p in paths] == expected
    # Pairs, transparency and file sizes, as `figkit check` tests them in CI.
    assert check(out) == []


def test_gradcam_wraps_into_rows_within_the_web_width(tmp_path: Path) -> None:
    draw = figures.gradcam_figure(_explanations(tmp_path, 7), columns=3)

    paths = figkit.render(tmp_path, "gradcam", draw, figures.PHOTO, themes=[figkit.DARK])

    with Image.open(paths[0]) as image:
        assert image.mode == "RGBA"
        assert image.getchannel("A").getextrema()[0] == 0
        assert image.width <= (figkit.WEB_WIDTH + 0.3) * figkit.RASTER_DPI
        # Three rows of image pairs are taller than wide.
        assert image.height > image.width


def test_robustness_legend_takes_the_free_cell_of_the_grid(tmp_path: Path) -> None:
    draw = figures.robustness_figure(_robustness(5), columns=3)

    with figkit.styled(figkit.LIGHT):
        figure = draw(figkit.LIGHT)
    axes = figure.get_axes()

    legend = axes[5].get_legend()
    assert len(axes) == 6
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == [
        "ResNet18",
        "HOG + linear SVM",
        "ResNet18 INT8",
    ]
