from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sdi import figures
from sdi.dataset import CLASSES, Sample
from sdi.metrics import evaluate


def test_every_figure_is_written_for_light_and_dark_pages(tmp_path: Path) -> None:
    image = tmp_path / "crazing_1.png"
    cv2.imwrite(str(image), np.full((200, 200), 128, dtype=np.uint8))
    samples = [Sample(image, label) for label in range(len(CLASSES))]
    labels = list(range(len(CLASSES)))
    report = evaluate("m", "test", CLASSES, labels, labels[::-1])
    history = [{"epoch": 1, "loss": 0.5, "val_macro_f1": 0.8}]
    jobs = {
        "samples": figures.samples_figure(samples, per_class=1),
        "confusion": figures.confusion_figure([report, report]),
        "per_class_f1": figures.per_class_figure(report, report),
        "training": figures.training_figure({0: history, 1: history}),
        "errors": figures.errors_figure([figures.Prediction(image, "a", "b", 0.6)] * 2),
        "gradcam": figures.gradcam_figure(
            [figures.Explanation(image, "a", "p = 1", np.eye(8, dtype=np.float32))] * 2
        ),
    }

    for name, draw in jobs.items():
        paths = figures.render(tmp_path / "out", name, draw)

        assert [p.name for p in paths] == [f"{name}.png", f"{name}-dark.png"]
        assert all(cv2.imread(str(p)) is not None for p in paths)


def _explanations(tmp_path: Path, count: int) -> list[figures.Explanation]:
    image = tmp_path / "crazing_1.png"
    cv2.imwrite(str(image), np.full((200, 200), 128, dtype=np.uint8))
    return [figures.Explanation(image, "a", "p = 1", np.eye(8, dtype=np.float32))] * count


def test_web_photo_is_transparent_webp_of_fixed_width(tmp_path: Path) -> None:
    draw = figures.gradcam_figure(_explanations(tmp_path, 7), columns=3)

    paths = figures.render(tmp_path / "web", "gradcam", draw, output=figures.WEB_PHOTO)

    assert [p.name for p in paths] == ["gradcam.webp", "gradcam-dark.webp"]
    for path in paths:
        with Image.open(path) as image:
            assert image.width == figures.WEB_PHOTO.pixels
            assert image.mode == "RGBA"
            assert image.getchannel("A").getextrema()[0] == 0


def test_web_chart_is_svg_without_background_or_date(tmp_path: Path) -> None:
    data = {
        "models": ["ResNet18", "HOG + linear SVM"],
        "robustness": [
            {
                "name": f"c{i}",
                "parameter": "p",
                "strengths": [0, 1],
                "accuracy": {"ResNet18": [1.0, 0.5], "HOG + linear SVM": [0.6, 0.3]},
            }
            for i in range(5)
        ],
    }
    draw = figures.robustness_figure(data, columns=3)

    first = figures.render(tmp_path / "a", "robustness", draw, output=figures.WEB)
    second = figures.render(tmp_path / "b", "robustness", draw, output=figures.WEB)

    text = first[0].read_text(encoding="utf-8")
    assert first[0].name == "robustness.svg"
    assert "<dc:date>" not in text
    assert f"fill: {figures.LIGHT.surface}" not in text
    assert [p.read_bytes() for p in first] == [p.read_bytes() for p in second]


def test_wide_figure_is_scaled_to_the_web_width(tmp_path: Path) -> None:
    wide = figures.gradcam_figure(_explanations(tmp_path, 9))
    output = figures.Output(format="png", dpi=50, width=figures.WEB_WIDTH)

    (path, *_) = figures.render(tmp_path, "wide", wide, themes=[figures.LIGHT], output=output)

    with Image.open(path) as image:
        assert image.width <= (figures.WEB_WIDTH + 0.3) * 50
