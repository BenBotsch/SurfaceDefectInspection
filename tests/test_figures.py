import json
from pathlib import Path

import cv2
import figkit
import numpy as np
import pytest
from figkit.check import check
from figkit.language import speaking
from PIL import Image

from sdi import cli, figures
from sdi.dataset import CLASSES, Sample
from sdi.metrics import evaluate


def _image(tmp_path: Path) -> Path:
    image = tmp_path / "crazing_1.png"
    cv2.imwrite(str(image), np.full((200, 200), 128, dtype=np.uint8))
    return image


def _explanations(tmp_path: Path, count: int) -> list[figures.Explanation]:
    cam = np.eye(8, dtype=np.float32)
    return [figures.Explanation(_image(tmp_path), "crazing", "crazing", 1.0, cam)] * count


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
    # From the highest to the lowest mean accuracy, not in the order of ``models``.
    assert [t.get_text() for t in legend.get_texts()] == [
        "ResNet18",
        "ResNet18 INT8",
        "HOG + linear SVM",
    ]


def test_robustness_names_its_panels_and_models_in_german() -> None:
    accuracy = {"ResNet18": [1.0, 0.5], "HOG + linear SVM": [0.6, 0.3]}
    panel = {"name": "Gaussian noise", "parameter": "offset", "strengths": [0, 0.5]}
    data = {"models": list(accuracy), "robustness": [{**panel, "accuracy": accuracy}] * 4}
    draw = figures.robustness_figure(data, columns=3)

    with speaking("de"), figkit.styled(figkit.LIGHT):
        figure = draw(figkit.LIGHT)
    ax, spare = figure.get_axes()[0], figure.get_axes()[4]
    legend = spare.get_legend()

    assert ax.get_title(loc="left") == "Gaußsches Rauschen"
    assert ax.get_xlabel() == "Versatz"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "0,5"]
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ["ResNet18", "HOG + lineare SVM"]


def test_gradcam_writes_the_column_texts_in_the_language_drawn(tmp_path: Path) -> None:
    cam = np.eye(8, dtype=np.float32)
    right = figures.Explanation(_image(tmp_path), "pitted_surface", "pitted_surface", 0.987, cam)
    wrong = figures.Explanation(_image(tmp_path), "crazing", "rolled-in_scale", 0.6, cam)

    with speaking("de"):
        assert (right.title(), right.caption()) == ("Pitted Surface", "p = 0,99")
        assert wrong.title() == "falsch erkannt"
        assert wrong.caption() == "Crazing\n→ Rolled-in Scale"
    assert (right.title(), wrong.title()) == ("pitted surface", "misclassified")


def test_every_text_exists_in_each_language() -> None:
    keys = {code: set(texts) for code, texts in figures.TEXT.items()}

    assert set(keys) == set(figures.LANGUAGES)
    assert keys["en"] == keys["de"]
    assert set(figures.CLASS_LABELS["de"]) == set(CLASSES)


def test_captions_write_numbers_as_in_their_language() -> None:
    labels = list(range(len(CLASSES)))
    model = evaluate("ResNet18", "test", CLASSES, labels * 3, labels * 3)
    baseline = evaluate("HOG + linear SVM", "test", CLASSES, labels * 3, labels[::-1] * 3)

    captions = figures.confusion_captions(baseline, model)

    assert captions["en"].title == "ResNet18 misclassifies 0 of 18 test images, the baseline 18"
    assert "HOG + lineare SVM mit Accuracy 0,000" in captions["de"].alt
    assert "Accuracy 1,000" in captions["de"].alt


def test_page_figures_come_per_language_with_their_texts_in_a_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = Path("results")
    monkeypatch.setattr(cli, "RESULTS", results.resolve())
    monkeypatch.setattr(cli, "PAGE", tmp_path)
    histories = cli._histories()
    report = cli._load_report("resnet18")
    for name, draw in (
        ("training", figures.training_figure(histories)),
        ("confusion", figures.confusion_figure([cli._load_report("baseline"), report])),
    ):
        cli._write_page(name, draw)

    cli._write_manifest()

    manifest = json.loads((tmp_path / "figures.json").read_text(encoding="utf-8"))
    entries = {e["name"]: e for e in manifest["figures"]}
    assert set(manifest["languages"]) == set(figures.LANGUAGES)
    # Only the figures on disk: the others are written by their own commands.
    assert set(entries) == {"training", "confusion"}
    assert entries["confusion"]["files"]["de"]["dark"] == "confusion.de-dark.svg"
    assert "Testbildern" in entries["confusion"]["title"]["de"]
    assert all(entries["training"]["alt"][code] for code in figures.LANGUAGES)
    assert check(tmp_path) == []
