import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import pytest
import torch

from sdi.baseline import HogSvm
from sdi.dataset import CLASSES, Sample
from sdi.export import OnnxClassifier, export_onnx, parity_fixture
from sdi.metrics import evaluate, to_markdown
from sdi.model import build_model
from sdi.preprocess import MEAN, SIZE, STD, augment, to_tensor
from sdi.train import TrainConfig, load, predict, train


def test_tensor_is_resized_and_normalised() -> None:
    tensor = to_tensor(np.full((200, 200), 255, dtype=np.uint8))

    assert tensor.shape == (1, SIZE, SIZE)
    assert tensor.dtype == np.float32
    assert tensor[0, 0, 0] == pytest.approx((1 - MEAN) / STD)


def test_augment_keeps_shape_and_type() -> None:
    image = np.arange(200 * 200, dtype=np.uint32).reshape(200, 200).astype(np.uint8)

    out = augment(image, np.random.default_rng(0))

    assert out.shape == (200, 200)
    assert out.dtype == np.uint8


def test_model_takes_one_channel_and_returns_one_score_per_class() -> None:
    model = build_model(len(CLASSES), pretrained=False).eval()

    assert model(torch.zeros(2, 1, SIZE, SIZE)).shape == (2, len(CLASSES))


def test_onnx_matches_pytorch(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_model(len(CLASSES), pretrained=False).eval()
    batch = torch.randn(3, 1, SIZE, SIZE)
    with torch.no_grad():
        expected = torch.softmax(model(batch), dim=1).numpy()

    onnx = OnnxClassifier(export_onnx(model, tmp_path / "model.onnx"))

    np.testing.assert_allclose(onnx.probabilities(batch.numpy()), expected, atol=1e-5)


def test_export_has_no_identity_nodes_for_opencv(tmp_path: Path) -> None:
    path = export_onnx(build_model(len(CLASSES), pretrained=False), tmp_path / "model.onnx")

    assert "Identity" not in {node.op_type for node in onnx.load(str(path)).graph.node}


def test_parity_fixture_is_consistent(tmp_path: Path) -> None:
    directory = parity_fixture(tmp_path / "fixture")

    expected = json.loads((directory / "expected.json").read_text())
    assert expected["classes"] == list(CLASSES)
    assert sum(expected["probabilities"]) == pytest.approx(1, abs=1e-5)
    image = cv2.imread(str(directory / "image.png"), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    assert image.shape == (200, 200)


def synthetic(directory: Path, per_class: int = 4) -> list[Sample]:
    """Images whose class is a distinct stripe frequency, so a model can learn them."""
    directory.mkdir(parents=True, exist_ok=True)
    samples = []
    rng = np.random.default_rng(0)
    for label in range(len(CLASSES)):
        for i in range(per_class):
            x = np.linspace(0, (label + 1) * np.pi * 4, 200)
            image = (127 + 100 * np.sin(x))[np.newaxis].repeat(200, 0) + rng.normal(
                0, 5, (200, 200)
            )
            path = directory / f"{label}_{i}.png"
            cv2.imwrite(str(path), np.clip(image, 0, 255).astype(np.uint8))
            samples.append(Sample(path, label))
    return samples


def test_training_writes_the_best_model_and_history(tmp_path: Path) -> None:
    samples = synthetic(tmp_path / "images", per_class=2)
    config = TrainConfig(epochs=1, batch_size=4, pretrained=False)

    path = train(samples, samples, tmp_path / "run", config, log=lambda _: None)

    history = json.loads((tmp_path / "run" / "history.json").read_text())
    assert [entry["epoch"] for entry in history] == [1]
    assert json.loads((tmp_path / "run" / "config.json").read_text())["epochs"] == 1
    assert len(predict(load(path), samples, torch.device("cpu"))) == len(samples)


def test_baseline_learns_separable_images(tmp_path: Path) -> None:
    samples = synthetic(tmp_path / "images")
    model = HogSvm(c=1.0)

    model.fit(samples)

    assert model.predict(samples) == [s.label for s in samples]


def test_report_counts_and_renders() -> None:
    report = evaluate("m", "test", ("a", "b"), [0, 0, 1, 1], [0, 1, 1, 1])

    assert report.accuracy == 0.75
    assert report.confusion == ((1, 1), (0, 2))
    assert report.classes[0].recall == 0.5
    assert "| a | 1 | 1 |" in to_markdown(report)
