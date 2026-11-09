from pathlib import Path

import numpy as np
import onnx
import pytest

from sdi.benchmark import Row, benchmark_to_markdown, time_call
from sdi.cli import render
from sdi.dataset import CLASSES, Sample
from sdi.export import OnnxClassifier, calibration_samples, parity_fixture, quantize_onnx
from sdi.preprocess import read_gray, to_tensor


def test_int8_model_is_smaller_and_returns_probabilities(tmp_path: Path) -> None:
    fixture = parity_fixture(tmp_path / "fixture")
    image = fixture / "image.png"

    int8 = quantize_onnx(fixture / "model.onnx", tmp_path / "int8.onnx", [Sample(image, 0)] * 4)
    out = OnnxClassifier(int8).probabilities(to_tensor(read_gray(image))[np.newaxis])

    assert out.shape == (1, len(CLASSES))
    assert out.sum() == pytest.approx(1, abs=1e-5)
    assert int8.stat().st_size < (fixture / "model.onnx").stat().st_size / 3
    assert any(n.op_type == "QuantizeLinear" for n in onnx.load(str(int8)).graph.node)


def test_quantisation_needs_calibration_samples(tmp_path: Path) -> None:
    fixture = parity_fixture(tmp_path / "fixture")

    with pytest.raises(ValueError, match="calibration"):
        quantize_onnx(fixture / "model.onnx", tmp_path / "int8.onnx", [])


def test_calibration_takes_the_first_samples_of_each_class() -> None:
    samples = [Sample(Path(f"{c}_{i}.png"), c) for i in range(5) for c in range(len(CLASSES))]

    chosen = calibration_samples(samples, per_class=2)

    assert [s.path.name for s in chosen[:2]] == ["0_0.png", "0_1.png"]
    assert len(chosen) == 2 * len(CLASSES)


def test_time_call_warms_up_and_reports_milliseconds() -> None:
    calls = []
    ticks = iter(float(t) for t in range(100))

    timing = time_call(lambda: calls.append(1), iterations=10, warmup=3, clock=lambda: next(ticks))

    assert len(calls) == 13
    assert timing.median_ms == pytest.approx(1000)
    assert timing.p95_ms == pytest.approx(1000)
    with pytest.raises(ValueError, match="iteration"):
        time_call(lambda: None, iterations=0)


def test_benchmark_json_renders_to_a_table() -> None:
    rows = [
        Row("ONNX Runtime INT8", "cpu", 0.98333, 11.26, 5.514, 6.0).to_dict(),
        Row("OpenCV DNN (C++)", "cpu", None, 44.7, 12.0, 13.0).to_dict(),
    ]

    markdown = render({"benchmark": rows, "note": "Batch 1."})

    assert markdown == benchmark_to_markdown(rows, "Batch 1.")
    assert "| ONNX Runtime INT8 | cpu | 0.983 | 11.3 MB | 5.51 ms | 6.00 ms |" in markdown
    assert "| n/a |" in markdown
    assert markdown.endswith("Batch 1.")
