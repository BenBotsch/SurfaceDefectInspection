"""ONNX export, ONNX Runtime inference and the fixture for the C++ parity test."""

import json
import tempfile
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch import nn

from sdi.dataset import CLASSES, Sample
from sdi.model import build_model
from sdi.preprocess import SIZE, read_gray, to_tensor

# OpenCV DNN in Ubuntu 24.04 (OpenCV 4.6) reads this opset; newer ones fail on some operators.
OPSET = 13


def export_onnx(model: nn.Module, path: Path) -> Path:
    """Write the model as ONNX with a dynamic batch size and softmax probabilities as output."""
    wrapped = nn.Sequential(model.cpu().eval(), nn.Softmax(dim=1))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        (torch.zeros(1, 1, SIZE, SIZE),),
        str(path),
        input_names=["image"],
        output_names=["probabilities"],
        dynamic_axes={"image": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )
    graph = onnx.load(str(path))
    _drop_identities(graph.graph)
    onnx.save(graph, str(path))
    return path


def _drop_identities(graph: onnx.GraphProto) -> None:
    """Remove ``Identity`` nodes, which OpenCV 4.6 DNN cannot import when they read weights."""
    outputs = {output.name for output in graph.output}
    for node in [n for n in graph.node if n.op_type == "Identity"]:
        source, target = node.input[0], node.output[0]
        if target in outputs:
            continue
        graph.node.remove(node)
        for other in graph.node:
            for i, name in enumerate(other.input):
                if name == target:
                    other.input[i] = source


class _Calibration(CalibrationDataReader):  # type: ignore[misc]
    """Feeds preprocessed images to the quantiser, one at a time."""

    def __init__(self, samples: Sequence[Sample]) -> None:
        self._inputs: Iterator[dict[str, npt.NDArray[np.float32]]] = (
            {"image": to_tensor(read_gray(s.path))[np.newaxis]} for s in samples
        )

    def get_next(self) -> dict[str, npt.NDArray[np.float32]] | None:
        return next(self._inputs, None)


def calibration_samples(samples: Sequence[Sample], per_class: int = 20) -> list[Sample]:
    """The first ``per_class`` samples of each class, so calibration is the same on every run."""
    return [
        s for c in range(len(CLASSES)) for s in [x for x in samples if x.label == c][:per_class]
    ]


def quantize_onnx(source: Path, target: Path, calibration: Sequence[Sample]) -> Path:
    """Write an INT8 copy of an exported model, calibrated statically on the given samples.

    Weights are quantised per channel to signed 8 bit, activations to unsigned 8 bit, in the QDQ
    format that ONNX Runtime fuses on the CPU. The softmax stays in float, so the output is still
    a probability. Activation ranges come from the 99.999th percentile, which kept one more
    validation image correct than min-max or entropy calibration (178 against 177 of 180).
    Calibrate on training images only; the test split must stay unseen.
    """
    if not calibration:
        raise ValueError("calibration needs at least one sample")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        prepared = Path(tmp) / "prepared.onnx"
        quant_pre_process(str(source), str(prepared))
        softmax = [n.name for n in onnx.load(str(prepared)).graph.node if n.op_type == "Softmax"]
        quantize_static(
            str(prepared),
            str(target),
            _Calibration(calibration),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            nodes_to_exclude=softmax,
            calibrate_method=CalibrationMethod.Percentile,
        )
    return target


class OnnxClassifier:
    """Classify images with ONNX Runtime on the CPU."""

    def __init__(self, path: Path) -> None:
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def probabilities(self, tensors: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Probabilities per image for a batch of preprocessed tensors."""
        out = self._session.run(["probabilities"], {"image": tensors})[0]
        return np.asarray(out, dtype=np.float32)

    def predict(self, samples: Sequence[Sample]) -> tuple[list[int], float]:
        """Predicted class per sample and the median latency of one image in milliseconds."""
        predicted, latencies = [], []
        for sample in samples:
            tensor = to_tensor(read_gray(sample.path))[np.newaxis]
            start = time.perf_counter()
            probabilities = self.probabilities(tensor)
            latencies.append((time.perf_counter() - start) * 1000)
            predicted.append(int(probabilities[0].argmax()))
        return predicted, float(np.median(latencies))


def parity_fixture(directory: Path, seed: int = 0) -> Path:
    """An untrained model, a synthetic image and its expected probabilities for the C++ test.

    The C++ test reads ``model.onnx`` and ``image.png``, runs its own preprocessing and compares
    with ``expected.json``. The weights are random, so the fixture needs no dataset.
    """
    torch.manual_seed(seed)
    directory.mkdir(parents=True, exist_ok=True)
    model = build_model(len(CLASSES), pretrained=False)
    export_onnx(model, directory / "model.onnx")
    rng = np.random.default_rng(seed)
    gradient = np.tile(np.linspace(0, 255, 200, dtype=np.float32), (200, 1))
    image = np.clip(gradient + rng.normal(0, 20, (200, 200)), 0, 255).astype(np.uint8)
    cv2.imwrite(str(directory / "image.png"), image)
    probabilities = OnnxClassifier(directory / "model.onnx").probabilities(
        to_tensor(read_gray(directory / "image.png"))[np.newaxis]
    )[0]
    expected = {
        "classes": list(CLASSES),
        "probabilities": [float(p) for p in probabilities],
    }
    (directory / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return directory
