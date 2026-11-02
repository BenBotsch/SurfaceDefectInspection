"""ONNX export, ONNX Runtime inference and the fixture for the C++ parity test."""

import json
import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort
import torch
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
