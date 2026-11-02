"""Image preprocessing shared by training, ONNX export and the C++ classifier.

``cpp/src/classifier.cpp`` repeats these steps; ``tests/test_model_and_export.py`` and the C++
parity test keep both sides equal.
"""

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

SIZE = 224
# Grey-level mean and standard deviation of ImageNet, which the pretrained weights expect.
MEAN = 0.449
STD = 0.226

Image = npt.NDArray[np.uint8]
Tensor = npt.NDArray[np.float32]


def read_gray(path: Path) -> Image:
    """Read an image file as 8-bit greyscale.

    Raises:
        ValueError: If OpenCV cannot read the file.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read image {path}")
    return np.asarray(image, dtype=np.uint8)


def to_tensor(gray: Image) -> Tensor:
    """Resize to ``SIZE`` x ``SIZE`` bilinearly and normalise to a 1 x H x W float tensor."""
    resized = cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    scaled = resized.astype(np.float32) / np.float32(255)
    return ((scaled - np.float32(MEAN)) / np.float32(STD))[np.newaxis, :, :]


def augment(gray: Image, rng: np.random.Generator) -> Image:
    """Random flips and a mild brightness and contrast change for training.

    Flips keep the direction of scratches and rolled-in scale along the strip axes; rotations by
    90 degrees would not, so there are none.
    """
    if rng.random() < 0.5:
        gray = np.asarray(cv2.flip(gray, 1), dtype=np.uint8)
    if rng.random() < 0.5:
        gray = np.asarray(cv2.flip(gray, 0), dtype=np.uint8)
    alpha = rng.uniform(0.85, 1.15)
    beta = rng.uniform(-15, 15)
    return np.asarray(cv2.convertScaleAbs(gray, alpha=alpha, beta=beta), dtype=np.uint8)
