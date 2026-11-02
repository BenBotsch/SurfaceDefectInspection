"""Classical baseline: HOG features from OpenCV and a linear SVM."""

from collections.abc import Sequence

import cv2
import numpy as np
import numpy.typing as npt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from sdi.dataset import Sample
from sdi.preprocess import read_gray

# 128 x 128 window, 16-pixel blocks with 8-pixel stride, 9 orientation bins: 8100 features.
_HOG = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)


def hog_features(samples: Sequence[Sample]) -> npt.NDArray[np.float32]:
    """One HOG vector per image, computed on the image scaled to 128 x 128."""
    rows = [
        _HOG.compute(cv2.resize(read_gray(s.path), (128, 128), interpolation=cv2.INTER_AREA))
        for s in samples
    ]
    return np.stack([np.ravel(r) for r in rows]).astype(np.float32)


class HogSvm:
    """Standardised HOG features with a linear SVM; C is fixed, not tuned on the test split."""

    def __init__(self, c: float = 0.01, seed: int = 0) -> None:
        self._pipeline = make_pipeline(
            StandardScaler(), LinearSVC(C=c, random_state=seed, max_iter=20_000)
        )

    def fit(self, samples: Sequence[Sample]) -> None:
        """Train on the samples."""
        self._pipeline.fit(hog_features(samples), [s.label for s in samples])

    def predict(self, samples: Sequence[Sample]) -> list[int]:
        """Predicted class index per sample."""
        return [int(p) for p in self._pipeline.predict(hog_features(samples))]
