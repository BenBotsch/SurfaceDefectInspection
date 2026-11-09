from pathlib import Path

import cv2
import numpy as np

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
