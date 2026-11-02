"""Fine-tune ResNet18 on the training split and keep the epoch with the best validation F1."""

import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sdi.dataset import CLASSES, Sample
from sdi.model import build_model
from sdi.preprocess import augment, read_gray, to_tensor


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Hyperparameters; the defaults produced the results in the README."""

    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 0
    pretrained: bool = True


class ImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Samples as tensors, augmented when a random generator is given."""

    def __init__(self, samples: Sequence[Sample], rng: np.random.Generator | None = None) -> None:
        self._samples = samples
        self._images = [read_gray(s.path) for s in samples]
        self._rng = rng

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self._images[index]
        if self._rng is not None:
            image = augment(image, self._rng)
        return torch.from_numpy(to_tensor(image)), self._samples[index].label


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict(model: nn.Module, samples: Sequence[Sample], device: torch.device) -> list[int]:
    """Predicted class index per sample."""
    model.eval()
    loader = DataLoader(ImageDataset(samples), batch_size=64)
    out: list[int] = []
    with torch.no_grad():
        for images, _ in loader:
            out += model(images.to(device)).argmax(dim=1).tolist()
    return out


def train(
    train_samples: Sequence[Sample],
    val_samples: Sequence[Sample],
    output: Path,
    config: TrainConfig | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Train, write ``model.pt`` and ``history.json`` to ``output`` and return the model path.

    The saved weights are those of the epoch with the best macro F1 on the validation split.
    """
    config = config or TrainConfig()
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(CLASSES), pretrained=config.pretrained).to(device)
    loader = DataLoader(
        ImageDataset(train_samples, np.random.default_rng(config.seed)),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    loss_fn = nn.CrossEntropyLoss()
    truth = [s.label for s in val_samples]
    output.mkdir(parents=True, exist_ok=True)
    path = output / "model.pt"
    best, history = -1.0, []
    for epoch in range(1, config.epochs + 1):
        model.train()
        start, total = time.perf_counter(), 0.0
        for images, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
        schedule.step()
        val_f1 = float(f1_score(truth, predict(model, val_samples, device), average="macro"))
        history.append({"epoch": epoch, "loss": total / len(train_samples), "val_macro_f1": val_f1})
        log(
            f"epoch {epoch:2d}: loss {total / len(train_samples):.4f}, val macro F1 {val_f1:.4f}"
            f" ({time.perf_counter() - start:.0f}s)"
        )
        if val_f1 > best:
            best = val_f1
            torch.save(model.state_dict(), path)
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return path


def load(path: Path, device: torch.device | None = None) -> nn.Module:
    """A trained model from ``model.pt``."""
    model = build_model(len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device or "cpu", weights_only=True))
    return model.to(device or torch.device("cpu")).eval()
