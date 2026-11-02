"""Command line: ``sdi download | baseline | train | evaluate | export | parity-fixture``."""

import argparse
import json
import platform
from pathlib import Path

import torch

from sdi.baseline import HogSvm
from sdi.dataset import CLASSES, download, load_split
from sdi.export import OnnxClassifier, export_onnx, parity_fixture
from sdi.metrics import Report, evaluate, to_markdown
from sdi.train import TrainConfig, load, predict, train

DATA = Path("data/neu-det")
RUNS = Path("runs")
RESULTS = Path("results")


def _write(report: Report, name: str, extra: dict[str, object] | None = None) -> None:
    RESULTS.mkdir(exist_ok=True)
    data = report.to_dict() | (extra or {})
    (RESULTS / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (RESULTS / f"{name}.md").write_text(to_markdown(report) + "\n", encoding="utf-8")
    print(to_markdown(report))


def main(argv: list[str] | None = None) -> None:
    """Run one command."""
    parser = argparse.ArgumentParser(prog="sdi", description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA, help="dataset directory")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="download NEU-DET at the pinned revision and verify it")
    sub.add_parser("baseline", help="train HOG + linear SVM and evaluate it on the test split")
    train_cmd = sub.add_parser("train", help="fine-tune ResNet18, best epoch by validation F1")
    train_cmd.add_argument("--epochs", type=int, default=TrainConfig().epochs)
    train_cmd.add_argument("--seed", type=int, default=TrainConfig().seed)
    sub.add_parser("evaluate", help="evaluate the trained model with PyTorch and ONNX Runtime")
    sub.add_parser("export", help="export runs/model.pt to runs/model.onnx")
    fixture = sub.add_parser("parity-fixture", help="write the fixture of the C++ parity test")
    fixture.add_argument("directory", type=Path)
    args = parser.parse_args(argv)

    if args.command == "download":
        fetched = download(args.data, progress=lambda n: print(f"\r{n} files checked", end=""))
        print(f"\n{fetched} files downloaded to {args.data}")
    elif args.command == "baseline":
        svm = HogSvm()
        svm.fit(load_split(args.data, "train"))
        test = load_split(args.data, "test")
        report = evaluate(
            "HOG + linear SVM",
            "test",
            CLASSES,
            [s.label for s in test],
            svm.predict(test),
        )
        _write(report, "baseline")
    elif args.command == "train":
        config = TrainConfig(epochs=args.epochs, seed=args.seed)
        train(load_split(args.data, "train"), load_split(args.data, "val"), RUNS, config)
    elif args.command == "evaluate":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test = load_split(args.data, "test")
        truth = [s.label for s in test]
        model = load(RUNS / "model.pt", device)
        _write(
            evaluate("ResNet18", "test", CLASSES, truth, predict(model, test, device)),
            "resnet18",
        )
        onnx_path = export_onnx(load(RUNS / "model.pt"), RUNS / "model.onnx")
        predicted, latency = OnnxClassifier(onnx_path).predict(test)
        extra = {
            "median_latency_ms": round(latency, 2),
            "latency_device": f"ONNX Runtime CPU, batch 1, {platform.processor() or 'unknown'}",
        }
        _write(
            evaluate("ResNet18 (ONNX)", "test", CLASSES, truth, predicted),
            "resnet18_onnx",
            extra,
        )
        print(f"median latency {latency:.2f} ms per image on the CPU")
    elif args.command == "export":
        print(export_onnx(load(RUNS / "model.pt"), RUNS / "model.onnx"))
    elif args.command == "parity-fixture":
        print(parity_fixture(args.directory))
