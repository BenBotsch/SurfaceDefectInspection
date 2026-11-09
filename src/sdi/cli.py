"""Command line of the project; ``sdi --help`` lists the commands."""

import argparse
import json
import platform
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sdi.baseline import HogSvm
from sdi.benchmark import Row, Timing, benchmark_to_markdown, time_call
from sdi.dataset import CLASSES, download, load_split
from sdi.export import (
    OnnxClassifier,
    calibration_samples,
    export_onnx,
    parity_fixture,
    quantize_onnx,
)
from sdi.metrics import (
    Report,
    SeedSummary,
    evaluate,
    seeds_to_markdown,
    summarize_seeds,
    to_markdown,
)
from sdi.preprocess import Image, read_gray, to_tensor
from sdi.robustness import evaluate_robustness, robustness_to_markdown
from sdi.train import TrainConfig, image_probabilities, load, probabilities, train

DATA = Path("data/neu-det")
RUNS = Path("runs")
RESULTS = Path("results")
FIGURES = Path("docs/figures")


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text with LF line endings, so results are identical on every platform."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _write_json(name: str, data: object) -> None:
    write_text(RESULTS / f"{name}.json", json.dumps(data, indent=2) + "\n")


def _write(report: Report, name: str, extra: dict[str, object] | None = None) -> None:
    _write_json(name, report.to_dict() | (extra or {}))
    write_text(RESULTS / f"{name}.md", to_markdown(report) + "\n")
    print(to_markdown(report))


def render(data: dict[str, Any]) -> str | None:
    """The Markdown of a results JSON file, or None for files without one."""
    if "confusion" in data:
        return to_markdown(Report.from_dict(data))
    if "summary" in data:
        s = data["summary"]
        summary = SeedSummary(tuple(s["seeds"]), tuple(s["accuracy"]), tuple(s["macro_f1"]))
        return seeds_to_markdown(data["model"], summary)
    if "benchmark" in data:
        return benchmark_to_markdown(data["benchmark"], data["note"])
    if "robustness" in data:
        return robustness_to_markdown(data)
    return None


def report(check: bool) -> int:
    """Render ``results/*.md`` from the JSON next to it, or only check that they agree.

    Returns:
        The number of Markdown files that are missing or differ.
    """
    stale = 0
    for path in sorted(RESULTS.glob("*.json")):
        markdown = render(json.loads(path.read_text(encoding="utf-8")))
        if markdown is None:
            continue
        target = path.with_suffix(".md")
        current = target.read_text(encoding="utf-8") if target.is_file() else None
        if current != markdown + "\n":
            stale += 1
            if check:
                print(f"{target} does not match {path.name}")
            else:
                write_text(target, markdown + "\n")
                print(f"wrote {target}")
    return stale


def _baseline(data: Path) -> None:
    svm = HogSvm()
    svm.fit(load_split(data, "train"))
    test = load_split(data, "test")
    _write(
        evaluate("HOG + linear SVM", "test", CLASSES, [s.label for s in test], svm.predict(test)),
        "baseline",
    )


def _evaluate(data: Path, run_dir: Path) -> None:
    test = load_split(data, "test")
    truth = [s.label for s in test]
    probs = probabilities(load(run_dir / "model.pt", _device()), test, _device())
    predicted = [int(i) for i in probs.argmax(axis=1)]
    _write(evaluate("ResNet18", "test", CLASSES, truth, predicted), "resnet18")
    _write_json(
        "resnet18_predictions",
        [
            {
                "image": s.path.name,
                "label": CLASSES[s.label],
                "predicted": CLASSES[p],
                "probabilities": [round(float(v), 6) for v in row],
            }
            for s, p, row in zip(test, predicted, probs, strict=True)
        ],
    )
    onnx_path = export_onnx(load(run_dir / "model.pt"), run_dir / "model.onnx")
    onnx_predicted, latency = OnnxClassifier(onnx_path).predict(test)
    extra = {
        "median_latency_ms": round(latency, 2),
        "latency_device": f"ONNX Runtime CPU, batch 1, {platform.processor() or 'unknown'}",
    }
    _write(
        evaluate("ResNet18 (ONNX)", "test", CLASSES, truth, onnx_predicted), "resnet18_onnx", extra
    )
    print(f"median latency {latency:.2f} ms per image on the CPU")


def _seeds(data: Path, run_dirs: list[Path]) -> None:
    test = load_split(data, "test")
    truth = [s.label for s in test]
    seeds, reports = [], []
    for run_dir in run_dirs:
        config = run_dir / "config.json"
        if not config.is_file():
            raise SystemExit(f"{config} is missing; it is written by `sdi train`")
        seed = int(json.loads(config.read_text(encoding="utf-8"))["seed"])
        probs = probabilities(load(run_dir / "model.pt", _device()), test, _device())
        result = evaluate("ResNet18", "test", CLASSES, truth, [int(i) for i in probs.argmax(1)])
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        _write_json(f"seeds/seed{seed}", result.to_dict() | {"seed": seed, "history": history})
        seeds.append(seed)
        reports.append(result)
    order = sorted(range(len(seeds)), key=seeds.__getitem__)
    summary = summarize_seeds([seeds[i] for i in order], [reports[i] for i in order])
    _write_json("seeds", {"model": "ResNet18", "summary": summary.to_dict()})
    markdown = seeds_to_markdown("ResNet18", summary)
    write_text(RESULTS / "seeds.md", markdown + "\n")
    print(markdown)


def _load_report(name: str) -> Report:
    return Report.from_dict(json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8")))


def _figures(data: Path) -> None:
    # Imported here so that the other commands do not load matplotlib.
    from sdi import figures  # noqa: PLC0415

    test = load_split(data, "test")
    baseline, resnet = _load_report("baseline"), _load_report("resnet18")
    histories = {
        int(d["seed"]): d["history"]
        for d in (
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((RESULTS / "seeds").glob("seed*.json"))
        )
    }
    predictions = json.loads((RESULTS / "resnet18_predictions.json").read_text(encoding="utf-8"))
    errors = [
        figures.Prediction(
            data / "images" / "test" / p["image"],
            p["label"],
            p["predicted"],
            max(p["probabilities"]),
        )
        for p in predictions
        if p["label"] != p["predicted"]
    ]
    jobs = {
        "samples": figures.samples_figure(test),
        "confusion": figures.confusion_figure([baseline, resnet]),
        "per_class_f1": figures.per_class_figure(baseline, resnet),
        "training": figures.training_figure(histories),
        "errors": figures.errors_figure(errors),
    }
    for name, draw in jobs.items():
        for path in figures.render(FIGURES, name, draw):
            print(f"wrote {path}")


def _gradcam(data: Path, run_dir: Path) -> None:
    from sdi import figures  # noqa: PLC0415
    from sdi.gradcam import grad_cam  # noqa: PLC0415

    model = load(run_dir / "model.pt")
    by_name = {s.path.name: s for s in load_split(data, "test")}
    predictions = json.loads((RESULTS / "resnet18_predictions.json").read_text(encoding="utf-8"))
    correct: dict[str, dict[str, Any]] = {}
    for p in predictions:
        if p["label"] == p["predicted"]:
            correct.setdefault(p["label"], p)
    chosen = [correct[c] for c in CLASSES if c in correct]
    chosen += [p for p in predictions if p["label"] != p["predicted"]]
    explanations = []
    for p in chosen:
        sample = by_name[p["image"]]
        tensor = torch.from_numpy(to_tensor(read_gray(sample.path)))[None]
        wrong = p["label"] != p["predicted"]
        explanations.append(
            figures.Explanation(
                path=sample.path,
                title=("misclassified" if wrong else p["label"].replace("_", " ")),
                caption=(
                    f"{p['label']}\n→ {p['predicted']}"
                    if wrong
                    else f"p = {max(p['probabilities']):.2f}"
                ),
                cam=grad_cam(model, tensor, CLASSES.index(p["predicted"])),
            )
        )
    for path in figures.render(FIGURES, "gradcam", figures.gradcam_figure(explanations)):
        print(f"wrote {path}")


def _megabytes(path: Path) -> float:
    return path.stat().st_size / 1e6


def _benchmark(data: Path, run_dir: Path, iterations: int) -> None:
    test = load_split(data, "test")
    truth = [s.label for s in test]
    int8 = _int8_model(data, run_dir)
    fp32 = run_dir / "model.onnx"
    int8_predicted, _ = OnnxClassifier(int8).predict(test)
    _write(
        evaluate("ResNet18 (ONNX INT8)", "test", CLASSES, truth, int8_predicted), "resnet18_int8"
    )

    # Every runtime gets the same decoded image and does its own preprocessing, like the C++ one.
    gray = read_gray(test[0].path)
    cpu = f"{platform.processor() or 'unknown CPU'}, {torch.get_num_threads()} threads"
    rows = []
    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    for device in devices:
        model = load(run_dir / "model.pt", device)
        predicted = probabilities(model, test, device).argmax(axis=1)

        def forward(model: torch.nn.Module = model, device: torch.device = device) -> object:
            with torch.inference_mode():
                # .cpu() waits for the GPU, so the time covers the whole inference.
                return model(torch.from_numpy(to_tensor(gray))[None].to(device)).cpu()

        timing = time_call(forward, iterations)
        name = torch.cuda.get_device_name(device) if device.type == "cuda" else cpu
        accuracy = float(np.mean(predicted == truth))
        size = _megabytes(run_dir / "model.pt")
        rows.append(Row(f"PyTorch {device.type.upper()}", name, accuracy, size, *_ms(timing)))
    for label, path in (("ONNX Runtime FP32", fp32), ("ONNX Runtime INT8", int8)):
        session = OnnxClassifier(path)
        predicted_onnx, _ = session.predict(test)

        def run(session: OnnxClassifier = session) -> object:
            return session.probabilities(to_tensor(gray)[None])

        timing = time_call(run, iterations)
        accuracy = float(np.mean(np.array(predicted_onnx) == truth))
        rows.append(Row(label, cpu, accuracy, _megabytes(path), *_ms(timing)))
    cpp = RESULTS / "benchmark_cpp.json"
    if cpp.is_file():
        c = json.loads(cpp.read_text(encoding="utf-8"))
        where = f"same CPU, Docker on WSL2, {c['threads']} threads"
        rows.append(Row(c["runtime"], where, None, _megabytes(fp32), c["median_ms"], c["p95_ms"]))
    note = (
        f"Batch 1, one test image decoded beforehand; preprocessing and inference timed together, "
        f"{iterations} calls after 20 warm-up calls. The C++ classifier reads the FP32 model and "
        "matches ONNX Runtime in the parity test, so its accuracy was not measured separately."
    )
    _write_json("benchmark", {"benchmark": [r.to_dict() for r in rows], "note": note})
    markdown = benchmark_to_markdown([r.to_dict() for r in rows], note)
    write_text(RESULTS / "benchmark.md", markdown + "\n")
    print(markdown)


def _ms(timing: Timing) -> tuple[float, float]:
    return timing.median_ms, timing.p95_ms


def _int8_model(data: Path, run_dir: Path) -> Path:
    fp32, int8 = run_dir / "model.onnx", run_dir / "model_int8.onnx"
    if not fp32.is_file():
        export_onnx(load(run_dir / "model.pt"), fp32)
    return quantize_onnx(fp32, int8, calibration_samples(load_split(data, "train")))


def _robustness(data: Path, run_dir: Path) -> None:
    from sdi import figures  # noqa: PLC0415

    test = load_split(data, "test")
    svm = HogSvm()
    svm.fit(load_split(data, "train"))
    model = load(run_dir / "model.pt", _device())
    int8 = OnnxClassifier(_int8_model(data, run_dir))

    def resnet(images: Sequence[Image]) -> list[int]:
        return [int(i) for i in image_probabilities(model, images, _device()).argmax(axis=1)]

    def quantised(images: Sequence[Image]) -> list[int]:
        tensors = np.stack([to_tensor(i) for i in images])
        return [int(i) for i in int8.probabilities(tensors).argmax(axis=1)]

    result = evaluate_robustness(
        [read_gray(s.path) for s in test],
        [s.label for s in test],
        {"HOG + linear SVM": svm.predict_images, "ResNet18": resnet, "ResNet18 INT8": quantised},
    )
    _write_json("robustness", result)
    markdown = robustness_to_markdown(result)
    write_text(RESULTS / "robustness.md", markdown + "\n")
    print(markdown)
    for path in figures.render(FIGURES, "robustness", figures.robustness_figure(result)):
        print(f"wrote {path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdi", description="Steel surface defect classification on NEU-DET."
    )
    parser.add_argument("--data", type=Path, default=DATA, help="dataset directory")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="download NEU-DET at the pinned revision and verify it")
    sub.add_parser("baseline", help="train HOG + linear SVM and evaluate it on the test split")
    train_cmd = sub.add_parser("train", help="fine-tune ResNet18, best epoch by validation F1")
    train_cmd.add_argument("--epochs", type=int, default=TrainConfig().epochs)
    train_cmd.add_argument("--seed", type=int, default=TrainConfig().seed)
    train_cmd.add_argument("--run-dir", type=Path, default=RUNS, help="output directory")
    eval_cmd = sub.add_parser("evaluate", help="evaluate a model with PyTorch and ONNX Runtime")
    eval_cmd.add_argument("--run-dir", type=Path, default=RUNS)
    seeds_cmd = sub.add_parser("seeds", help="evaluate runs of different seeds, mean and std")
    seeds_cmd.add_argument("run_dirs", type=Path, nargs="+", metavar="RUN_DIR")
    export_cmd = sub.add_parser("export", help="export model.pt to model.onnx")
    export_cmd.add_argument("--run-dir", type=Path, default=RUNS)
    report_cmd = sub.add_parser("report", help="render results/*.md from results/*.json")
    report_cmd.add_argument("--check", action="store_true", help="fail if a file is stale")
    sub.add_parser("figures", help="render the README figures to docs/figures/")
    gradcam_cmd = sub.add_parser("gradcam", help="render Grad-CAM maps to docs/figures/")
    gradcam_cmd.add_argument("--run-dir", type=Path, default=RUNS)
    bench_cmd = sub.add_parser(
        "benchmark", help="quantise to INT8 and time each runtime on one image"
    )
    bench_cmd.add_argument("--run-dir", type=Path, default=RUNS)
    bench_cmd.add_argument("--iterations", type=int, default=200)
    robust_cmd = sub.add_parser(
        "robustness", help="accuracy of all models on corrupted test images"
    )
    robust_cmd.add_argument("--run-dir", type=Path, default=RUNS)
    fixture = sub.add_parser("parity-fixture", help="write the fixture of the C++ parity test")
    fixture.add_argument("directory", type=Path)
    return parser


def _download(args: argparse.Namespace) -> None:
    fetched = download(args.data, progress=lambda n: print(f"\r{n} files checked", end=""))
    print(f"\n{fetched} files downloaded to {args.data}")


def _train(args: argparse.Namespace) -> None:
    config = TrainConfig(epochs=args.epochs, seed=args.seed)
    train(load_split(args.data, "train"), load_split(args.data, "val"), args.run_dir, config)


def _report(args: argparse.Namespace) -> None:
    stale = report(args.check)
    if args.check and stale:
        raise SystemExit(f"{stale} results file(s) are stale; run `sdi report`")


COMMANDS: dict[str, Callable[[argparse.Namespace], object]] = {
    "download": _download,
    "baseline": lambda a: _baseline(a.data),
    "train": _train,
    "evaluate": lambda a: _evaluate(a.data, a.run_dir),
    "seeds": lambda a: _seeds(a.data, a.run_dirs),
    "export": lambda a: print(export_onnx(load(a.run_dir / "model.pt"), a.run_dir / "model.onnx")),
    "report": _report,
    "figures": lambda a: _figures(a.data),
    "gradcam": lambda a: _gradcam(a.data, a.run_dir),
    "benchmark": lambda a: _benchmark(a.data, a.run_dir, a.iterations),
    "robustness": lambda a: _robustness(a.data, a.run_dir),
    "parity-fixture": lambda a: print(parity_fixture(a.directory)),
}


def main(argv: list[str] | None = None) -> None:
    """Run one command."""
    args = _parser().parse_args(argv)
    COMMANDS[args.command](args)
