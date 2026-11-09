import json
from pathlib import Path

import pytest

from sdi import cli
from sdi.metrics import Report, evaluate, seeds_to_markdown, summarize_seeds, to_markdown


def test_report_survives_a_json_round_trip() -> None:
    report = evaluate("m", "test", ("crazing", "scratches"), [0, 1, 1], [0, 1, 0])

    assert Report.from_dict(json.loads(json.dumps(report.to_dict()))) == report


def test_confusion_matrix_uses_neu_abbreviations() -> None:
    report = evaluate("m", "test", ("crazing", "rolled-in_scale"), [0, 1], [0, 1])

    assert "| | Cr | RS |" in to_markdown(report)


def test_seed_summary_has_mean_and_sample_std() -> None:
    reports = [
        evaluate("m", "test", ("a", "b"), [0, 1], predicted) for predicted in ([0, 1], [0, 0])
    ]

    summary = summarize_seeds([3, 4], reports)

    assert summary.accuracy == (1.0, 0.5)
    assert summary.accuracy_mean == pytest.approx(0.75)
    assert summary.accuracy_std == pytest.approx(0.3536, abs=1e-4)
    assert "| mean ± std | 0.750 ± 0.354 |" in seeds_to_markdown("m", summary)


def test_seed_summary_needs_one_report_per_seed() -> None:
    with pytest.raises(ValueError, match="one report per seed"):
        summarize_seeds([0, 1], [evaluate("m", "test", ("a",), [0], [0])])


def test_report_check_finds_stale_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    report = evaluate("m", "test", ("a", "b"), [0, 1], [0, 1])
    (tmp_path / "m.json").write_text(json.dumps(report.to_dict()))
    (tmp_path / "other.json").write_text("[]")

    assert cli.report(check=True) == 1
    assert cli.report(check=False) == 1
    assert cli.report(check=True) == 0
    assert (tmp_path / "m.md").read_text() == to_markdown(report) + "\n"
    assert not (tmp_path / "other.md").exists()
