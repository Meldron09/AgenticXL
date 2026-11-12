"""Full-corpus agreement mode (issue #20): synthetic rationale logs + fake
scorers, no network and no model. Agreement is with the Baseline judge's
recorded verdicts, never accuracy."""

from __future__ import annotations

import json

import pytest
from helpers import write_synthetic_workbook

from agenticxl.config import RunConfig
from scripts import run_system1_bakeoff
from scripts.run_system1_bakeoff import (
    build_agreement_report,
    evaluate_full_corpus,
    render_agreement_markdown,
)

# The reader finds the header by content (>= 20 non-blank cells), so pad to 20.
_PAD = [f"C{i}" for i in range(18)]


class FakeScorer:
    name = "fake"
    model_id = "fake/fake"
    revision = "rev0"
    template = "t {keywords}"
    gate_eligible = True

    def __init__(self, score_fn):
        self._score_fn = score_fn
        self.seen: list[str] = []

    def score(self, texts):
        self.seen.extend(texts)
        return [self._score_fn(t) for t in texts]


def _lift_score(text):
    return 0.9 if "lift" in text else 0.1


def _write_corpus(root):
    """a.xlsx: baseline says rows 0,1 match, 2 no, 3 (a lift) no.
    b.xlsx: baseline row 0 match (not a lift), row 1 no.
    c_rationale.jsonl is a stale baseline whose workbook is missing."""
    (root / "Documents").mkdir()
    (root / "logs").mkdir()

    def book(name, texts):
        write_synthetic_workbook(
            root / "Documents" / name, [], ["ID", "PRODUCT DETAILS", *_PAD],
            [[i, t, *["x"] * 18] for i, t in enumerate(texts)],
        )

    book("a.xlsx", ["scissor lift A", "scissor lift B", "usb cable", "boom lift X"])
    book("b.xlsx", ["work platform", "usb hub"])

    def log(stem, source, verdicts):
        lines = [{"source_file": source, "row_id": i, "match": m, "rationale": "r"} for i, m in enumerate(verdicts)]
        (root / "logs" / f"{stem}_rationale.jsonl").write_text("\n".join(map(json.dumps, lines)) + "\n")

    log("a", "a.xlsx", [True, True, False, False])
    log("b", "b.xlsx", [True, False])
    log("c", "c.xlsx", [True])


def _config(root):
    return RunConfig(project_root=root, input_dir="Documents", logs_dir="logs", keywords=["lift"])


def test_scores_every_row_and_reports_agreement_with_baseline(tmp_path):
    _write_corpus(tmp_path)
    scorer = FakeScorer(_lift_score)

    result = evaluate_full_corpus(scorer, _config(tmp_path))

    assert len(scorer.seen) == 6  # every row of every available file
    assert result.overall.total == 6
    # a: rows 0,1,2 agree, row 3 disagrees (scorer yes, baseline no)
    # b: row 0 disagrees (scorer no, baseline yes), row 1 agrees
    assert (result.overall.true_positives, result.overall.false_positives) == (2, 1)
    assert (result.overall.false_negatives, result.overall.true_negatives) == (1, 2)
    assert result.overall.accuracy == pytest.approx(4 / 6)  # i.e. the agreement rate
    assert result.per_file["a.xlsx"].accuracy == 0.75
    assert result.per_file["b.xlsx"].accuracy == 0.5


def test_files_with_missing_workbook_are_skipped_and_recorded(tmp_path):
    _write_corpus(tmp_path)

    result = evaluate_full_corpus(FakeScorer(_lift_score), _config(tmp_path))

    assert "c.xlsx" not in result.per_file
    assert [s[0] for s in result.skipped] == ["c.xlsx"]


def test_fixed_half_cut_is_applied(tmp_path):
    _write_corpus(tmp_path)

    result = evaluate_full_corpus(FakeScorer(lambda t: 0.5), _config(tmp_path))

    assert result.overall.true_positives + result.overall.false_positives == 6  # 0.5 counts as yes


def test_report_and_markdown_say_agreement_never_accuracy(tmp_path):
    _write_corpus(tmp_path)
    result = evaluate_full_corpus(FakeScorer(_lift_score), _config(tmp_path))

    report = build_agreement_report(result, ["lift"])
    md = render_agreement_markdown(report)

    assert report["mode"] == "agreement_with_baseline_judge"
    assert report["rows_scored"] == 6
    assert report["agreement_rate"] == pytest.approx(4 / 6)
    assert "accuracy" not in json.dumps(report).lower()
    assert "agreement with the baseline judge" in md.lower()
    assert "accuracy" not in md.lower().replace("never accuracy", "")
    assert "verdict" not in report  # agreement never feeds the GO gate


def _write_config(root):
    (root / "config").mkdir()
    (root / "config" / "run_config.yaml").write_text("keywords: [lift]\n")
    return str(root / "config" / "run_config.yaml")


def test_cli_full_corpus_flag_writes_agreement_report(tmp_path, monkeypatch, capsys):
    _write_corpus(tmp_path)
    cfg = _write_config(tmp_path)
    monkeypatch.setattr(run_system1_bakeoff, "build_scorers", lambda names, kw: [FakeScorer(_lift_score)])

    rc = run_system1_bakeoff.main(["--config", cfg, "--full-corpus", "--candidates", "fake"])

    assert rc == 0
    out = json.loads((tmp_path / "logs" / "system1_bakeoff_agreement.json").read_text())
    assert out["rows_scored"] == 6
    assert "agreement with the Baseline judge" in capsys.readouterr().out


def test_cli_is_off_by_default(tmp_path, monkeypatch):
    _write_corpus(tmp_path)
    cfg = _write_config(tmp_path)
    monkeypatch.setattr(run_system1_bakeoff, "build_scorers", lambda names, kw: pytest.fail("must not score"))

    (tmp_path / "logs" / "ground_truth_labels.jsonl").write_text("")
    (tmp_path / "logs" / "ground_truth_sample.jsonl").write_text("")

    rc = run_system1_bakeoff.main(["--config", cfg])  # empty ground truth -> the normal path bails

    assert rc == 1
    assert not (tmp_path / "logs" / "system1_bakeoff_agreement.json").exists()


# --- review fixes: failed rows, empty logs, raising scorers, gate-ineligible ---


def test_nan_scores_are_failed_rows_counted_as_disagreement(tmp_path):
    _write_corpus(tmp_path)
    scorer = FakeScorer(lambda t: float("nan") if "usb" in t else _lift_score(t))
    report = build_agreement_report(evaluate_full_corpus(scorer, _config(tmp_path)), ["lift"])
    md = render_agreement_markdown(report)

    # failed: "usb cable" (baseline no) and "usb hub" (baseline no) -> each a disagreement
    assert report["failed_rows"] == 2
    assert report["rows"] == 6
    assert report["agree"] == 2  # only a.xlsx rows 0,1 ("scissor lift A/B") agree
    assert "2 row(s) failed" in md and "disagreement" in md


def test_empty_rationale_file_is_skipped_not_fatal(tmp_path):
    _write_corpus(tmp_path)
    (tmp_path / "logs" / "empty_rationale.jsonl").write_text("\n")

    result = evaluate_full_corpus(FakeScorer(_lift_score), _config(tmp_path))

    assert ("empty_rationale.jsonl", "empty rationale file") in result.skipped
    assert result.overall.total == 6


def test_cli_scorer_that_raises_exits_nonzero_with_message(tmp_path, monkeypatch, capsys):
    _write_corpus(tmp_path)
    cfg = _write_config(tmp_path)
    scorer = FakeScorer(_lift_score)

    def boom(texts):
        raise RuntimeError("weights missing")

    scorer.score = boom
    monkeypatch.setattr(run_system1_bakeoff, "build_scorers", lambda names, kw: [scorer])

    rc = run_system1_bakeoff.main(["--config", cfg, "--full-corpus", "--candidates", "fake"])

    assert rc == 1
    assert "weights missing" in capsys.readouterr().err
    assert not (tmp_path / "logs" / "system1_bakeoff_agreement.json").exists()


def test_cli_refuses_gate_ineligible_candidate(tmp_path, monkeypatch, capsys):
    _write_corpus(tmp_path)
    cfg = _write_config(tmp_path)
    scorer = FakeScorer(_lift_score)
    scorer.gate_eligible = False
    monkeypatch.setattr(run_system1_bakeoff, "build_scorers", lambda names, kw: [scorer])

    rc = run_system1_bakeoff.main(["--config", cfg, "--full-corpus", "--candidates", "fake"])

    assert rc == 2
    assert "gate-ineligible" in capsys.readouterr().err
    assert scorer.seen == []
