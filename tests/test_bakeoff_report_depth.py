"""Report depth and provenance (issue #14): per-file breakdowns, per-candidate
miss lists, and recorded provenance. Fake scorers only -- no models."""

from __future__ import annotations

import json
from dataclasses import replace

from scripts.build_ground_truth_report import SampleRow
from scripts.run_system1_bakeoff import (
    build_report,
    render_markdown,
    run_bakeoff,
    write_report,
)


class FakeScorer:
    template = "Is this about: {keywords}"

    def __init__(self, name, score_fn, gate_eligible=True):
        self.name = name
        self.model_id = f"fake/{name}"
        self.revision = f"rev-{name}"
        self.gate_eligible = gate_eligible
        self._score_fn = score_fn

    def score(self, texts):
        return [self._score_fn(t) for t in texts]


def _row(i, source_file, text):
    return SampleRow(
        sample_index=i, source_file=source_file, row_id=i, quadrant="both_match",
        product_details_text=text, laya_score=0.0, laya_match=False, llm_match=True, llm_rationale="",
    )


# a.xlsx: 0 pos, 1 pos, 2 neg ; b.xlsx: 3 pos, 4 neg
ROWS = [
    _row(0, "a.xlsx", "scissor lift platform"),
    _row(1, "a.xlsx", "JIB kit for boom"),
    _row(2, "a.xlsx", "cotton fabric"),
    _row(3, "b.xlsx", "spider lift"),
    _row(4, "b.xlsx", "steel | pipe\nfitting"),
]
LABELS = {0: (True, ""), 1: (True, ""), 2: (False, ""), 3: (True, ""), 4: (False, "")}
POOLS = {"both_match": 5, "both_no_match": 0, "laya_false_negative": 0, "laya_false_positive": 0}


def _misses_jib_and_spider(text):
    # misses the jargon positives; false-alarms on the pipe row
    if "JIB" in text or "spider" in text:
        return 0.1
    return 0.9 if "lift" in text or "pipe" in text else 0.1


def _report(scorers=None):
    scorers = scorers or [FakeScorer("weak", _misses_jib_and_spider)]
    results = run_bakeoff(scorers, ROWS, LABELS, POOLS)
    return build_report(results, ["lift"], len(ROWS), POOLS, sample_rows=ROWS, labels=LABELS)


def test_per_file_breakdown_in_json():
    per_file = _report()["per_file"]["weak"]

    assert set(per_file) == {"a.xlsx", "b.xlsx"}
    assert per_file["a.xlsx"]["total"] == 3
    assert per_file["a.xlsx"]["recall"] == 0.5  # 1 of 2 positives (JIB missed)
    assert per_file["b.xlsx"]["total"] == 2
    assert per_file["b.xlsx"]["recall"] == 0.0  # spider missed
    assert per_file["b.xlsx"]["false_positives"] == 1  # pipe row


def test_miss_list_has_kind_score_and_full_text():
    misses = _report()["misses"]["weak"]
    by_index = {m["sample_index"]: m for m in misses}

    assert set(by_index) == {1, 3, 4}  # rows 0 and 2 are correct
    assert by_index[1]["kind"] == "false_negative"
    assert by_index[1]["text"] == "JIB kit for boom"
    assert by_index[1]["score"] == 0.1
    assert by_index[1]["source_file"] == "a.xlsx"
    assert by_index[4]["kind"] == "false_positive"


def test_perfect_candidate_has_empty_miss_list():
    positives = {"scissor lift platform", "JIB kit for boom", "spider lift"}
    oracle = FakeScorer("oracle", lambda t: 0.9 if t in positives else 0.1)

    assert _report([oracle])["misses"]["oracle"] == []


def test_markdown_shows_per_file_and_misses_per_candidate():
    md = render_markdown(_report([FakeScorer("weak", _misses_jib_and_spider), FakeScorer("other", lambda t: 0.9)]))

    assert "## Per-file breakdown" in md
    assert "a.xlsx" in md and "b.xlsx" in md
    assert "## Missed Ground-truth rows" in md
    assert "JIB kit for boom" in md
    assert "spider lift" in md
    assert "false_negative" in md
    assert "cotton fabric" in md  # `other` false-alarms on everything
    # the pipe/newline row is flattened so it cannot break the Markdown
    assert "steel \\| pipe fitting" in md
    assert "steel | pipe\nfitting" not in md


def test_markdown_miss_text_is_truncated_json_is_full():
    long_text = "scissor lift " * 100
    rows = [_row(0, "a.xlsx", long_text)]
    labels = {0: (True, "")}
    pools = {"both_match": 1, "both_no_match": 0, "laya_false_negative": 0, "laya_false_positive": 0}
    results = run_bakeoff([FakeScorer("never", lambda t: 0.0)], rows, labels, pools)
    report = build_report(results, ["lift"], 1, pools, sample_rows=rows, labels=labels)

    assert report["misses"]["never"][0]["text"] == long_text
    assert long_text not in render_markdown(report)


def test_provenance_recorded_for_every_candidate():
    report = _report([FakeScorer("weak", _misses_jib_and_spider), FakeScorer("other", lambda t: 0.9)])
    md = render_markdown(report)

    assert "fake/weak" in md and "rev-weak" in md and "fake/other" in md and "rev-other" in md
    assert "Is this about: {keywords}" in md
    assert report["candidates"][0]["model_id"] == "fake/weak"
    assert report["candidates"][0]["revision"] == "rev-weak"
    assert report["candidates"][0]["template"] == "Is this about: {keywords}"


def test_caveats_state_provenance_size_and_stratification():
    results = run_bakeoff([FakeScorer("weak", _misses_jib_and_spider)], ROWS, LABELS, POOLS)
    md = render_markdown(build_report(results, ["lift"], 526, POOLS))

    assert "Claude-labelled under human supervision" in md
    assert "526 rows" in md
    assert "stratified on Laya / Baseline-judge disagreement" in md
    assert "noisy" in md


def test_candidate_without_scores_is_skipped_not_crashing():
    results = run_bakeoff([FakeScorer("weak", _misses_jib_and_spider)], ROWS, LABELS, POOLS)
    results = [replace(results[0], scores={})]
    report = build_report(results, ["lift"], len(ROWS), POOLS, sample_rows=ROWS, labels=LABELS)

    assert report["misses"]["weak"] == []
    assert report["per_file"]["weak"] == {}
    render_markdown(report)


def test_report_without_rows_still_renders():
    results = run_bakeoff([FakeScorer("weak", _misses_jib_and_spider)], ROWS, LABELS, POOLS)
    report = build_report(results, ["lift"], len(ROWS), POOLS)

    assert "per_file" not in report
    assert "## Per-file breakdown" not in render_markdown(report)


def test_json_and_markdown_written_and_round_trip(tmp_path):
    report = _report()
    json_path, md_path = write_report(report, tmp_path / "logs" / "r.json", tmp_path / "logs" / "r.md")

    assert json.loads(json_path.read_text()) == report
    assert "JIB kit for boom" in md_path.read_text()
