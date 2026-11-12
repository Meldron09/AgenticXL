"""Failure isolation and duplicate-text sharing in the bake-off driver (issue #13).
Fake scorers only."""

from __future__ import annotations

import json

from scripts.build_ground_truth_report import SampleRow
from scripts.run_system1_bakeoff import (
    build_report,
    evaluate_candidate,
    render_markdown,
    run_bakeoff,
    score_texts,
)


class FakeScorer:
    template = "Is this about: {keywords}"

    def __init__(self, name, score_fn=None, gate_eligible=True, scores_fn=None, init_error=None):
        self.name = name
        self.model_id = f"fake/{name}"
        self.revision = "rev0"
        self.gate_eligible = gate_eligible
        self._score_fn = score_fn
        self._scores_fn = scores_fn
        self.calls: list[list[str]] = []

    def score(self, texts):
        self.calls.append(list(texts))
        if self._scores_fn:
            return self._scores_fn(texts)
        return [self._score_fn(t) for t in texts]


def _row(i, text=None):
    return SampleRow(
        sample_index=i, source_file="a.xlsx", row_id=i, quadrant="both_match",
        product_details_text=text if text is not None else f"row {i}",
        laya_score=0.0, laya_match=False, llm_match=True, llm_rationale="",
    )


ROWS = [_row(i) for i in range(10)]
LABELS = {i: (i < 8, "") for i in range(10)}
POOLS = {"both_match": 10, "both_no_match": 0, "laya_false_negative": 0, "laya_false_positive": 0}


def _oracle(text):
    return 0.9 if int(text.split()[1]) < 8 else 0.1


def _boom(_texts):
    raise RuntimeError("weights missing")


# --- could not run ---


def test_candidate_that_raises_is_could_not_run_and_others_complete():
    bad = FakeScorer("bad", scores_fn=_boom)
    good = FakeScorer("good", _oracle)
    results = run_bakeoff([bad, good, FakeScorer("good2", _oracle)], ROWS, LABELS, POOLS)

    assert [r.name for r in results] == ["bad", "good", "good2"]
    assert results[0].status == "could_not_run"
    assert "weights missing" in results[0].error
    assert results[0].verdict == "COULD-NOT-RUN"
    assert [r.status for r in results[1:]] == ["ok", "ok"]
    assert results[1].verdict == "GO"


def test_wrong_score_count_is_could_not_run():
    scorer = FakeScorer("short", scores_fn=lambda texts: [0.9])
    (result,) = run_bakeoff([scorer], ROWS, LABELS, POOLS)

    assert result.status == "could_not_run"
    assert "short" in result.error


def test_could_not_run_is_reported_with_reason_in_json_and_markdown():
    results = run_bakeoff([FakeScorer("bad", scores_fn=_boom), FakeScorer("good", _oracle)], ROWS, LABELS, POOLS)
    report = build_report(results, keywords=["scissor"], sample_size=10, quadrant_pool_sizes=POOLS)
    md = render_markdown(report)

    bad = report["candidates"][0]
    assert bad["status"] == "could_not_run" and "weights missing" in bad["error"]
    assert json.loads(json.dumps(report)) == report
    assert "could not run" in md.lower() and "weights missing" in md
    assert "| bad | COULD-NOT-RUN | n/a |" in md  # no fabricated 0% metrics


# --- per-row bad scores ---


def test_non_finite_or_out_of_range_scores_fail_only_that_row():
    bad_by_text = {"row 0": float("nan"), "row 1": float("inf"), "row 2": 1.5, "row 3": -0.1}
    scorer = FakeScorer("odd", lambda t: bad_by_text.get(t, _oracle(t)))
    result = evaluate_candidate(scorer, ROWS, LABELS, POOLS)

    assert result.status == "ok"
    assert sorted(result.failed_rows) == [0, 1, 2, 3]
    assert set(result.scores) == set(range(4, 10))  # only valid scores kept
    # each failed row is still in the metrics, counted as wrong (4 positives missed)
    assert result.raw.total == 10
    assert result.raw.false_negatives == 4
    assert result.raw.accuracy == 0.6
    assert result.auroc == 1.0  # computed from the valid rows only


def test_failed_negative_row_counts_as_wrong_too():
    scorer = FakeScorer("odd", lambda t: float("nan") if t == "row 9" else _oracle(t))
    result = evaluate_candidate(scorer, ROWS, LABELS, POOLS)

    assert result.failed_rows == [9]
    assert result.raw.false_positives == 1


def test_non_numeric_score_is_a_row_failure_not_a_crash():
    scorer = FakeScorer("odd", lambda t: None if t == "row 0" else _oracle(t))
    result = evaluate_candidate(scorer, ROWS, LABELS, POOLS)

    assert result.failed_rows == [0]


def test_row_failures_are_surfaced_in_report():
    scorer = FakeScorer("odd", lambda t: float("nan") if t == "row 0" else _oracle(t))
    results = run_bakeoff([scorer, FakeScorer("clean", _oracle)], ROWS, LABELS, POOLS)
    report = build_report(results, keywords=["scissor"], sample_size=10, quadrant_pool_sizes=POOLS)
    md = render_markdown(report)

    assert report["candidates"][0]["failed_rows"] == [0]
    assert report["candidates"][1]["failed_rows"] == []
    assert "odd" in md and "1 row(s) failed" in md and "[0]" in md


def test_all_clean_run_has_no_failure_section_noise():
    results = run_bakeoff([FakeScorer("clean", _oracle)], ROWS, LABELS, POOLS)
    md = render_markdown(build_report(results, ["x"], 10, POOLS))

    assert "could not run" not in md.lower()
    assert "row(s) failed" not in md


# --- duplicate texts ---


def test_identical_texts_are_scored_once_and_share_the_score():
    scorer = FakeScorer("dup", lambda t: 0.9 if t == "same" else 0.1)
    scores = score_texts(scorer, ["same", "other", "same", "same"])

    assert scorer.calls == [["same", "other"]]  # unique texts, first-seen order, one call
    assert scores == [0.9, 0.1, 0.9, 0.9]


def test_duplicate_rows_are_still_scored_individually_in_metrics():
    rows = [_row(0, "same"), _row(1, "same"), _row(2, "same"), _row(3, "other")]
    labels = {0: (True, ""), 1: (False, ""), 2: (True, ""), 3: (False, "")}
    pools = {"both_match": 4, "both_no_match": 0, "laya_false_negative": 0, "laya_false_positive": 0}
    scorer = FakeScorer("dup", lambda t: 0.9 if t == "same" else 0.1)

    result = evaluate_candidate(scorer, rows, labels, pools)

    assert scorer.calls == [["same", "other"]]
    assert result.raw.total == 4
    assert (result.raw.true_positives, result.raw.false_positives, result.raw.true_negatives) == (2, 1, 1)
    assert set(result.scores) == {0, 1, 2, 3}


def test_a_bad_score_on_a_duplicated_text_fails_every_row_sharing_it():
    rows = [_row(0, "same"), _row(1, "same"), _row(2, "other")]
    labels = {0: (True, ""), 1: (True, ""), 2: (False, "")}
    pools = {"both_match": 3, "both_no_match": 0, "laya_false_negative": 0, "laya_false_positive": 0}
    scorer = FakeScorer("dup", lambda t: float("nan") if t == "same" else 0.1)

    result = evaluate_candidate(scorer, rows, labels, pools)

    assert sorted(result.failed_rows) == [0, 1]


# --- JSON must not publish placeholder / meaningless metrics (review fixes) ---


def test_could_not_run_json_has_null_metrics_and_downstream_survives():
    results = run_bakeoff([FakeScorer("bad", scores_fn=_boom), FakeScorer("good", _oracle)], ROWS, LABELS, POOLS)
    report = build_report(results, ["scissor"], 10, POOLS, ROWS, LABELS)
    bad = report["candidates"][0]

    assert bad["raw"] is None and bad["population_weighted"] is None and bad["auroc"] is None
    assert report["candidates"][1]["raw"]["total"] == 10
    assert "| bad | COULD-NOT-RUN | n/a |" in render_markdown(report)


def test_gate_ineligible_json_publishes_auroc_only():
    (result,) = run_bakeoff([FakeScorer("ref", _oracle, gate_eligible=False)], ROWS, LABELS, POOLS)
    (c,) = build_report([result], ["scissor"], 10, POOLS)["candidates"]

    assert c["raw"] is None and c["population_weighted"] is None
    assert c["auroc"] == 1.0 and c["verdict"] == "AUROC-only"
