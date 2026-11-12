"""Bake-off driver tests (issue #12): everything goes through the Scorer seam
with fake scorers -- no network, no model download, no torch/transformers."""

from __future__ import annotations

import json
import sys

import pytest

from scripts.build_ground_truth_report import SampleRow
from scripts.run_system1_bakeoff import (
    GO_MIN_POPULATION_ACCURACY,
    GO_MIN_RECALL,
    THRESHOLD,
    auroc,
    build_report,
    determine_verdict,
    evaluate_candidate,
    render_markdown,
    run_bakeoff,
    write_report,
)
from scripts import bakeoff_scorers
from scripts.bakeoff_scorers import CANDIDATES, build_scorers, fill_template


class FakeScorer:
    template = "Is this about: {keywords}"

    def __init__(self, name, score_fn, gate_eligible=True):
        self.name = name
        self.model_id = f"fake/{name}"
        self.revision = "rev0"
        self.gate_eligible = gate_eligible
        self._score_fn = score_fn
        self.seen: list[str] = []

    def score(self, texts):
        self.seen.extend(texts)
        return [self._score_fn(t) for t in texts]


def _row(i, quadrant="both_match", text=None):
    return SampleRow(
        sample_index=i,
        source_file="a.xlsx",
        row_id=i,
        quadrant=quadrant,
        product_details_text=text if text is not None else f"row {i}",
        laya_score=0.0,
        laya_match=False,
        llm_match=True,
        llm_rationale="",
    )


# 10 rows, 8 positive + 2 negative, all one quadrant (so weights are 1).
ROWS = [_row(i) for i in range(10)]
LABELS = {i: (i < 8, "") for i in range(10)}
POOLS = {"both_match": 10, "both_no_match": 0, "laya_false_negative": 0, "laya_false_positive": 0}


def _oracle(text):
    return 0.9 if int(text.split()[1]) < 8 else 0.1


def test_perfect_scorer_is_go_with_perfect_metrics():
    result = evaluate_candidate(FakeScorer("oracle", _oracle), ROWS, LABELS, POOLS)

    assert result.status == "ok"
    assert result.raw.accuracy == 1.0
    assert result.population_weighted.accuracy == 1.0
    assert result.population_weighted.recall == 1.0
    assert result.auroc == 1.0
    assert result.verdict == "GO"


def test_cut_is_fixed_at_half_and_inclusive():
    assert THRESHOLD == 0.5
    result = evaluate_candidate(FakeScorer("edge", lambda t: 0.5), ROWS, LABELS, POOLS)
    # every row predicted positive at exactly 0.5
    assert result.raw.true_positives == 8
    assert result.raw.false_positives == 2


def test_high_auroc_does_not_rescue_a_candidate_that_misses_at_half():
    # Perfectly separable but everything sits below 0.5 -> recall 0, AUROC 1.
    scorer = FakeScorer("miscalibrated", lambda t: 0.4 if int(t.split()[1]) < 8 else 0.1)
    result = evaluate_candidate(scorer, ROWS, LABELS, POOLS)

    assert result.auroc == 1.0
    assert result.population_weighted.recall == 0.0
    assert result.verdict == "NO-GO"


def test_recall_floor_fails_even_with_accuracy_above_gate():
    # 100 positives + 100 negatives... use pool-weighted: misses 1 of 100 positives (99% recall passes),
    # misses 3 of 100 (97% recall fails) while accuracy stays >= 98.5%.
    assert determine_verdict(0.985, 0.97, gate_eligible=True) == "NO-GO"
    assert determine_verdict(0.985, 0.98, gate_eligible=True) == "GO"
    assert determine_verdict(0.949, 1.0, gate_eligible=True) == "NO-GO"
    assert (GO_MIN_POPULATION_ACCURACY, GO_MIN_RECALL) == (0.95, 0.98)


def test_non_gate_eligible_candidate_never_goes():
    result = evaluate_candidate(FakeScorer("ref", _oracle, gate_eligible=False), ROWS, LABELS, POOLS)

    assert result.population_weighted.accuracy == 1.0
    assert result.verdict == "AUROC-only"
    assert result.auroc == 1.0


def test_population_weighting_is_applied_to_scorer_predictions():
    # Quadrant A (sample 1 row, wrongly predicted negative) stands for 90 corpus rows;
    # quadrant B (sample 1 row, correctly positive) stands for 10.
    rows = [_row(0, "laya_false_negative", "row 0"), _row(1, "both_match", "row 1")]
    labels = {0: (True, ""), 1: (True, "")}
    pools = {"laya_false_negative": 90, "both_match": 10, "both_no_match": 0, "laya_false_positive": 0}
    scorer = FakeScorer("s", lambda t: 0.1 if t == "row 0" else 0.9)

    result = evaluate_candidate(scorer, rows, labels, pools)

    assert result.raw.recall == pytest.approx(0.5)
    assert result.population_weighted.recall == pytest.approx(0.1)


def test_scorer_is_called_once_with_row_texts_in_sample_order():
    scorer = FakeScorer("oracle", _oracle)
    evaluate_candidate(scorer, ROWS, LABELS, POOLS)
    assert scorer.seen == [f"row {i}" for i in range(10)]


def test_auroc_basic_ties_and_degenerate():
    assert auroc([True, True, False, False], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert auroc([True, True, False, False], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert auroc([True, False], [0.5, 0.5]) == 0.5
    assert auroc([True, False, True, False], [0.8, 0.4, 0.4, 0.4]) == pytest.approx(0.75)
    assert auroc([True, True], [0.1, 0.2]) is None  # one class only


def test_run_bakeoff_returns_one_result_per_scorer_in_order():
    scorers = [FakeScorer("a", _oracle), FakeScorer("b", lambda t: 0.1)]
    results = run_bakeoff(scorers, ROWS, LABELS, POOLS)
    assert [r.name for r in results] == ["a", "b"]
    assert [r.verdict for r in results] == ["GO", "NO-GO"]


def _report(keywords=("scissor", "boom lift")):
    results = run_bakeoff(
        [FakeScorer("a", _oracle), FakeScorer("ref", _oracle, gate_eligible=False)], ROWS, LABELS, POOLS
    )
    return build_report(results, keywords=list(keywords), sample_size=len(ROWS), quadrant_pool_sizes=POOLS)


def test_report_records_template_model_revision_keywords_and_provenance():
    report = _report()
    cand = report["candidates"][0]

    assert report["threshold"] == 0.5
    assert report["gate"] == {"population_accuracy": 0.95, "recall": 0.98}
    assert report["keywords"] == ["scissor", "boom lift"]
    assert (cand["name"], cand["model_id"], cand["revision"]) == ("a", "fake/a", "rev0")
    assert cand["template"] == "Is this about: {keywords}"
    assert cand["verdict"] == "GO"
    assert "526" not in json.dumps(report)  # size comes from the data, not hardcoded
    assert report["sample_size"] == 10


def test_markdown_has_row_per_candidate_verdict_and_caveats():
    md = render_markdown(_report())

    assert "| a |" in md and "| ref |" in md
    assert "GO" in md and "AUROC-only" in md
    assert "fake/a" in md and "rev0" in md
    assert "Is this about: {keywords}" in md
    assert "Claude-labelled" in md
    assert "stratified" in md
    assert "scissor" in md


def test_report_round_trips_through_write_report(tmp_path):
    report = _report()
    json_path, md_path = write_report(report, tmp_path / "r.json", tmp_path / "r.md")

    assert json.loads(json_path.read_text()) == report
    assert md_path.read_text() == render_markdown(report)


def test_fill_template_uses_config_keywords():
    assert fill_template("Is this about: {keywords}", ["scissor", "boom lift"]) == "Is this about: scissor; boom lift"
    assert fill_template("Is this about: {keywords}", ["x"]) == "Is this about: x"


def test_registry_builds_scorers_by_name_from_keywords_without_loading_models():
    assert "qwen3-reranker-0.6b" in CANDIDATES
    (scorer,) = build_scorers(["qwen3-reranker-0.6b"], ["scissor"])

    assert scorer.name == "qwen3-reranker-0.6b"
    assert scorer.model_id == "Qwen/Qwen3-Reranker-0.6B"
    assert len(scorer.revision) == 40
    assert "{keywords}" in scorer.template
    assert scorer.gate_eligible is True


def test_unknown_candidate_name_is_rejected():
    with pytest.raises(ValueError, match="nope"):
        build_scorers(["nope"], ["scissor"])


def test_importing_bakeoff_modules_does_not_import_model_runtimes():
    assert bakeoff_scorers  # imported above
    for heavy in ("torch", "transformers", "sentence_transformers", "gliclass"):
        assert heavy not in sys.modules
