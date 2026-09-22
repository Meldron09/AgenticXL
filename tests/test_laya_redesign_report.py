from __future__ import annotations

import json

import pytest

from agenticxl.filtering.laya_client import LayaClient

from scripts.build_ground_truth_report import Quadrant, SampleRow
from scripts.build_laya_redesign_report import (
    BASELINE_POPULATION_ACCURACY,
    GO_ACCURACY_THRESHOLD,
    build_variant_report,
    determine_verdict,
    effective_batch_size,
    run_variant_and_report,
    run_variant_scores,
    score_variant_at_threshold,
    write_variant_score_dump,
)


class StubAgent:
    """Minimal stand-in for a loaded laya_mlx.Agent, matching the one in
    tests/test_laya_client.py: records the (state, questions) it was called
    with and returns a canned predict() result."""

    def __init__(self, responder=None):
        self.calls: list[tuple[object, dict]] = []
        self._responder = responder

    def predict(self, state, questions):
        self.calls.append((state, questions))
        if self._responder is None:
            return {"answers": {}}
        return self._responder(state, questions)


def _noul_answer(score: float) -> dict:
    return {"type": "noul", "noul": score, "confidence": max(score, 1 - score)}


def _sample_row(**overrides) -> SampleRow:
    defaults = dict(
        sample_index=0,
        source_file="a.xlsx",
        row_id=0,
        quadrant="both_match",
        product_details_text="a scissor lift",
        laya_score=0.9,
        laya_match=True,
        llm_match=True,
        llm_rationale="scissor lift matches 'scissor'",
    )
    defaults.update(overrides)
    return SampleRow(**defaults)


# --- effective_batch_size ---


def test_effective_batch_size_combined_mode_unchanged_regardless_of_keyword_count():
    assert effective_batch_size(50, "combined", keyword_count=6) == 50
    assert effective_batch_size(50, "combined", keyword_count=1) == 50


def test_effective_batch_size_per_keyword_divides_and_floors():
    assert effective_batch_size(50, "per_keyword", keyword_count=6) == 8  # 50 // 6 == 8


def test_effective_batch_size_per_keyword_minimum_is_one():
    assert effective_batch_size(3, "per_keyword", keyword_count=10) == 1


# --- determine_verdict ---


def test_determine_verdict_go_when_at_or_above_gate():
    assert determine_verdict(0.95) == "GO"
    assert determine_verdict(0.99) == "GO"


def test_determine_verdict_no_go_when_no_meaningful_movement():
    assert determine_verdict(BASELINE_POPULATION_ACCURACY) == "NO-GO"


def test_determine_verdict_iterate_when_gap_meaningfully_closed():
    gap = GO_ACCURACY_THRESHOLD - BASELINE_POPULATION_ACCURACY
    accuracy = BASELINE_POPULATION_ACCURACY + 0.5 * gap
    assert determine_verdict(accuracy) == "iterate"


def test_determine_verdict_no_go_just_under_the_meaningful_closure_floor():
    gap = GO_ACCURACY_THRESHOLD - BASELINE_POPULATION_ACCURACY
    accuracy = BASELINE_POPULATION_ACCURACY + 0.24 * gap
    assert determine_verdict(accuracy) == "NO-GO"


def test_determine_verdict_iterate_right_at_the_meaningful_closure_floor():
    gap = GO_ACCURACY_THRESHOLD - BASELINE_POPULATION_ACCURACY
    accuracy = BASELINE_POPULATION_ACCURACY + 0.25 * gap
    assert determine_verdict(accuracy) == "iterate"


# --- run_variant_scores ---


def test_run_variant_scores_chunks_by_batch_size_for_legacy_layout():
    agent = StubAgent(responder=lambda state, questions: {"answers": {qid: _noul_answer(0.7) for qid in questions}})
    client = LayaClient(agent=agent)

    rows = [(i, f"text {i}") for i in range(5)]
    scores = run_variant_scores(
        client, rows, ["k"], batch_size=2, question_mode="combined", payload_layout="legacy"
    )

    assert len(agent.calls) == 3  # chunks of 2, 2, 1
    assert set(scores) == {0, 1, 2, 3, 4}
    assert all(score == 0.7 for score in scores.values())


def test_run_variant_scores_passes_question_mode_and_payload_layout_through():
    agent = StubAgent(responder=lambda state, questions: {"answers": {qid: _noul_answer(0.9) for qid in questions}})
    client = LayaClient(agent=agent)

    rows = [(0, "widget A")]
    run_variant_scores(
        client, rows, ["widget", "gadget"], batch_size=50, question_mode="per_keyword", payload_layout="legacy"
    )

    (state, questions), = agent.calls
    assert state == {"keywords": ["widget", "gadget"]}
    assert set(questions) == {"0::widget", "0::gadget"}  # per_keyword shape, not combined


def test_run_variant_scores_raw_score_is_threshold_independent():
    agent = StubAgent(responder=lambda state, questions: {"answers": {"0": _noul_answer(0.2)}})
    client = LayaClient(agent=agent)

    scores = run_variant_scores(
        client, [(0, "a")], ["k"], batch_size=50, question_mode="combined", payload_layout="legacy"
    )

    assert scores[0] == 0.2  # raw score, never pre-thresholded


# --- score_variant_at_threshold ---


def test_score_variant_at_threshold_reuses_compute_classifier_metrics_for_raw():
    ground_truth = {0: True, 1: False}
    scores = {0: 0.9, 1: 0.9}  # both above threshold -> row 1 is a false positive
    sample_rows_by_index = {
        0: _sample_row(sample_index=0, quadrant="both_match"),
        1: _sample_row(sample_index=1, quadrant="both_no_match", llm_match=False),
    }
    ground_truth_labels = {0: (True, "r0"), 1: (False, "r1")}
    quadrant_pool_sizes: dict[Quadrant, int] = {
        "both_match": 1,
        "both_no_match": 1,
        "laya_false_negative": 0,
        "laya_false_positive": 0,
    }

    result = score_variant_at_threshold(
        ground_truth, scores, 0.5, sample_rows_by_index, ground_truth_labels, quadrant_pool_sizes
    )

    assert result.threshold == 0.5
    assert result.raw.true_positives == 1
    assert result.raw.false_positives == 1
    assert result.raw.accuracy == 0.5


def test_score_variant_at_threshold_population_weighted_reweights_by_pool_size():
    # 9 sampled both_match rows (true pool 900) correctly predicted; 1 sampled
    # laya_false_negative row (true pool 100) incorrectly predicted at this
    # threshold -- population-weighted accuracy should read ~90%, not ~90/91%.
    sample_rows_by_index: dict[int, SampleRow] = {}
    ground_truth: dict[int, bool] = {}
    ground_truth_labels: dict[int, tuple[bool, str]] = {}
    scores: dict[int, float] = {}
    for i in range(9):
        sample_rows_by_index[i] = _sample_row(sample_index=i, quadrant="both_match", llm_match=True)
        ground_truth[i] = True
        ground_truth_labels[i] = (True, "relevant")
        scores[i] = 0.9
    idx = 100
    sample_rows_by_index[idx] = _sample_row(sample_index=idx, quadrant="laya_false_negative", llm_match=True)
    ground_truth[idx] = True
    ground_truth_labels[idx] = (True, "relevant")
    scores[idx] = 0.1

    quadrant_pool_sizes: dict[Quadrant, int] = {
        "both_match": 900,
        "both_no_match": 0,
        "laya_false_negative": 100,
        "laya_false_positive": 0,
    }

    result = score_variant_at_threshold(
        ground_truth, scores, 0.5, sample_rows_by_index, ground_truth_labels, quadrant_pool_sizes
    )

    assert result.population_weighted.accuracy == pytest.approx(0.9)


# --- build_variant_report ---


def test_build_variant_report_sweeps_threshold_and_scores_against_ground_truth():
    ground_truth = {0: True, 1: True, 2: False}
    scores = {0: 0.9, 1: 0.8, 2: 0.1}
    sample_rows_by_index = {
        0: _sample_row(sample_index=0, row_id=0, quadrant="both_match", llm_match=True),
        1: _sample_row(sample_index=1, row_id=1, quadrant="both_match", llm_match=True),
        2: _sample_row(sample_index=2, row_id=2, quadrant="both_no_match", llm_match=False),
    }
    ground_truth_labels = {0: (True, "r"), 1: (True, "r"), 2: (False, "r")}
    quadrant_pool_sizes: dict[Quadrant, int] = {
        "both_match": 2,
        "both_no_match": 1,
        "laya_false_negative": 0,
        "laya_false_positive": 0,
    }

    report = build_variant_report(
        "combined",
        "legacy",
        50,
        ground_truth,
        scores,
        sample_rows_by_index,
        ground_truth_labels,
        quadrant_pool_sizes,
        legacy_threshold=0.65,
    )

    assert report.question_mode == "combined"
    assert report.payload_layout == "legacy"
    assert report.effective_batch_size == 50
    assert report.swept_best.raw.accuracy == 1.0
    assert report.swept_best.population_weighted.accuracy == 1.0
    assert report.legacy.threshold == 0.65
    assert report.verdict == "GO"


def test_build_variant_report_verdict_is_no_go_when_variant_performs_at_baseline():
    ground_truth = {0: True, 1: False}
    scores = {0: 0.1, 1: 0.9}  # both wrong at any reasonable threshold
    sample_rows_by_index = {
        0: _sample_row(sample_index=0, row_id=0, quadrant="laya_false_negative", llm_match=True),
        1: _sample_row(sample_index=1, row_id=1, quadrant="laya_false_positive", llm_match=False),
    }
    ground_truth_labels = {0: (True, "r"), 1: (False, "r")}
    quadrant_pool_sizes: dict[Quadrant, int] = {
        "both_match": 0,
        "both_no_match": 0,
        "laya_false_negative": 1,
        "laya_false_positive": 1,
    }

    report = build_variant_report(
        "per_keyword",
        "convention",
        8,
        ground_truth,
        scores,
        sample_rows_by_index,
        ground_truth_labels,
        quadrant_pool_sizes,
    )

    assert report.verdict == "NO-GO"


# --- write_variant_score_dump ---


def test_write_variant_score_dump_remaps_sample_index_to_source_file_and_row_id(tmp_path):
    sample_rows = [
        _sample_row(sample_index=10, source_file="b.xlsx", row_id=3),
        _sample_row(sample_index=11, source_file="a.xlsx", row_id=1),
    ]
    scores = {10: 0.8, 11: 0.2}
    path = tmp_path / "combined_legacy.jsonl"

    result_path = write_variant_score_dump(
        path,
        sample_rows,
        scores,
        threshold=0.5,
        model_identifier="org/model@rev",
        question_mode="combined",
        payload_layout="legacy",
    )

    assert result_path == path
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == [
        {
            "source_file": "a.xlsx",
            "row_id": 1,
            "score": 0.2,
            "match": False,
            "model": "org/model@rev",
            "question_mode": "combined",
            "payload_layout": "legacy",
            "threshold": 0.5,
        },
        {
            "source_file": "b.xlsx",
            "row_id": 3,
            "score": 0.8,
            "match": True,
            "model": "org/model@rev",
            "question_mode": "combined",
            "payload_layout": "legacy",
            "threshold": 0.5,
        },
    ]


def test_write_variant_score_dump_paths_for_different_variants_never_collide(tmp_path):
    sample_rows = [_sample_row(sample_index=0, source_file="a.xlsx", row_id=0)]
    scores = {0: 0.9}

    path_a = write_variant_score_dump(
        tmp_path / "combined_legacy.jsonl", sample_rows, scores, 0.5, "m", "combined", "legacy"
    )
    path_b = write_variant_score_dump(
        tmp_path / "per_keyword_convention.jsonl", sample_rows, scores, 0.5, "m", "per_keyword", "convention"
    )

    assert path_a != path_b
    assert path_a.exists() and path_b.exists()
    record_a = json.loads(path_a.read_text())
    record_b = json.loads(path_b.read_text())
    assert record_a["question_mode"] == "combined" and record_a["payload_layout"] == "legacy"
    assert record_b["question_mode"] == "per_keyword" and record_b["payload_layout"] == "convention"


# --- run_variant_and_report (glue) ---


def test_run_variant_and_report_end_to_end_with_a_stub_agent(tmp_path):
    agent = StubAgent(responder=lambda state, questions: {"answers": {qid: _noul_answer(0.9) for qid in questions}})
    client = LayaClient(agent=agent, model_id="org/model", revision="rev1")

    labeled_rows = [
        _sample_row(sample_index=0, source_file="a.xlsx", row_id=0, quadrant="both_match", llm_match=True),
        _sample_row(sample_index=1, source_file="a.xlsx", row_id=1, quadrant="both_no_match", llm_match=False),
    ]
    ground_truth_labels = {0: (True, "r0"), 1: (False, "r1")}
    quadrant_pool_sizes: dict[Quadrant, int] = {
        "both_match": 1,
        "both_no_match": 1,
        "laya_false_negative": 0,
        "laya_false_positive": 0,
    }

    report = run_variant_and_report(
        client,
        labeled_rows,
        ["widget"],
        configured_batch_size=50,
        question_mode="combined",
        payload_layout="legacy",
        ground_truth_labels=ground_truth_labels,
        quadrant_pool_sizes=quadrant_pool_sizes,
        scores_dir=tmp_path,
    )

    assert report.effective_batch_size == 50  # combined mode: unchanged
    scores_path = tmp_path / "combined_legacy.jsonl"
    assert scores_path.exists()
    lines = [json.loads(line) for line in scores_path.read_text().splitlines()]
    assert len(lines) == 2
    assert all(line["question_mode"] == "combined" and line["payload_layout"] == "legacy" for line in lines)


def test_run_variant_and_report_uses_effective_batch_size_for_per_keyword(tmp_path):
    agent = StubAgent(responder=lambda state, questions: {"answers": {qid: _noul_answer(0.9) for qid in questions}})
    client = LayaClient(agent=agent)

    labeled_rows = [_sample_row(sample_index=0, source_file="a.xlsx", row_id=0)]

    report = run_variant_and_report(
        client,
        labeled_rows,
        ["widget", "gadget", "gizmo"],  # 3 keywords
        configured_batch_size=10,
        question_mode="per_keyword",
        payload_layout="legacy",
        ground_truth_labels={0: (True, "r")},
        quadrant_pool_sizes={"both_match": 1, "both_no_match": 0, "laya_false_negative": 0, "laya_false_positive": 0},
        scores_dir=tmp_path,
    )

    assert report.effective_batch_size == 3  # 10 // 3 == 3
