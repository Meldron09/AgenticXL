"""Embedding-similarity AUROC-only reference (issue #19). No model, no network:
the sentence-transformers runtime is replaced by a tiny fake in sys.modules."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from scripts.bakeoff_scorers import CANDIDATES, build_scorers
from scripts.build_ground_truth_report import SampleRow
from scripts.run_system1_bakeoff import (
    build_report,
    determine_verdict,
    evaluate_candidate,
    render_markdown,
)

NAME = "gte-large-en-v1.5"

# Axis 0 = "scissor lift", axis 1 = "boom", axis 2 = unrelated.
_VECS = {
    "scissor lift": [1.0, 0.0, 0.0],
    "boom": [0.0, 1.0, 0.0],
    "scissor lift rental": [1.0, 0.0, 0.0],
    "boom truck": [0.0, 1.0, 0.0],
    "cotton fabric": [0.0, 0.0, 1.0],
    "opposite": [-1.0, 0.0, 0.0],
}
POOLS = {"both_match": 10, "both_no_match": 0, "laya_false_negative": 0, "laya_false_positive": 0}


@pytest.fixture
def fake_runtime(monkeypatch):
    calls = {"init": [], "encode": []}

    class FakeST:
        def __init__(self, model_id, **kw):
            calls["init"].append((model_id, kw))

        def encode(self, texts, **kw):
            calls["encode"].append((list(texts), kw))
            return np.array([_VECS[t] for t in texts])

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    return calls


def _row(i):
    return SampleRow(sample_index=i, source_file="a.xlsx", row_id=i, quadrant="both_match",
                     product_details_text=["scissor lift rental", "opposite"][i % 2],
                     laya_score=0.0, laya_match=False, llm_match=True, llm_rationale="")


ROWS = [_row(i) for i in range(10)]
LABELS = {i: (i % 2 == 0, "") for i in range(10)}


def test_registered_and_flagged_gate_ineligible_with_pinned_revision():
    assert NAME in CANDIDATES
    (s,) = build_scorers([NAME], ["scissor lift"])
    assert s.gate_eligible is False
    assert s.model_id == "Alibaba-NLP/gte-large-en-v1.5"
    assert len(s.revision) == 40
    assert "{keywords}" in s.template


def test_score_is_max_cosine_over_keywords_rescaled_to_unit_interval(fake_runtime):
    (s,) = build_scorers([NAME], ["scissor lift", "boom"])
    scores = s.score(["scissor lift rental", "boom truck", "cotton fabric", "opposite"])

    assert scores == pytest.approx([1.0, 1.0, 0.5, 0.5])
    assert all(0.0 <= x <= 1.0 for x in scores)


def test_runtime_loads_lazily_once_with_pinned_revisions(fake_runtime):
    (s,) = build_scorers([NAME], ["scissor lift"])
    assert fake_runtime["init"] == []  # nothing loaded at construction

    s.score(["cotton fabric"])
    s.score(["boom truck"])

    assert len(fake_runtime["init"]) == 1
    model_id, kw = fake_runtime["init"][0]
    assert model_id == s.model_id
    assert kw["revision"] == s.revision
    assert kw["trust_remote_code"] is True
    # the model's remote modeling code lives in another repo and must be pinned too
    assert len(kw["config_kwargs"]["code_revision"]) == 40
    assert kw["model_kwargs"]["code_revision"] == kw["config_kwargs"]["code_revision"]


def test_keywords_are_embedded_from_the_frozen_template(fake_runtime):
    (s,) = build_scorers([NAME], ["scissor lift", "boom"])
    s.score(["cotton fabric"])

    assert fake_runtime["encode"][0][0] == ["scissor lift", "boom"]


def test_gate_ineligible_candidate_reports_auroc_only_and_never_passes(fake_runtime):
    (s,) = build_scorers([NAME], ["scissor lift"])

    result = evaluate_candidate(s, ROWS, LABELS, POOLS)

    # Perfect separation and perfect metrics at the cut: still no verdict.
    assert result.auroc == 1.0
    assert result.population_weighted.accuracy == 1.0
    assert result.verdict == "AUROC-only"

    report = build_report([result], ["scissor lift"], 10, POOLS)
    (row,) = [ln for ln in render_markdown(report).splitlines() if ln.startswith(f"| {NAME} ")]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1] == "AUROC-only"
    assert cells[-1] == "100.0%"  # AUROC shown
    assert set(cells[2:-1]) == {"n/a"}  # no threshold metrics for a reference


def test_verdict_is_never_go_unless_gate_eligible_is_literally_true():
    for flag in (False, None, 0, "no"):
        assert determine_verdict(1.0, 1.0, flag) == "AUROC-only"
    assert determine_verdict(1.0, 1.0, True) == "GO"


def test_build_report_refuses_a_passing_verdict_on_a_gate_ineligible_candidate(fake_runtime):
    (s,) = build_scorers([NAME], ["scissor lift"])
    result = evaluate_candidate(s, ROWS, LABELS, POOLS)
    result.verdict = "GO"  # corrupted result

    with pytest.raises(ValueError, match="gate-ineligible"):
        build_report([result], ["scissor lift"], 10, POOLS)
