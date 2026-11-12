"""gte-reranker-modernbert-base candidate (issue #18): driven through the Scorer
seam with a fake runtime injected -- no torch, no model download, no network."""

from __future__ import annotations

import contextlib
import math
import re

from scripts.bakeoff_scorers import CANDIDATES, build_scorers
from scripts.build_ground_truth_report import SampleRow
from scripts.run_system1_bakeoff import THRESHOLD, evaluate_candidate

NAME = "gte-reranker-modernbert-base"


class _Tensor:
    def __init__(self, vals):
        self.vals = vals

    def view(self, _):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.vals


class _FakeTorch:
    def no_grad(self):
        return contextlib.nullcontext()

    def sigmoid(self, t):
        return _Tensor([1 / (1 + math.exp(-v)) for v in t.vals])


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTok:
    def __init__(self):
        self.calls = []

    def __call__(self, queries, docs, **kw):
        self.calls.append((list(queries), list(docs), kw))
        return _FakeBatch(docs=list(docs))


class _FakeModel:
    """Logit = +4 if the document mentions 'lift' else -4."""

    def __call__(self, docs):
        class _Out:
            logits = _Tensor([4.0 if "lift" in d else -4.0 for d in docs])

        return _Out()


def _scorer(batch_size=2):
    s = build_scorers([NAME], ["boom lift", "scissor lift"])[0]
    s._batch_size = batch_size
    s._loaded = (_FakeTorch(), _FakeTok(), _FakeModel(), "cpu")
    return s


def test_registered_with_pinned_revision_and_frozen_template():
    assert NAME in CANDIDATES
    s = build_scorers([NAME], ["boom lift"])[0]
    assert s.model_id == "Alibaba-NLP/gte-reranker-modernbert-base"
    assert re.fullmatch(r"[0-9a-f]{40}", s.revision)
    assert "{keywords}" in s.template
    assert s.gate_eligible is True
    assert s._loaded is None  # construction never loads the runtime


def test_one_pair_per_row_sigmoid_in_order_across_batches():
    s = _scorer(batch_size=2)
    texts = ["a lift", "a pump", "scissor lift", "a crane", "b"]
    out = s.score(texts)

    assert len(out) == 5
    assert out[0] == out[2] and out[0] > 0.98
    assert out[1] == out[3] == out[4] and out[1] < 0.02
    tok = s._loaded[1]
    assert [len(q) for q, _, _ in tok.calls] == [2, 2, 1]
    assert all("boom lift; scissor lift" in q for qs, _, _ in tok.calls for q in qs)
    assert [d for _, ds, _ in tok.calls for d in ds] == texts


def test_evaluates_through_the_driver_at_half():
    rows = [
        SampleRow(i, "a.xlsx", i, "both_match", t, 0.0, False, True, "")
        for i, t in enumerate(["x lift", "y pump"])
    ]
    labels = {0: (True, ""), 1: (False, "")}
    pools = {"both_match": 2, "both_no_match": 0, "laya_false_negative": 0, "laya_false_positive": 0}
    res = evaluate_candidate(_scorer(), rows, labels, pools)
    assert THRESHOLD == 0.5
    assert res.raw.accuracy == 1.0
