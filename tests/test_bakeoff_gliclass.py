"""GLiClass large v3.0 candidate (issue #16), tested through the Scorer interface
with a fake `gliclass`/`transformers`/`torch` runtime -- no model, no network."""

from __future__ import annotations

import math
import sys
import types

import pytest

from scripts.bakeoff_scorers import CANDIDATES, build_scorers


class _FakePipeline:
    calls: list = []
    loads: list = []

    def __init__(self, model, tokenizer, classification_type, device):
        self.classification_type = classification_type

    def __call__(self, texts, labels, threshold, batch_size):
        _FakePipeline.calls.append((list(texts), list(labels), threshold))
        # label scores keyed off the text so the test controls the max
        return [[{"label": l, "score": 0.9 if l in t else 0.1} for l in labels] for t in texts]


@pytest.fixture
def fake_runtime(monkeypatch):
    _FakePipeline.calls, _FakePipeline.loads = [], []

    def from_pretrained(model_id, **kw):
        _FakePipeline.loads.append((model_id, kw))
        return object()

    gliclass = types.ModuleType("gliclass")
    gliclass.GLiClassModel = types.SimpleNamespace(from_pretrained=from_pretrained)
    gliclass.ZeroShotClassificationPipeline = _FakePipeline
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=from_pretrained)
    torch = types.ModuleType("torch")
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    for name, mod in [("gliclass", gliclass), ("transformers", transformers), ("torch", torch)]:
        monkeypatch.setitem(sys.modules, name, mod)


def test_registered_with_pinned_identity():
    scorer = build_scorers(["gliclass-large-v3.0"], ["scissor lift"])[0]

    assert CANDIDATES["gliclass-large-v3.0"] is type(scorer)
    assert scorer.model_id == "knowledgator/gliclass-large-v3.0"
    assert len(scorer.revision) == 40
    assert scorer.gate_eligible is True
    assert scorer.template  # frozen, reported


def test_probability_is_max_over_keywords_in_one_pass_per_batch(fake_runtime):
    scorer = build_scorers(["gliclass-large-v3.0"], ["scissor lift", "boom lift"])[0]

    probs = scorer.score(["a boom lift unit", "bananas"])

    assert probs == [0.9, 0.1]
    assert len(_FakePipeline.calls) == 1  # all rows, all keywords: one call
    texts, labels, threshold = _FakePipeline.calls[0]
    assert labels == ["scissor lift", "boom lift"]  # all keywords scored together
    assert threshold == 0.0  # every label's score comes back; the 0.5 cut is the driver's


def test_checkpoint_loaded_at_pinned_revision_and_only_once(fake_runtime):
    scorer = build_scorers(["gliclass-large-v3.0"], ["lift"])[0]

    scorer.score(["x"])
    scorer.score(["y"])

    assert [kw["revision"] for _, kw in _FakePipeline.loads] == [scorer.revision] * 2  # model + tokenizer


def test_row_with_no_returned_labels_is_a_bad_score_not_zero(fake_runtime, monkeypatch):
    monkeypatch.setattr(
        _FakePipeline, "__call__",
        lambda self, texts, labels, threshold, batch_size: [[], [{"label": "lift", "score": 0.9}]],
    )
    scorer = build_scorers(["gliclass-large-v3.0"], ["lift"])[0]

    probs = scorer.score(["a", "b"])

    assert math.isnan(probs[0]) and probs[1] == 0.9
