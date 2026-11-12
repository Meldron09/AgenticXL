"""DeBERTa zero-shot NLI candidate (issue #17). The model call is faked at the
adapter's one runtime-touching method, so the one-hypothesis-per-keyword and
max-aggregation logic is tested through the public Scorer interface."""

from __future__ import annotations

from scripts.bakeoff_scorers import CANDIDATES, DebertaNliScorer, build_scorers

NAME = "deberta-v3-large-zeroshot-v2.0"


class FakeNli(DebertaNliScorer):
    """Entailment prob looked up by (premise, hypothesis); no model."""

    def __init__(self, keywords):
        super().__init__(keywords)
        self.table: dict[tuple[str, str], float] = {}
        self.pairs: list[tuple[str, str]] = []

    def _entail(self, pairs):
        self.pairs.extend(pairs)
        return [self.table.get(p, 0.0) for p in pairs]


def _hyp(scorer, kw):
    return scorer.template.format(keywords=kw)


def test_row_score_is_max_entailment_across_keyword_hypotheses():
    s = FakeNli(["scissor lift", "boom lift", "forklift"])
    s.table = {
        ("a", _hyp(s, "scissor lift")): 0.2,
        ("a", _hyp(s, "boom lift")): 0.9,
        ("a", _hyp(s, "forklift")): 0.4,
        ("b", _hyp(s, "scissor lift")): 0.1,
        ("b", _hyp(s, "boom lift")): 0.05,
        ("b", _hyp(s, "forklift")): 0.3,
    }
    assert s.score(["a", "b"]) == [0.9, 0.3]


def test_one_hypothesis_per_keyword_per_row():
    s = FakeNli(["x", "y"])
    s.score(["r1", "r2"])
    assert sorted(s.pairs) == sorted((t, _hyp(s, k)) for t in ("r1", "r2") for k in ("x", "y"))


def test_registered_with_pinned_revision_and_frozen_template():
    assert NAME in CANDIDATES
    (s,) = build_scorers([NAME], ["scissor lift"])
    assert s.model_id == "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
    assert len(s.revision) == 40
    assert "{keywords}" in s.template
    assert s.gate_eligible is True
