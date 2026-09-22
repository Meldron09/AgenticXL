from __future__ import annotations

from agenticxl.filtering.laya_client import (
    LAYA_MODEL_ID,
    LAYA_MODEL_REVISION,
    LayaClient,
    LayaVerdict,
    build_relevance_questions,
)
from helpers import FakeLayaClient


class StubAgent:
    """Minimal stand-in for a loaded laya_mlx.Agent: records the (state, questions) it
    was called with and returns a canned `predict()` result, so LayaClient's own
    question-building/response-parsing logic can be tested without mlx or a real
    checkpoint."""

    def __init__(self, responder=None):
        self.calls: list[tuple[dict, dict]] = []
        self._responder = responder

    def predict(self, state, questions):
        self.calls.append((state, questions))
        if self._responder is None:
            return {"answers": {}}
        return self._responder(state, questions)


def _noul_answer(score: float, confidence: float | None = None) -> dict:
    return {"type": "noul", "noul": score, "confidence": confidence if confidence is not None else max(score, 1 - score)}


def test_build_relevance_questions_one_noul_question_per_row():
    questions = build_relevance_questions([(0, "widget A"), (1, "widget B")], ["widget"])

    assert set(questions) == {"0", "1"}
    for q in questions.values():
        assert q["type"] == "noul"
        assert "true" in q["criteria"] and "false" in q["criteria"]
    assert "widget A" in questions["0"]["instructions"]
    assert "widget B" in questions["1"]["instructions"]


def test_build_relevance_questions_dedupes_truncation_artifact_in_text():
    text = "BRAND APJ MODEL AP36J UPPER STRUCTURE ONLYBRAND APJ MODEL AP36J UPPER STRUCTURE ONLY"
    questions = build_relevance_questions([(0, text)], ["x"])

    assert text not in questions["0"]["instructions"]
    assert "BRAND APJ MODEL AP36J UPPER STRUCTURE ONLY" in questions["0"]["instructions"]


def test_judge_batch_uses_one_shared_state_for_native_batching():
    agent = StubAgent(responder=lambda state, questions: {
        "answers": {qid: _noul_answer(0.9) for qid in questions}
    })
    client = LayaClient(agent=agent)

    client.judge_batch([(0, "a"), (1, "b"), (2, "c")], ["keyword"])

    assert len(agent.calls) == 1  # one native-batched predict() call, not one per row
    state, questions = agent.calls[0]
    assert state == {"keywords": ["keyword"]}
    assert set(questions) == {"0", "1", "2"}


def test_judge_batch_applies_threshold_to_noul_score():
    agent = StubAgent(responder=lambda state, questions: {
        "answers": {
            "0": _noul_answer(0.9),
            "1": _noul_answer(0.2),
        }
    })
    client = LayaClient(agent=agent)

    verdicts = client.judge_batch([(0, "a"), (1, "b")], ["k"], threshold=0.5)

    by_id = {v.row_id: v for v in verdicts}
    assert by_id[0].match is True and by_id[0].score == 0.9
    assert by_id[1].match is False and by_id[1].score == 0.2


def test_judge_batch_row_missing_from_response_is_non_match_not_guessed():
    agent = StubAgent(responder=lambda state, questions: {"answers": {"0": _noul_answer(0.9)}})
    client = LayaClient(agent=agent)

    verdicts = client.judge_batch([(0, "a"), (1, "b")], ["k"])

    by_id = {v.row_id: v for v in verdicts}
    assert by_id[1].match is False
    assert by_id[1].score == 0.0
    assert by_id[1].confidence == 0.0


def test_judge_batch_predict_failure_isolated_to_all_non_match():
    def boom(state, questions):
        raise RuntimeError("checkpoint hiccup")

    agent = StubAgent(responder=boom)
    client = LayaClient(agent=agent)

    verdicts = client.judge_batch([(0, "a"), (1, "b")], ["k"])

    assert all(v.match is False and v.score == 0.0 for v in verdicts)


def test_judge_batch_empty_rows_returns_empty_list_without_calling_agent():
    agent = StubAgent()
    client = LayaClient(agent=agent)

    assert client.judge_batch([], ["k"]) == []
    assert agent.calls == []


def test_model_identifier_combines_id_and_revision():
    client = LayaClient(agent=StubAgent(), model_id="org/repo", revision="deadbeef")
    assert client.model_identifier == "org/repo@deadbeef"


def test_default_model_id_and_revision_are_pinned():
    assert LAYA_MODEL_ID == "aac6fef/laya-typed-decisions-mlx"
    assert LAYA_MODEL_REVISION == "f9e501c2080cc57c13d6887820329758f5351125"


def test_verdicts_carry_the_model_identifier():
    agent = StubAgent(responder=lambda state, questions: {"answers": {"0": _noul_answer(0.8)}})
    client = LayaClient(agent=agent, model_id="org/repo", revision="rev1")

    verdicts = client.judge_batch([(0, "a")], ["k"])

    assert verdicts == [
        LayaVerdict(row_id=0, match=True, score=0.8, confidence=0.8, model="org/repo@rev1")
    ]


def test_fake_laya_client_matches_real_interface():
    def responder(rows, keywords, threshold):
        return [
            LayaVerdict(row_id=row_id, match=True, score=1.0, confidence=1.0, model="fake")
            for row_id, _text in rows
        ]

    fake = FakeLayaClient(responder=responder)
    verdicts = fake.judge_batch([(0, "a"), (1, "b")], ["k"], threshold=0.5)

    assert [v.row_id for v in verdicts] == [0, 1]
    assert all(v.match for v in verdicts)


def test_fake_laya_client_default_responder_returns_no_matches():
    fake = FakeLayaClient()
    verdicts = fake.judge_batch([(0, "a")], ["k"])
    assert verdicts == [LayaVerdict(row_id=0, match=False, score=0.0, confidence=0.0, model="fake")]
