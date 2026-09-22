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


# --- question_mode / payload_layout toggles (issue #7) ---


def test_build_relevance_questions_default_is_byte_identical_to_combined_legacy():
    rows = [(0, "widget A"), (1, "widget B")]
    keywords = ["widget", "gadget"]

    default = build_relevance_questions(rows, keywords)
    explicit = build_relevance_questions(
        rows, keywords, question_mode="combined", payload_layout="legacy"
    )

    assert default == explicit


def test_build_relevance_questions_per_keyword_legacy_one_question_per_row_keyword_pair():
    questions = build_relevance_questions(
        [(0, "widget A"), (1, "widget B")],
        ["widget", "gadget"],
        question_mode="per_keyword",
        payload_layout="legacy",
    )

    assert len(questions) == 4
    q = questions["0::widget"]
    assert q["type"] == "noul"
    assert "widget A" in q["instructions"]
    assert "widget" in q["instructions"]
    assert "gadget" not in q["instructions"]
    assert "widget B" not in q["instructions"]


def test_build_relevance_questions_convention_combined_uses_fixed_instructions_no_row_text():
    questions = build_relevance_questions(
        [(0, "widget A"), (1, "widget B")],
        ["widget", "gadget"],
        question_mode="combined",
        payload_layout="convention",
    )

    assert set(questions) == {"0", "1"}
    assert questions["0"]["instructions"] == questions["1"]["instructions"]
    assert "widget A" not in questions["0"]["instructions"]
    assert "widget" in questions["0"]["instructions"]
    assert "gadget" in questions["0"]["instructions"]


def test_build_relevance_questions_convention_per_keyword_names_only_one_keyword():
    questions = build_relevance_questions(
        [(0, "widget A")],
        ["widget", "gadget"],
        question_mode="per_keyword",
        payload_layout="convention",
    )

    assert set(questions) == {"0::widget", "0::gadget"}
    assert "widget A" not in questions["0::widget"]["instructions"]
    assert "widget" in questions["0::widget"]["instructions"]
    assert "gadget" not in questions["0::widget"]["instructions"]
    assert "gadget" in questions["0::gadget"]["instructions"]
    assert "widget" not in questions["0::gadget"]["instructions"]


def test_build_relevance_questions_convention_dedupes_truncation_artifact():
    text = "BRAND APJ MODEL AP36J UPPER STRUCTURE ONLYBRAND APJ MODEL AP36J UPPER STRUCTURE ONLY"
    # Convention layout doesn't put the row text in `instructions`, so this only
    # matters once the text lands in `state` -- exercised via judge_batch below.
    questions = build_relevance_questions([(0, text)], ["x"], payload_layout="convention")
    assert questions["0"]["instructions"]  # fixed rubric, unaffected either way


def test_judge_batch_legacy_layout_still_one_shared_call_regardless_of_question_mode():
    agent = StubAgent(
        responder=lambda state, questions: {"answers": {qid: _noul_answer(0.9) for qid in questions}}
    )
    client = LayaClient(agent=agent)

    client.judge_batch(
        [(0, "a"), (1, "b")], ["k1", "k2"], question_mode="per_keyword", payload_layout="legacy"
    )

    assert len(agent.calls) == 1
    state, questions = agent.calls[0]
    assert state == {"keywords": ["k1", "k2"]}
    assert set(questions) == {"0::k1", "0::k2", "1::k1", "1::k2"}


def test_judge_batch_convention_layout_calls_once_per_row_with_row_text_as_state():
    agent = StubAgent(
        responder=lambda state, questions: {"answers": {qid: _noul_answer(0.9) for qid in questions}}
    )
    client = LayaClient(agent=agent)

    client.judge_batch(
        [(0, "widget A"), (1, "widget B")],
        ["widget"],
        question_mode="combined",
        payload_layout="convention",
    )

    assert len(agent.calls) == 2
    states = {state for state, _questions in agent.calls}
    assert states == {"widget A", "widget B"}


def test_judge_batch_per_keyword_convention_batches_one_call_per_row_with_max_score_aggregation():
    def responder(state, questions):
        if state == "widget A":
            return {"answers": {"0::widget": _noul_answer(0.9), "0::gadget": _noul_answer(0.1)}}
        return {"answers": {"1::widget": _noul_answer(0.2), "1::gadget": _noul_answer(0.3)}}

    agent = StubAgent(responder=responder)
    client = LayaClient(agent=agent)

    verdicts = client.judge_batch(
        [(0, "widget A"), (1, "widget B")],
        ["widget", "gadget"],
        threshold=0.5,
        question_mode="per_keyword",
        payload_layout="convention",
    )

    assert len(agent.calls) == 2  # one predict() call per row, not per (row, keyword)
    for state, questions in agent.calls:
        assert isinstance(state, str)  # the row's own PRODUCT DETAILS text, not the keyword list
        assert set(questions) == {
            f"{0 if state == 'widget A' else 1}::widget",
            f"{0 if state == 'widget A' else 1}::gadget",
        }

    by_id = {v.row_id: v for v in verdicts}
    assert by_id[0].score == 0.9 and by_id[0].match is True and by_id[0].or_threshold_match is True
    assert by_id[1].score == 0.3 and by_id[1].match is False and by_id[1].or_threshold_match is False


def test_judge_batch_convention_layout_dedupes_truncation_artifact_in_state():
    text = "BRAND APJ MODEL AP36J UPPER STRUCTURE ONLYBRAND APJ MODEL AP36J UPPER STRUCTURE ONLY"
    agent = StubAgent(responder=lambda state, questions: {"answers": {"0": _noul_answer(0.9)}})
    client = LayaClient(agent=agent)

    client.judge_batch([(0, text)], ["x"], payload_layout="convention")

    (state, _questions), = agent.calls
    assert state == "BRAND APJ MODEL AP36J UPPER STRUCTURE ONLY"


def test_judge_batch_per_keyword_aggregates_via_max_score_then_threshold():
    def responder(state, questions):
        return {
            "answers": {
                "0::widget": _noul_answer(0.9),
                "0::gadget": _noul_answer(0.3),
            }
        }

    agent = StubAgent(responder=responder)
    client = LayaClient(agent=agent)

    verdicts = client.judge_batch(
        [(0, "a")], ["widget", "gadget"], threshold=0.5, question_mode="per_keyword"
    )

    assert len(verdicts) == 1
    verdict = verdicts[0]
    assert verdict.score == 0.9
    assert verdict.match is True
    assert verdict.confidence == 0.9


def test_judge_batch_per_keyword_exposes_or_of_thresholds_as_secondary_figure():
    def responder(state, questions):
        return {
            "answers": {
                "0::widget": _noul_answer(0.6),
                "0::gadget": _noul_answer(0.4),
            }
        }

    agent = StubAgent(responder=responder)
    client = LayaClient(agent=agent)

    verdicts = client.judge_batch(
        [(0, "a")], ["widget", "gadget"], threshold=0.5, question_mode="per_keyword"
    )

    verdict = verdicts[0]
    assert verdict.match is True  # primary: max score 0.6 >= threshold
    assert verdict.or_threshold_match is True  # secondary: at least one binary cleared 0.5


def test_judge_batch_combined_mode_or_threshold_match_is_none():
    agent = StubAgent(responder=lambda state, questions: {"answers": {"0": _noul_answer(0.9)}})
    client = LayaClient(agent=agent)

    verdicts = client.judge_batch([(0, "a")], ["k"])

    assert verdicts[0].or_threshold_match is None


def test_judge_batch_per_keyword_missing_answer_treated_as_non_match_zero_score():
    agent = StubAgent(responder=lambda state, questions: {"answers": {"0::widget": _noul_answer(0.9)}})
    client = LayaClient(agent=agent)

    verdicts = client.judge_batch([(0, "a")], ["widget", "gadget"], question_mode="per_keyword")

    assert verdicts[0].score == 0.9
    assert verdicts[0].match is True


def test_judge_batch_convention_layout_isolates_failure_to_the_failing_row():
    def responder(state, questions):
        if state == "boom":
            raise RuntimeError("checkpoint hiccup")
        return {"answers": {qid: _noul_answer(0.9) for qid in questions}}

    agent = StubAgent(responder=responder)
    client = LayaClient(agent=agent)

    verdicts = client.judge_batch(
        [(0, "boom"), (1, "fine")], ["k"], payload_layout="convention"
    )

    by_id = {v.row_id: v for v in verdicts}
    assert by_id[0].match is False and by_id[0].score == 0.0
    assert by_id[1].match is True and by_id[1].score == 0.9
