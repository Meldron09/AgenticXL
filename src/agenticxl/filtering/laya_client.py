"""Local laya-mlx classifier client for Stage 1 relevance judging.

Mirrors the `LLMClient` dependency-injection pattern used elsewhere in this
codebase (see `models/base.py`): production code constructs a real
`LayaClient`, tests substitute `FakeLayaClient` (`tests/helpers.py`), so
relevance judging stays unit-testable without a live model or network call.
`LayaClient` itself additionally accepts an injected `agent` object (any
object exposing `.predict(state, questions)`), which is how this module's
own tests exercise its question-building/response-parsing logic without
mlx or a real checkpoint -- the real `laya_mlx.Agent` is only constructed
when `agent` is omitted.

Pinned to `laya-mlx==0.2.0` and the exact Hugging Face checkpoint revision
for `aac6fef/laya-typed-decisions-mlx` -- CLAUDE.md/issue #1's "Further
Notes" documents this as a consciously accepted supply-chain risk (a very
new package/checkpoint, reviewed once for this exact pin). Any future
version or checkpoint bump requires the same re-verification, not an
automatic upgrade.

Question construction (issue #6/#7): two independent, keyword-only toggles
select among four ways to build the `noul` question(s) `laya_mlx.Agent.predict`
is asked -- both default to today's original behavior, so existing callers
(`validate_laya_filtering.py`, this module's original tests) are unaffected.

- `question_mode="combined"` (default): one `noul` question per row, phrased
  as a single compound "relevant to any keyword?" judgment.
- `question_mode="per_keyword"`: one atomic `noul` question per `(row_id,
  keyword)` pair. The per-row verdict aggregates via the max raw `noul`
  score across that row's keyword questions, thresholded -- this is the
  primary verdict. The OR-of-thresholded-binaries alternative is also
  computed and exposed as `LayaVerdict.or_threshold_match`, a secondary
  sanity-check figure, never the primary `match`.
- `payload_layout="legacy"` (default): the row's PRODUCT DETAILS text is
  embedded in the question's `instructions`; the shared `state` carries
  only the keyword list. This is the inverted-from-convention behavior the
  original implementation shipped with, preserved unchanged as the default.
- `payload_layout="convention"`: matches laya-mlx's own documented
  `state`/`instructions` split (large variable payload in `state`, small
  fixed rubric in `instructions`) and this pinned checkpoint's own
  `rl_agent_config.json` token budgets (`head_max_len=256` for
  `instructions`, larger `max_len` budget for `state`). Because
  `Agent.predict(state, questions)` takes exactly one shared `state` for
  every question in the call, and each row's PRODUCT DETAILS text is now
  the `state`, `judge_batch` issues one `predict()` call *per row* under
  this layout (batching only across that row's keyword questions, not
  across rows) -- unlike `payload_layout="legacy"`, which still native-
  batches every row in `rows` into a single shared-`state` call. A failed
  `predict()` call is therefore isolated to that one row under
  `"convention"`, versus the whole batch under `"legacy"`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agenticxl.filtering.relevance import dedupe_truncation_artifact

LAYA_MODEL_ID = "aac6fef/laya-typed-decisions-mlx"
LAYA_MODEL_REVISION = "f9e501c2080cc57c13d6887820329758f5351125"
DEFAULT_MATCH_THRESHOLD = 0.5

QuestionMode = Literal["combined", "per_keyword"]
PayloadLayout = Literal["legacy", "convention"]

_COMBINED_CRITERIA = {
    "true": "relevant to at least one keyword in `keywords`",
    "false": "not relevant to any keyword in `keywords`",
}


@dataclass(frozen=True)
class LayaVerdict:
    row_id: int
    match: bool
    score: float
    confidence: float
    model: str
    # Only populated for question_mode="per_keyword": the OR-of-thresholded-binaries
    # alternative aggregation, reported alongside the primary max-score verdict as a
    # secondary sanity-check figure -- never itself used as `match`.
    or_threshold_match: bool | None = None


def _per_keyword_key(row_id: int, keyword: str) -> str:
    return f"{row_id}::{keyword}"


def _per_keyword_criteria(keyword: str) -> dict:
    return {
        "true": f'relevant to the keyword "{keyword}"',
        "false": f'not relevant to the keyword "{keyword}"',
    }


def _relevance_clause(location: str, target: str) -> str:
    """The shared question phrasing both layouts and both modes build on, varying
    only where the PRODUCT DETAILS text lives (`location`) and what it's being
    judged against (`target`)."""
    return (
        f"Does the PRODUCT DETAILS text {location} describe, or a clear synonym, "
        f"sub-component, variant, or accessory of, {target}"
    )


def build_relevance_questions(
    rows: list[tuple[int, str]],
    keywords: list[str],
    *,
    question_mode: QuestionMode = "combined",
    payload_layout: PayloadLayout = "legacy",
) -> dict:
    """Build the `noul` question dict for `rows`.

    Under `payload_layout="legacy"` (default), each row's PRODUCT DETAILS text is
    embedded directly in the returned question(s)' `instructions`, and `keywords`
    itself lives in the shared `state` built by `LayaClient.judge_batch`, not here.

    Under `payload_layout="convention"`, `instructions` is a small fixed rubric with
    no row-specific text at all -- the row's PRODUCT DETAILS text is meant to be the
    `state` for that row's own `predict()` call instead (see `LayaClient.judge_batch`
    and the module docstring for why that means one `predict()` call per row).
    """
    questions: dict[str, dict] = {}
    for row_id, text in rows:
        if question_mode == "combined":
            if payload_layout == "legacy":
                clean_text = dedupe_truncation_artifact(text)
                instructions = (
                    _relevance_clause("below", "any keyword in `keywords`")
                    + f'? PRODUCT DETAILS: "{clean_text}"'
                )
            else:
                instructions = (
                    _relevance_clause("in `state`", f"any keyword in this list: {', '.join(keywords)}")
                    + "?"
                )
            questions[str(row_id)] = {
                "type": "noul",
                "instructions": instructions,
                "criteria": dict(_COMBINED_CRITERIA),
            }
        else:
            clean_text = dedupe_truncation_artifact(text) if payload_layout == "legacy" else None
            for keyword in keywords:
                target = f'the keyword "{keyword}"'
                if payload_layout == "legacy":
                    instructions = (
                        _relevance_clause("below", target) + f'? PRODUCT DETAILS: "{clean_text}"'
                    )
                else:
                    instructions = _relevance_clause("in `state`", target) + "?"
                questions[_per_keyword_key(row_id, keyword)] = {
                    "type": "noul",
                    "instructions": instructions,
                    "criteria": _per_keyword_criteria(keyword),
                }
    return questions


def _verdict_from_answers(
    row_id: int,
    keywords: list[str],
    answers: dict,
    threshold: float,
    question_mode: QuestionMode,
    model_identifier: str,
) -> LayaVerdict:
    """Build one row's verdict from an `answers` dict, never fabricating a value for
    a missing/malformed entry -- a missing question key is a non-match, not a guess."""
    if question_mode == "combined":
        answer = answers.get(str(row_id))
        if not answer or "noul" not in answer:
            return LayaVerdict(row_id=row_id, match=False, score=0.0, confidence=0.0, model=model_identifier)
        score = float(answer["noul"])
        confidence = float(answer.get("confidence", 0.0))
        return LayaVerdict(
            row_id=row_id, match=score >= threshold, score=score, confidence=confidence, model=model_identifier
        )

    best_score, best_confidence, any_thresholded = 0.0, 0.0, False
    for keyword in keywords:
        answer = answers.get(_per_keyword_key(row_id, keyword))
        if not answer or "noul" not in answer:
            score, confidence = 0.0, 0.0
        else:
            score = float(answer["noul"])
            confidence = float(answer.get("confidence", 0.0))
        if score >= threshold:
            any_thresholded = True
        if score > best_score:
            best_score, best_confidence = score, confidence
    return LayaVerdict(
        row_id=row_id,
        match=best_score >= threshold,
        score=best_score,
        confidence=best_confidence,
        model=model_identifier,
        or_threshold_match=any_thresholded,
    )


class LayaClient:
    """Wraps a loaded laya-mlx `Agent` for batched relevance classification."""

    def __init__(
        self,
        model_id: str = LAYA_MODEL_ID,
        revision: str = LAYA_MODEL_REVISION,
        batch_size: int = 50,
        agent=None,
    ):
        self.model_id = model_id
        self.revision = revision
        if agent is not None:
            self._agent = agent
        else:
            from laya_mlx import Agent

            self._agent = Agent(model_id, revision=revision, batch_size=batch_size)

    @property
    def model_identifier(self) -> str:
        return f"{self.model_id}@{self.revision}"

    def _predict(self, state, questions) -> dict:
        """One `predict()` call, isolated: a broken batch (bad checkpoint state,
        non-finite outputs, etc.) becomes empty answers -- never a raised exception
        that would abort the rest of the run -- exactly like judge_batch's
        StructuredOutputError handling in filtering/relevance.py."""
        try:
            result = self._agent.predict(state, questions)
            return result.get("answers", {})
        except Exception:
            return {}

    def judge_batch(
        self,
        rows: list[tuple[int, str]],
        keywords: list[str],
        threshold: float = DEFAULT_MATCH_THRESHOLD,
        *,
        question_mode: QuestionMode = "combined",
        payload_layout: PayloadLayout = "legacy",
    ) -> list[LayaVerdict]:
        """Judge every (row_id, product_details_text) pair in `rows`.

        Under `payload_layout="legacy"`, this is one native-batched `predict()` call
        for the whole of `rows` (a failed call is a non-match for every row in it).
        Under `payload_layout="convention"`, this is one `predict()` call per row,
        since each row's own text becomes that call's `state` (a failed call is a
        non-match only for that one row) -- see the module docstring for why.
        """
        if not rows:
            return []

        if payload_layout == "legacy":
            return self._judge_batch_legacy(rows, keywords, threshold, question_mode)
        return self._judge_batch_convention(rows, keywords, threshold, question_mode)

    def _judge_batch_legacy(self, rows, keywords, threshold, question_mode) -> list[LayaVerdict]:
        questions = build_relevance_questions(
            rows, keywords, question_mode=question_mode, payload_layout="legacy"
        )
        state = {"keywords": list(keywords)}
        answers = self._predict(state, questions)
        return [
            _verdict_from_answers(row_id, keywords, answers, threshold, question_mode, self.model_identifier)
            for row_id, _text in rows
        ]

    def _judge_batch_convention(self, rows, keywords, threshold, question_mode) -> list[LayaVerdict]:
        verdicts = []
        for row_id, text in rows:
            questions = build_relevance_questions(
                [(row_id, text)], keywords, question_mode=question_mode, payload_layout="convention"
            )
            state = dedupe_truncation_artifact(text)
            answers = self._predict(state, questions)
            verdicts.append(
                _verdict_from_answers(row_id, keywords, answers, threshold, question_mode, self.model_identifier)
            )
        return verdicts
