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

Question shape: `laya_mlx.Agent.predict(state, questions)` batches natively
over `questions` for one shared `state` (its `batch_size` chunks the
per-question items, not per-state calls) -- so to get real batched
inference across many rows in a single call, each row becomes its own
`noul` question (keyed by row_id, carrying that row's PRODUCT DETAILS text
in its instructions), all judged against one shared `state` holding the
keyword list, rather than one `state` per row.
"""

from __future__ import annotations

from dataclasses import dataclass

from agenticxl.filtering.relevance import dedupe_truncation_artifact

LAYA_MODEL_ID = "aac6fef/laya-typed-decisions-mlx"
LAYA_MODEL_REVISION = "f9e501c2080cc57c13d6887820329758f5351125"
DEFAULT_MATCH_THRESHOLD = 0.5


@dataclass(frozen=True)
class LayaVerdict:
    row_id: int
    match: bool
    score: float
    confidence: float
    model: str


def build_relevance_questions(rows: list[tuple[int, str]], keywords: list[str]) -> dict:
    """One `noul` question per row, each asking overall relevance against the full
    keyword list in a single judgment (not one question per keyword). `keywords`
    itself lives in the shared `state` (see `LayaClient.judge_batch`), not here.
    """
    del keywords  # embedded in the shared state, not per-question instructions
    questions: dict[str, dict] = {}
    for row_id, text in rows:
        clean_text = dedupe_truncation_artifact(text)
        questions[str(row_id)] = {
            "type": "noul",
            "instructions": (
                "Does the PRODUCT DETAILS text below describe, or a clear synonym, "
                "sub-component, variant, or accessory of, any keyword in `keywords`? "
                f'PRODUCT DETAILS: "{clean_text}"'
            ),
            "criteria": {
                "true": "relevant to at least one keyword in `keywords`",
                "false": "not relevant to any keyword in `keywords`",
            },
        }
    return questions


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

    def judge_batch(
        self,
        rows: list[tuple[int, str]],
        keywords: list[str],
        threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> list[LayaVerdict]:
        """Judge one batch of (row_id, product_details_text) pairs in a single
        native-batched laya-mlx `predict()` call. A row missing from a
        malformed/partial response, or a batch that fails outright, becomes a
        non-match (score/confidence 0.0) -- never a guessed match, and never a
        raised exception that would abort the rest of the run."""
        if not rows:
            return []

        questions = build_relevance_questions(rows, keywords)
        state = {"keywords": list(keywords)}
        row_ids = [row_id for row_id, _ in rows]

        try:
            result = self._agent.predict(state, questions)
            answers = result.get("answers", {})
        except Exception:
            # laya-mlx runs local inference rather than parsing a remote LLM's
            # text response, so failure here means a genuinely broken batch
            # (bad checkpoint state, non-finite outputs, etc.) -- isolate it to
            # this batch exactly like judge_batch's StructuredOutputError handling
            # in filtering/relevance.py.
            answers = {}

        verdicts = []
        for row_id in row_ids:
            answer = answers.get(str(row_id))
            if not answer or "noul" not in answer:
                verdicts.append(
                    LayaVerdict(
                        row_id=row_id, match=False, score=0.0, confidence=0.0, model=self.model_identifier
                    )
                )
                continue
            score = float(answer["noul"])
            confidence = float(answer.get("confidence", 0.0))
            verdicts.append(
                LayaVerdict(
                    row_id=row_id,
                    match=score >= threshold,
                    score=score,
                    confidence=confidence,
                    model=self.model_identifier,
                )
            )
        return verdicts
