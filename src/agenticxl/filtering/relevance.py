"""Batched semantic relevance judgment against the PRODUCT DETAILS column.

Per the approved plan, every row goes through an LLM judgment (no
deterministic keyword pre-filter -- empirically measured to miss ~31% of
true matches on this dataset, see the run log discussion) -- rows are
batched to bound the number of LLM calls on large files (e.g. JLG's 20,009
rows), with the LLM reasoning about semantic relevance (synonyms,
sub-components, variants), not literal substring matching.

Two lossless optimizations on top of that: rows with byte-identical
PRODUCT DETAILS text are judged once and share a verdict (duplicate text
always gets an identical judgment), and batches run concurrently through a
thread pool shared across the whole run -- not per file -- so Stage 1's
existing per-file concurrency (asyncio.gather in pipeline.py) doesn't get
multiplied into an unbounded number of simultaneous LLM calls.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from agenticxl.models import LLMClient, StructuredOutputError

DEFAULT_RELEVANCE_CONCURRENCY = 1

_executor_lock = threading.Lock()
_shared_executor: ThreadPoolExecutor | None = None
_shared_executor_size: int | None = None


def _get_shared_executor(max_workers: int) -> ThreadPoolExecutor:
    """A single ThreadPoolExecutor shared across every file being judged
    concurrently in a run. Recreated only if the requested size changes, so
    the concurrency cap is genuinely global rather than per-file."""
    global _shared_executor, _shared_executor_size
    with _executor_lock:
        if _shared_executor is None or _shared_executor_size != max_workers:
            if _shared_executor is not None:
                _shared_executor.shutdown(wait=False)
            _shared_executor = ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="relevance-judge"
            )
            _shared_executor_size = max_workers
        return _shared_executor

RELEVANCE_SCHEMA = {
    "title": "relevance_verdicts",
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "row_id": {"type": "integer"},
                    "match": {"type": "boolean"},
                    "rationale": {"type": "string"},
                },
                "required": ["row_id", "match", "rationale"],
            },
        },
    },
    "required": ["verdicts"],
}


@dataclass
class RelevanceVerdict:
    row_id: int
    match: bool
    rationale: str


def dedupe_truncation_artifact(text: str) -> str:
    """Strip the recurring 'truncated value re-concatenated with its own
    prefix' artifact seen in PRODUCT DETAILS text (see plan) -- a
    token-efficiency optimization for what's sent to the LLM prompt only.
    Never applied to data written into filtered/consolidated/final files."""
    if not text:
        return text
    for length in range(min(len(text), 400), 11, -1):
        prefix = text[:length]
        idx = text.find(prefix, length)
        if idx != -1:
            return text[:idx]
    return text


def build_batch_prompt(rows: list[tuple[int, str]], keywords: list[str]) -> str:
    keyword_list = "\n".join(f"- {k}" for k in keywords)
    lines = [f'Row {row_id}: "{dedupe_truncation_artifact(text)}"' for row_id, text in rows]
    rows_block = "\n".join(lines)
    return (
        "You are judging whether shipment records are semantically relevant to a set "
        "of keywords describing products of interest.\n\n"
        "Keywords (a row is relevant if its PRODUCT DETAILS text describes any of "
        "these, a clear synonym, or an obvious sub-component/variant/accessory of "
        "one of them -- not only a literal substring match):\n"
        f"{keyword_list}\n\n"
        "For EACH numbered row below, decide if it is relevant. Respond with a JSON "
        "object with a 'verdicts' array containing exactly one entry per row, each "
        "with: row_id (integer, copied from the row number), match (boolean), and "
        "rationale (one short sentence explaining the judgment).\n\n"
        f"{rows_block}"
    )


def judge_batch(
    rows: list[tuple[int, str]],
    keywords: list[str],
    llm_client: LLMClient,
) -> list[RelevanceVerdict]:
    """Judge one batch of (row_id, product_details_text) pairs. Rows missing
    from a malformed/partial LLM response are filled in as non-matches with
    a 'judgment_failed' rationale, so one bad batch doesn't abort the file."""
    if not rows:
        return []

    prompt = build_batch_prompt(rows, keywords)
    row_ids = [row_id for row_id, _ in rows]

    try:
        result = llm_client.complete_structured(prompt, RELEVANCE_SCHEMA)
        raw_verdicts = result.get("verdicts", [])
    except StructuredOutputError:
        raw_verdicts = []

    by_id: dict[int, RelevanceVerdict] = {}
    for v in raw_verdicts:
        try:
            rid = int(v["row_id"])
            by_id[rid] = RelevanceVerdict(
                row_id=rid,
                match=bool(v["match"]),
                rationale=str(v.get("rationale", "")),
            )
        except (KeyError, ValueError, TypeError):
            continue

    return [
        by_id.get(row_id) or RelevanceVerdict(row_id=row_id, match=False, rationale="judgment_failed")
        for row_id in row_ids
    ]


def judge_all_rows(
    rows: list[dict],
    product_details_column: str,
    keywords: list[str],
    llm_client: LLMClient,
    batch_size: int = 50,
    max_concurrency: int = DEFAULT_RELEVANCE_CONCURRENCY,
) -> list[RelevanceVerdict]:
    """Judge every row in `rows` (indexed 0..len(rows)-1 as row_id).

    Rows with byte-identical PRODUCT DETAILS text are judged only once --
    lossless, since identical text always gets an identical judgment -- and
    every row sharing that text gets a copy of the same verdict, still
    logged individually by its own row_id.

    `max_concurrency` defaults to 1 (fully sequential, matching prior
    behavior) so existing callers/tests are unaffected; production callers
    pass RunConfig.relevance_concurrency to run batches concurrently through
    a thread pool shared across the whole run (see _get_shared_executor).
    """
    text_to_row_ids: dict[str, list[int]] = {}
    for row_id, row in enumerate(rows):
        text = str(row.get(product_details_column) or "")
        text_to_row_ids.setdefault(text, []).append(row_id)

    unique_texts = list(text_to_row_ids.keys())

    batches: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for text_index, text in enumerate(unique_texts):
        current.append((text_index, text))
        if len(current) >= batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)

    verdict_by_text_index: dict[int, RelevanceVerdict] = {}
    if max_concurrency <= 1 or len(batches) <= 1:
        for batch in batches:
            for v in judge_batch(batch, keywords, llm_client):
                verdict_by_text_index[v.row_id] = v
    else:
        executor = _get_shared_executor(max_concurrency)
        futures = [executor.submit(judge_batch, batch, keywords, llm_client) for batch in batches]
        for future in futures:
            for v in future.result():
                verdict_by_text_index[v.row_id] = v

    all_verdicts: list[RelevanceVerdict] = []
    for text_index, text in enumerate(unique_texts):
        v = verdict_by_text_index.get(text_index) or RelevanceVerdict(
            row_id=text_index, match=False, rationale="judgment_failed"
        )
        for row_id in text_to_row_ids[text]:
            all_verdicts.append(RelevanceVerdict(row_id=row_id, match=v.match, rationale=v.rationale))

    all_verdicts.sort(key=lambda v: v.row_id)
    return all_verdicts


def write_rationale_log(path: str | Path, source_file: str, verdicts: list[RelevanceVerdict]) -> Path:
    """Write one JSON line per row verdict, for auditability of why each row
    was kept or dropped."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for v in verdicts:
            record = {"source_file": source_file, **asdict(v)}
            f.write(json.dumps(record) + "\n")
    return path
