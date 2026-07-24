"""Stage 3, step 1: try to derive an enrichment column's value from data
already present in the same row, before falling back to web search."""

from __future__ import annotations

from dataclasses import dataclass

from agenticxl.models import LLMClient, StructuredOutputError

LOCAL_DERIVATION_SCHEMA = {
    "title": "local_derivation_results",
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "row_id": {"type": "integer"},
                    "value": {"type": ["string", "null"]},
                    "resolved": {"type": "boolean"},
                    "rationale": {"type": "string"},
                },
                "required": ["row_id", "resolved", "rationale"],
            },
        },
    },
    "required": ["results"],
}


@dataclass
class LocalDerivationResult:
    row_id: int
    value: str | None
    resolved: bool
    rationale: str


def build_local_derivation_prompt(rows: list[tuple[int, dict]], column: str) -> str:
    lines = []
    for row_id, row in rows:
        context = {k: v for k, v in row.items() if v not in (None, "")}
        lines.append(f"Row {row_id}: {context}")
    rows_block = "\n".join(lines)
    return (
        f'For each row below, determine the value for a new column called "{column}" '
        "using ONLY the data already present in that row (you may compute or infer it "
        "from other fields -- e.g. combining, categorizing, or reformatting existing "
        "values -- but do not guess facts that aren't supported by the row's data).\n\n"
        "Respond with a JSON object with a 'results' array, one entry per row, each with: "
        "row_id (integer, copied from the row number), value (string, the derived value, "
        "or null if not resolvable from this row alone), resolved (boolean, true only if "
        "you are confident the value is correct and supported by the row's own data), "
        "rationale (short sentence).\n\n"
        f"Rows:\n{rows_block}"
    )


def derive_local_batch(
    rows: list[tuple[int, dict]],
    column: str,
    llm_client: LLMClient,
) -> list[LocalDerivationResult]:
    """Batched local-derivation attempt for one enrichment column across a
    batch of rows. Rows the LLM couldn't confidently resolve (or that are
    missing from a malformed response) come back as resolved=False so the
    caller falls through to the web-search stage."""
    if not rows:
        return []

    prompt = build_local_derivation_prompt(rows, column)
    row_ids = [row_id for row_id, _ in rows]

    try:
        result = llm_client.complete_structured(prompt, LOCAL_DERIVATION_SCHEMA)
        raw_results = result.get("results", [])
    except StructuredOutputError:
        raw_results = []

    by_id: dict[int, LocalDerivationResult] = {}
    for r in raw_results:
        try:
            rid = int(r["row_id"])
            by_id[rid] = LocalDerivationResult(
                row_id=rid,
                value=r.get("value"),
                resolved=bool(r.get("resolved")),
                rationale=str(r.get("rationale", "")),
            )
        except (KeyError, ValueError, TypeError):
            continue

    return [
        by_id.get(rid)
        or LocalDerivationResult(row_id=rid, value=None, resolved=False, rationale="derivation_failed")
        for rid in row_ids
    ]
