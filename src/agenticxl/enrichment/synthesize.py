"""Stage 3, step 2: synthesize a column value from web search snippets. Must
not fabricate -- returns resolved=False whenever the snippets don't clearly
and specifically support a value."""

from __future__ import annotations

from dataclasses import dataclass

from agenticxl.models import LLMClient, StructuredOutputError

from .web_search import SearchResult

SYNTHESIZE_SCHEMA = {
    "title": "synthesis_result",
    "type": "object",
    "properties": {
        "value": {"type": ["string", "null"]},
        "resolved": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["resolved", "rationale"],
}


@dataclass
class SynthesisResult:
    value: str | None
    resolved: bool
    rationale: str


def build_synthesis_prompt(column: str, row_context: dict, search_results: list[SearchResult]) -> str:
    context = {k: v for k, v in row_context.items() if v not in (None, "")}
    snippets = (
        "\n".join(f"- {r.title}: {r.snippet} ({r.link})" for r in search_results)
        or "(no results returned)"
    )
    return (
        f'You need to determine the value of a column called "{column}" for this '
        f"shipment record, using the web search results below.\n\n"
        f"Row data: {context}\n\n"
        f"Search results:\n{snippets}\n\n"
        "Respond with a JSON object: value (string or null), resolved (boolean, true "
        "only if the search results clearly and specifically support the value -- do "
        "not guess or extrapolate beyond what the snippets actually say), rationale "
        "(short sentence citing which result supported the value, or explaining why "
        "it's unresolved)."
    )


def synthesize_value(
    column: str,
    row_context: dict,
    search_results: list[SearchResult],
    llm_client: LLMClient,
) -> SynthesisResult:
    prompt = build_synthesis_prompt(column, row_context, search_results)
    try:
        result = llm_client.complete_structured(prompt, SYNTHESIZE_SCHEMA)
        return SynthesisResult(
            value=result.get("value"),
            resolved=bool(result.get("resolved")),
            rationale=str(result.get("rationale", "")),
        )
    except StructuredOutputError:
        return SynthesisResult(value=None, resolved=False, rationale="synthesis_failed")
