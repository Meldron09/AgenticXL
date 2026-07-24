from __future__ import annotations

import re

from agenticxl.enrichment.enrich import SOURCE_LOCAL, SOURCE_UNRESOLVED, SOURCE_WEB, apply_enrichment
from agenticxl.enrichment.web_search import SearchResult
from helpers import FakeLLMClient


def _ids_in_prompt(prompt: str) -> list[int]:
    return [int(m) for m in re.findall(r"Row (\d+):", prompt)]


class FakeSerperClient:
    def __init__(self, results: list[SearchResult]):
        self._results = results
        self.search_calls: list[str] = []

    def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        self.search_calls.append(query)
        return self._results


def _make_llm(local_resolved_row_ids: set[int], synthesis_value: str | None = "Wonder Corp"):
    def structured_responder(prompt, schema):
        if "results" in schema.get("properties", {}):
            ids = _ids_in_prompt(prompt)
            return {
                "results": [
                    {
                        "row_id": i,
                        "value": "LocalValue" if i in local_resolved_row_ids else None,
                        "resolved": i in local_resolved_row_ids,
                        "rationale": "from row data" if i in local_resolved_row_ids else "not enough info",
                    }
                    for i in ids
                ]
            }
        # synthesize_value schema
        if synthesis_value is None:
            return {"value": None, "resolved": False, "rationale": "snippets inconclusive"}
        return {"value": synthesis_value, "resolved": True, "rationale": "from search snippet"}

    return FakeLLMClient(structured_responder=structured_responder, text_responder=lambda p: "some search query")


def test_local_derivation_resolves_without_web_search():
    rows = [{"PRODUCT DETAILS": "acme widget", "BRAND NAME": "Acme"}]
    llm = _make_llm(local_resolved_row_ids={0})
    serper = FakeSerperClient([])

    enriched, records = apply_enrichment(rows, ["Manufacturer"], llm, serper)

    assert enriched[0]["Manufacturer"] == "LocalValue"
    assert enriched[0]["Manufacturer Source"] == SOURCE_LOCAL
    assert records[0].source == SOURCE_LOCAL
    assert serper.search_calls == []  # never touched Serper when local derivation succeeded


def test_web_search_fallback_used_when_local_unresolved():
    rows = [{"PRODUCT DETAILS": "mystery gadget", "BRAND NAME": None}]
    llm = _make_llm(local_resolved_row_ids=set())
    serper = FakeSerperClient([SearchResult(title="t", snippet="Made by Wonder Corp", link="http://x")])

    enriched, records = apply_enrichment(rows, ["Manufacturer"], llm, serper)

    assert enriched[0]["Manufacturer"] == "Wonder Corp"
    assert enriched[0]["Manufacturer Source"] == SOURCE_WEB
    assert records[0].source == SOURCE_WEB
    assert len(serper.search_calls) == 1


def test_unresolved_when_no_serper_client_configured():
    rows = [{"PRODUCT DETAILS": "mystery gadget"}]
    llm = _make_llm(local_resolved_row_ids=set())

    enriched, records = apply_enrichment(rows, ["Manufacturer"], llm, None)

    assert enriched[0]["Manufacturer"] is None
    assert enriched[0]["Manufacturer Source"] == SOURCE_UNRESOLVED
    assert records[0].rationale == "no_serper_client_configured"


def test_unresolved_when_search_snippets_inconclusive_never_fabricates():
    rows = [{"PRODUCT DETAILS": "mystery gadget"}]
    llm = _make_llm(local_resolved_row_ids=set(), synthesis_value=None)
    serper = FakeSerperClient([SearchResult(title="t", snippet="unrelated content", link="http://x")])

    enriched, records = apply_enrichment(rows, ["Manufacturer"], llm, serper)

    assert enriched[0]["Manufacturer"] is None
    assert enriched[0]["Manufacturer Source"] == SOURCE_UNRESOLVED
    assert records[0].source == SOURCE_UNRESOLVED


def test_mixed_rows_get_correct_independent_sources():
    rows = [
        {"PRODUCT DETAILS": "acme widget", "BRAND NAME": "Acme"},
        {"PRODUCT DETAILS": "mystery gadget", "BRAND NAME": None},
    ]
    llm = _make_llm(local_resolved_row_ids={0})
    serper = FakeSerperClient([SearchResult(title="t", snippet="Made by Wonder Corp", link="http://x")])

    enriched, records = apply_enrichment(rows, ["Manufacturer"], llm, serper)

    assert enriched[0]["Manufacturer Source"] == SOURCE_LOCAL
    assert enriched[1]["Manufacturer Source"] == SOURCE_WEB


def test_serper_client_caches_repeated_queries():
    from agenticxl.enrichment.web_search import SerperClient

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"organic": [{"title": "t", "snippet": "s", "link": "http://x"}]}

    class FakeSession:
        def __init__(self):
            self.post_calls = 0

        def post(self, *args, **kwargs):
            self.post_calls += 1
            return FakeResponse()

    session = FakeSession()
    client = SerperClient(api_key="fake-key", session=session)

    client.search("same query")
    client.search("Same Query  ")  # normalized to the same cache key
    client.search("same query")

    assert session.post_calls == 1
