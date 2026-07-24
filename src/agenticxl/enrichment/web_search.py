"""Serper API integration -- the web-search fallback used only in Stage 3,
never bound as a tool to the Stage 1 filtering subagents."""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from agenticxl.models import LLMClient

SERPER_URL = "https://google.serper.dev/search"


class SerperAPIError(RuntimeError):
    """Raised when the Serper API fails after all retries. Callers should
    mark the affected cell 'unresolved' rather than fabricate a value."""


@dataclass
class SearchResult:
    title: str
    snippet: str
    link: str


class SerperClient:
    """Thin Serper client with retry/backoff and an in-memory query cache.

    The cache is keyed on the exact query string -- callers construct
    queries that already fold in the target column's intent, so identical
    (column, product) combinations naturally share a cache entry without
    re-hitting the API. This matters because many consolidated rows
    describe the same brand/model across different shipments."""

    def __init__(self, api_key: str, session: requests.Session | None = None, max_retries: int = 3):
        if not api_key:
            raise ValueError("Serper API key is required")
        self._api_key = api_key
        self._session = session or requests.Session()
        self._max_retries = max_retries
        self._cache: dict[str, list[SearchResult]] = {}

    def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        cache_key = " ".join(query.strip().lower().split())
        if cache_key in self._cache:
            return self._cache[cache_key]

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._session.post(
                    SERPER_URL,
                    headers={"X-API-KEY": self._api_key, "Content-Type": "application/json"},
                    json={"q": query, "num": num_results},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                results = [
                    SearchResult(
                        title=item.get("title", ""),
                        snippet=item.get("snippet", ""),
                        link=item.get("link", ""),
                    )
                    for item in data.get("organic", [])[:num_results]
                ]
                self._cache[cache_key] = results
                return results
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    time.sleep(2**attempt)

        raise SerperAPIError(
            f"Serper search failed after {self._max_retries} attempts: {last_exc}"
        ) from last_exc


def build_search_query_prompt(column: str, row_context: dict) -> str:
    context = {k: v for k, v in row_context.items() if v not in (None, "")}
    return (
        f"Given this shipment record: {context}\n\n"
        f'Write a single, concise web search query (no quotes, no explanation) that '
        f'would help find the value of "{column}" for this specific product/shipment.'
    )


def generate_search_query(column: str, row_context: dict, llm_client: LLMClient) -> str:
    prompt = build_search_query_prompt(column, row_context)
    return llm_client.complete(prompt).strip().strip('"')
