"""Stage 3 orchestration: for each enrichment column, try local derivation
first (batched), then fall back to web search per still-unresolved row, and
record which source produced each cell -- never silently fabricating a
value the process can't support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agenticxl.models import LLMClient

from .derive_local import derive_local_batch
from .synthesize import synthesize_value
from .web_search import SerperAPIError, SerperClient, generate_search_query

SOURCE_LOCAL = "local"
SOURCE_WEB = "web_search"
SOURCE_UNRESOLVED = "unresolved"


@dataclass
class EnrichmentRecord:
    row_id: int
    column: str
    value: Any
    source: str
    rationale: str


def enrich_column(
    rows: list[dict],
    column: str,
    llm_client: LLMClient,
    serper_client: SerperClient | None,
    batch_size: int = 25,
) -> list[EnrichmentRecord]:
    """Enrich a single column across all rows: batched local derivation
    first, then web-search fallback per row still unresolved."""
    indexed_rows = list(enumerate(rows))
    records: dict[int, EnrichmentRecord] = {}

    for start in range(0, len(indexed_rows), batch_size):
        batch = indexed_rows[start : start + batch_size]
        for res in derive_local_batch(batch, column, llm_client):
            if res.resolved and res.value:
                records[res.row_id] = EnrichmentRecord(
                    row_id=res.row_id,
                    column=column,
                    value=res.value,
                    source=SOURCE_LOCAL,
                    rationale=res.rationale,
                )

    unresolved_ids = [row_id for row_id, _ in indexed_rows if row_id not in records]
    for row_id in unresolved_ids:
        row = rows[row_id]
        if serper_client is None:
            records[row_id] = EnrichmentRecord(
                row_id=row_id,
                column=column,
                value=None,
                source=SOURCE_UNRESOLVED,
                rationale="no_serper_client_configured",
            )
            continue
        try:
            query = generate_search_query(column, row, llm_client)
            search_results = serper_client.search(query)
            synthesis = synthesize_value(column, row, search_results, llm_client)
        except SerperAPIError as exc:
            records[row_id] = EnrichmentRecord(
                row_id=row_id,
                column=column,
                value=None,
                source=SOURCE_UNRESOLVED,
                rationale=f"web_search_failed: {exc}",
            )
            continue

        if synthesis.resolved and synthesis.value:
            records[row_id] = EnrichmentRecord(
                row_id=row_id,
                column=column,
                value=synthesis.value,
                source=SOURCE_WEB,
                rationale=synthesis.rationale,
            )
        else:
            records[row_id] = EnrichmentRecord(
                row_id=row_id,
                column=column,
                value=None,
                source=SOURCE_UNRESOLVED,
                rationale=synthesis.rationale or "not_found",
            )

    return [records[row_id] for row_id, _ in indexed_rows]


def apply_enrichment(
    rows: list[dict],
    columns: list[str],
    llm_client: LLMClient,
    serper_client: SerperClient | None,
    batch_size: int = 25,
) -> tuple[list[dict], list[EnrichmentRecord]]:
    """Returns (enriched_rows, all_records). Each enriched row is a copy of
    the input row with every enrichment column set, plus a '<column> Source'
    companion column recording local/web_search/unresolved."""
    enriched_rows = [dict(r) for r in rows]
    all_records: list[EnrichmentRecord] = []

    for column in columns:
        records = enrich_column(rows, column, llm_client, serper_client, batch_size=batch_size)
        all_records.extend(records)
        for rec in records:
            enriched_rows[rec.row_id][column] = rec.value
            enriched_rows[rec.row_id][f"{column} Source"] = rec.source

    return enriched_rows, all_records
