"""Custom domain tools bound to deepagents subagents.

xlsx is a binary (zip) format, so these tools do NOT use deepagents' generic
`read_file`/`write_file` tools (which operate on plain text via
`FilesystemBackend`) -- they do their own I/O with openpyxl/our excel
package and hand the agent back small JSON-serializable summaries.

Each tool's real logic lives in a plain `_*_impl` function so it can be unit
tested directly without going through the LangChain `@tool` machinery or a
live LLM agent loop.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from agenticxl.config import RunConfig
from agenticxl.enrichment.enrich import apply_enrichment
from agenticxl.enrichment.web_search import SerperClient
from agenticxl.excel.reader import load_main_sheet
from agenticxl.excel.writer import write_rows_to_excel
from agenticxl.filtering.relevance import judge_all_rows, write_rationale_log
from agenticxl.models import LLMClient


def process_file_impl(
    file_path: str,
    run_config: RunConfig,
    llm_client: LLMClient,
    run_logs_dir: Path,
) -> dict:
    """Stage 1 per-file work: load, judge every row for relevance, write the
    filtered output + rationale log. Never raises -- any failure becomes an
    error-status result so the caller can isolate it from other files."""
    source = Path(file_path)
    try:
        sheet = load_main_sheet(source)
    except Exception as exc:  # malformed file, missing header, missing column, etc.
        return {
            "source_file": source.name,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "rows_in": 0,
            "rows_kept": 0,
            "output_path": None,
            "rationale_log_path": None,
        }

    try:
        verdicts = judge_all_rows(
            sheet.rows,
            sheet.product_details_column,
            run_config.keywords,
            llm_client,
            batch_size=run_config.relevance_batch_size,
            max_concurrency=run_config.relevance_concurrency,
        )
        kept_rows = [sheet.rows[v.row_id] for v in verdicts if v.match]

        stem = source.stem
        output_path = run_config.filtered_dir_resolved / f"filtered_{stem}.xlsx"
        write_rows_to_excel(
            output_path,
            sheet.header,
            kept_rows,
            pre_header_rows=sheet.pre_header_rows,
            merges=sheet.merges,
        )

        rationale_path = run_logs_dir / f"{stem}_rationale.jsonl"
        write_rationale_log(rationale_path, source.name, verdicts)
    except Exception as exc:
        return {
            "source_file": source.name,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "rows_in": len(sheet.rows),
            "rows_kept": 0,
            "output_path": None,
            "rationale_log_path": None,
        }

    return {
        "source_file": source.name,
        "status": "ok",
        "error": None,
        "rows_in": len(sheet.rows),
        "rows_kept": len(kept_rows),
        "output_path": str(output_path),
        "rationale_log_path": str(rationale_path),
    }


def enrich_columns_impl(
    consolidated_file_path: str,
    run_config: RunConfig,
    llm_client: LLMClient,
    serper_client: SerperClient | None,
) -> dict:
    """Stage 3 work: load the consolidated file, populate every configured
    enrichment column (local-first, web-search fallback), write the final
    enriched output."""
    try:
        sheet = load_main_sheet(consolidated_file_path)
        enriched_rows, records = apply_enrichment(
            sheet.rows,
            run_config.enrichment_columns,
            llm_client,
            serper_client,
            batch_size=run_config.enrichment_batch_size,
        )
        enrichment_header: list[str] = []
        for col in run_config.enrichment_columns:
            enrichment_header.extend([col, f"{col} Source"])
        final_header = sheet.header + enrichment_header

        output_path = run_config.output_dir_resolved / "final_enriched.xlsx"
        write_rows_to_excel(output_path, final_header, enriched_rows)

        source_counts: dict[str, int] = {}
        for rec in records:
            source_counts[rec.source] = source_counts.get(rec.source, 0) + 1

        return {
            "status": "ok",
            "error": None,
            "output_path": str(output_path),
            "rows": len(enriched_rows),
            "source_counts": source_counts,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "output_path": None,
            "rows": 0,
            "source_counts": {},
        }


def make_process_file_tool(run_config: RunConfig, llm_client: LLMClient, run_logs_dir: Path):
    @tool
    def process_file(file_path: str) -> str:
        """Load one raw Excel export file, judge every row's PRODUCT DETAILS
        text for semantic relevance to the configured keywords, and write
        matching rows to a filtered xlsx file. Returns a JSON summary with
        source_file, status, error, rows_in, rows_kept, output_path, and
        rationale_log_path."""
        result = process_file_impl(file_path, run_config, llm_client, run_logs_dir)
        return json.dumps(result)

    return process_file


def make_enrich_columns_tool(
    run_config: RunConfig,
    llm_client: LLMClient,
    serper_client: SerperClient | None,
):
    @tool
    def enrich_columns(consolidated_file_path: str) -> str:
        """Load the consolidated xlsx file and populate every configured
        enrichment column, deriving from existing row data first and
        falling back to Serper web search only when necessary. Returns a
        JSON summary with status, error, output_path, rows, and
        source_counts (local/web_search/unresolved)."""
        result = enrich_columns_impl(consolidated_file_path, run_config, llm_client, serper_client)
        return json.dumps(result)

    return enrich_columns
