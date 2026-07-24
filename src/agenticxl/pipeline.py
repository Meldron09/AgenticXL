"""Top-level pipeline entrypoint: Stage 1 (parallel per-file filtering) ->
Stage 2 (consolidation) -> Stage 3 (enrichment) -> run report.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from langchain_core.messages import HumanMessage

from agenticxl.agents.enrichment_agent import ENRICHMENT_SUBAGENT_NAME, build_enrichment_subagent
from agenticxl.agents.file_subagent import FILE_SUBAGENT_NAME, build_file_subagent
from agenticxl.agents.orchestrator import build_orchestrator, extract_last_tool_result
from agenticxl.agents.schemas import EnrichmentSummary, FileResult
from agenticxl.config import RunConfig
from agenticxl.consolidation.consolidate import consolidate_filtered_files
from agenticxl.enrichment.web_search import SerperClient
from agenticxl.models import LLMClient, get_llm_client
from agenticxl.reporting.report import RunReport, write_report


async def _invoke_for_file(orchestrator, file_path: Path) -> FileResult:
    message = (
        f"Process the file at path: {file_path}\n"
        f"Use the `{FILE_SUBAGENT_NAME}` subagent (via the task tool) to filter it."
    )
    try:
        state = await orchestrator.ainvoke({"messages": [HumanMessage(content=message)]})
        parsed = extract_last_tool_result(state["messages"])
        if parsed is None:
            raise RuntimeError("Could not parse a structured result from the orchestrator's output")
        return FileResult(**parsed)
    except Exception as exc:
        return FileResult(
            source_file=file_path.name,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_stage1(run_config: RunConfig, llm_client: LLMClient, run_logs_dir: Path) -> list[FileResult]:
    """Fan out one orchestrator invocation per input file, concurrently.

    Each invocation gets its own fresh graph state (no checkpointer/shared
    thread id), so files are logically isolated from one another; a
    per-invocation try/except means one file's exception is captured as an
    error-status FileResult rather than aborting the other files.
    """
    subagent = build_file_subagent(run_config, llm_client, run_logs_dir)
    orchestrator = build_orchestrator(llm_client, [subagent])

    input_files = sorted(run_config.input_dir_resolved.glob("*.xlsx"))
    if not input_files:
        raise FileNotFoundError(f"No .xlsx files found in {run_config.input_dir_resolved}")

    results = await asyncio.gather(*(_invoke_for_file(orchestrator, f) for f in input_files))
    return list(results)


def run_stage2(run_config: RunConfig, file_results: list[FileResult]) -> tuple[Path, list[str]]:
    ok_results = [r for r in file_results if r.status == "ok" and r.output_path]
    output_path = run_config.consolidated_dir_resolved / "consolidated.xlsx"
    result = consolidate_filtered_files(
        filtered_file_paths=[Path(r.output_path) for r in ok_results],
        source_file_names=[r.source_file for r in ok_results],
        output_path=output_path,
    )
    return result.output_path, result.warnings


async def run_stage3(
    run_config: RunConfig,
    llm_client: LLMClient,
    consolidated_path: Path,
    serper_client: SerperClient | None,
) -> EnrichmentSummary:
    subagent = build_enrichment_subagent(run_config, llm_client, serper_client)
    orchestrator = build_orchestrator(llm_client, [subagent])
    message = (
        f"Process the consolidated file at path: {consolidated_path}\n"
        f"Use the `{ENRICHMENT_SUBAGENT_NAME}` subagent (via the task tool) to enrich it."
    )
    try:
        state = await orchestrator.ainvoke({"messages": [HumanMessage(content=message)]})
        parsed = extract_last_tool_result(state["messages"])
        if parsed is None:
            raise RuntimeError("Could not parse a structured result from the orchestrator's output")
        return EnrichmentSummary(**parsed)
    except Exception as exc:
        return EnrichmentSummary(status="error", error=f"{type(exc).__name__}: {exc}")


async def run(
    config_path: str | Path,
    project_root: str | Path | None = None,
    skip_enrichment: bool | None = None,
) -> RunReport:
    """Run Stage 1 -> Stage 2 -> (optionally) Stage 3 -> report.

    `skip_enrichment`, when not None, overrides whatever `RunConfig.load`
    resolved from AGENTICXL_SKIP_ENRICHMENT/config -- this is how the CLI's
    `--skip-enrichment` flag takes precedence. When Stage 3 is skipped, no
    enrichment subagent/orchestrator is built and no Serper client is even
    constructed, so it's a real skip (no extra LLM call, no duplicate
    output file), not just an empty-columns no-op.
    """
    run_config = RunConfig.load(config_path, project_root=project_root)
    if skip_enrichment is not None:
        run_config.skip_enrichment = skip_enrichment
    run_config.model.validate_backend_ready()
    run_config.ensure_directories()

    llm_client = get_llm_client(run_config.model)

    file_results = await run_stage1(run_config, llm_client, run_config.logs_dir_resolved)
    consolidated_path, warnings = run_stage2(run_config, file_results)

    enrichment_summary = None
    if not run_config.skip_enrichment:
        serper_client = SerperClient(run_config.serper_api_key) if run_config.serper_api_key else None
        enrichment_summary = await run_stage3(run_config, llm_client, consolidated_path, serper_client)

    report = RunReport(
        file_results=file_results,
        consolidation_warnings=warnings,
        enrichment_summary=enrichment_summary,
        enrichment_skipped=run_config.skip_enrichment,
    )
    write_report(report, run_config.logs_dir_resolved)
    return report
