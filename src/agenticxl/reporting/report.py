"""Aggregates per-run outcomes (files processed, rows kept/dropped,
enrichment source breakdown, errors) into a JSON + Markdown report."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from agenticxl.agents.schemas import EnrichmentSummary, FileResult


@dataclass
class RunReport:
    file_results: list[FileResult] = field(default_factory=list)
    consolidation_warnings: list[str] = field(default_factory=list)
    enrichment_summary: EnrichmentSummary | None = None
    enrichment_skipped: bool = False

    @property
    def files_ok(self) -> list[FileResult]:
        return [r for r in self.file_results if r.status == "ok"]

    @property
    def files_error(self) -> list[FileResult]:
        return [r for r in self.file_results if r.status == "error"]

    @property
    def total_rows_in(self) -> int:
        return sum(r.rows_in for r in self.file_results)

    @property
    def total_rows_kept(self) -> int:
        return sum(r.rows_kept for r in self.file_results)


def write_report(report: RunReport, logs_dir: str | Path) -> tuple[Path, Path]:
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    json_path = logs_dir / "run_report.json"
    md_path = logs_dir / "run_report.md"

    payload = {
        "files": [r.model_dump() for r in report.file_results],
        "consolidation_warnings": report.consolidation_warnings,
        "enrichment_summary": (
            report.enrichment_summary.model_dump() if report.enrichment_summary else None
        ),
        "enrichment_skipped": report.enrichment_skipped,
        "totals": {
            "files_processed": len(report.file_results),
            "files_ok": len(report.files_ok),
            "files_error": len(report.files_error),
            "rows_in": report.total_rows_in,
            "rows_kept": report.total_rows_kept,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    lines = ["# Pipeline Run Report", "", "## Files", "", "| File | Status | Rows In | Rows Kept | Error |", "|---|---|---|---|---|"]
    for r in report.file_results:
        lines.append(f"| {r.source_file} | {r.status} | {r.rows_in} | {r.rows_kept} | {r.error or ''} |")
    lines.append("")
    lines.append(
        f"**Totals:** {len(report.files_ok)}/{len(report.file_results)} files ok, "
        f"{report.total_rows_kept}/{report.total_rows_in} rows kept."
    )
    lines.append("")

    if report.consolidation_warnings:
        lines.append("## Consolidation warnings")
        lines.append("")
        lines.extend(f"- {w}" for w in report.consolidation_warnings)
        lines.append("")

    if report.enrichment_skipped:
        lines.append("## Enrichment")
        lines.append("")
        lines.append("- Skipped (--skip-enrichment / AGENTICXL_SKIP_ENRICHMENT). ")
        lines.append("  The consolidated file from Stage 2 is the final deliverable for this run.")
        lines.append("")
    elif report.enrichment_summary:
        es = report.enrichment_summary
        lines.append("## Enrichment")
        lines.append("")
        lines.append(f"- Status: {es.status}")
        if es.error:
            lines.append(f"- Error: {es.error}")
        lines.append(f"- Output: {es.output_path}")
        lines.append(f"- Rows: {es.rows}")
        lines.append("- Source breakdown:")
        for source, count in es.source_counts.items():
            lines.append(f"  - {source}: {count}")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return json_path, md_path
