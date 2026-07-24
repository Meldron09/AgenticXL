"""Stage 2: merge per-file filtered xlsx outputs into one consolidated
workbook, reconciled to the canonical schema with a provenance column."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agenticxl.excel.reader import load_main_sheet
from agenticxl.excel.schema import CANONICAL_COLUMNS, PROVENANCE_COLUMN, reconcile_to_canonical
from agenticxl.excel.writer import write_rows_to_excel


@dataclass
class ConsolidationResult:
    output_path: Path
    header: list[str]
    total_rows: int
    warnings: list[str] = field(default_factory=list)


def consolidate_filtered_files(
    filtered_file_paths: list[Path],
    source_file_names: list[str],
    output_path: str | Path,
    canonical: list[str] | None = None,
) -> ConsolidationResult:
    """Reads each filtered file, reconciles its header to the canonical
    schema (tolerating schema drift), tags every row with its source file
    for provenance, and writes one consolidated xlsx."""
    canonical = canonical if canonical is not None else CANONICAL_COLUMNS

    all_rows: list[dict] = []
    warnings: list[str] = []
    extra_columns_seen: list[str] = []

    for path, source_name in zip(filtered_file_paths, source_file_names):
        sheet = load_main_sheet(path)
        result = reconcile_to_canonical(
            sheet.header, sheet.rows, source_file=source_name, canonical=canonical
        )
        warnings.extend(result.warnings)
        for col in result.header:
            if col not in canonical and col != PROVENANCE_COLUMN and col not in extra_columns_seen:
                extra_columns_seen.append(col)
        all_rows.extend(result.rows)

    final_header = list(canonical) + extra_columns_seen + [PROVENANCE_COLUMN]
    write_rows_to_excel(output_path, final_header, all_rows)

    return ConsolidationResult(
        output_path=Path(output_path),
        header=final_header,
        total_rows=len(all_rows),
        warnings=warnings,
    )
