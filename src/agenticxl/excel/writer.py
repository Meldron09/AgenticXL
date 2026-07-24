from __future__ import annotations

from pathlib import Path
from typing import Any

from openpyxl import Workbook


def write_rows_to_excel(
    file_path: str | Path,
    header: list[str],
    rows: list[dict[str, Any]],
    pre_header_rows: list[tuple[Any, ...]] | None = None,
    merges: list[str] | None = None,
) -> Path:
    """Write rows (each a dict keyed by column name) to a new xlsx file with
    the given header as the header row. Missing keys in a row become blank
    cells.

    `pre_header_rows`/`merges` optionally reproduce the source file's merged
    group-banner row(s) above the header (as captured by
    `excel.reader.load_main_sheet`), so a filtered/subset output can be
    structurally identical to its input, not just header-and-data. When
    omitted, the output has a single clean header row with no banner.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    for pre_row in pre_header_rows or []:
        ws.append(list(pre_row))
    for merge_range in merges or []:
        ws.merge_cells(merge_range)
    ws.append(header)
    for row in rows:
        ws.append([row.get(col) for col in header])
    wb.save(file_path)
    return file_path
