"""Excel reading with dynamic header detection.

Ground truth from inspecting all 14 sample files (see the implementation
plan): every file has a merged "group banner" row before the real 45-column
header row, except Genie which has an extra stray-cell row before the
banner too. A fixed "skip N rows" rule breaks on Genie, so the header row
is detected by content: the first of the first few rows with a high
non-blank cell count AND a cell that matches "PRODUCT DETAILS"
case-insensitively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

HEADER_SCAN_ROWS = 5
MIN_NONBLANK_HEADER_CELLS = 20
PRODUCT_DETAILS_NAME = "product details"


class HeaderNotFoundError(Exception):
    """Raised when no header row can be found within the scanned rows."""


class ProductDetailsColumnNotFoundError(Exception):
    """Raised when a header row is found but no column matches
    'PRODUCT DETAILS' case-insensitively."""


@dataclass
class SheetData:
    file_path: Path
    header: list[str]
    product_details_column: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    pre_header_rows: list[tuple[Any, ...]] = field(default_factory=list)
    merges: list[str] = field(default_factory=list)


def _normalize_cell(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def find_header_row(
    row_values: list[tuple[Any, ...]],
    min_nonblank: int = MIN_NONBLANK_HEADER_CELLS,
) -> tuple[int, list[str]]:
    """Given the first few rows (as tuples of raw cell values, 0-indexed
    list), return (row_index, header) for the first row with at least
    `min_nonblank` non-blank cells -- treated as the real header row, as
    opposed to a merged group-banner row above it (which has far fewer
    non-blank cells, since only each merge's top-left cell holds a value).

    Deliberately does NOT check for a 'PRODUCT DETAILS' cell here: that
    check is a separate, more specific step (`find_product_details_column`)
    so a file with a real header but no matching column raises the more
    informative `ProductDetailsColumnNotFoundError` instead of being
    indistinguishable from a file with no header row at all.
    """
    for idx, values in enumerate(row_values):
        cells = [_normalize_cell(v) for v in values]
        non_blank = [c for c in cells if c is not None]
        if len(non_blank) >= min_nonblank:
            return idx, [c if c is not None else "" for c in cells]
    raise HeaderNotFoundError(
        f"No row with >= {min_nonblank} non-blank cells found in the first {len(row_values)} rows."
    )


def find_product_details_column(header: list[str]) -> str:
    for name in header:
        if name and name.strip().lower() == PRODUCT_DETAILS_NAME:
            return name
    raise ProductDetailsColumnNotFoundError(
        f"No column matching 'PRODUCT DETAILS' (case-insensitive) in header: {header}"
    )


def _read_pre_header_merges(file_path: Path, header_offset: int) -> list[str]:
    """Return merge range strings (e.g. "A1:B1") that fall entirely within
    the pre-header rows (1-indexed rows 1..header_offset) -- i.e. the merged
    group-banner row(s) above the real header, not anything in the header or
    data rows.

    Requires a second, non-read-only open (read-only worksheets don't expose
    `merged_cells`); limited to the file's merge metadata, not a full-sheet
    scan, so the cost is roughly constant regardless of row count.
    """
    if header_offset <= 0:
        return []
    wb = load_workbook(file_path, data_only=True)
    try:
        ws = wb.worksheets[0]
        return [str(r) for r in ws.merged_cells.ranges if r.max_row <= header_offset]
    finally:
        wb.close()


def load_main_sheet(
    file_path: str | Path,
    header_scan_rows: int = HEADER_SCAN_ROWS,
    min_nonblank_header_cells: int = MIN_NONBLANK_HEADER_CELLS,
) -> SheetData:
    """Load the first sheet (sheet index 0) of an xlsx file, dynamically
    locating the header row and the PRODUCT DETAILS column. Rows that are
    entirely blank are skipped.

    `min_nonblank_header_cells` is exposed (rather than hardcoded) so tests
    can exercise the header-detection logic with smaller synthetic sheets
    than the real 45-column files; production callers should leave it at
    the default.
    """
    file_path = Path(file_path)
    wb = load_workbook(file_path, read_only=True, data_only=True)
    try:
        ws = wb.worksheets[0]
        scan_rows = list(
            ws.iter_rows(min_row=1, max_row=header_scan_rows, values_only=True)
        )
        header_offset, header = find_header_row(scan_rows, min_nonblank=min_nonblank_header_cells)
        product_details_column = find_product_details_column(header)
        pre_header_rows = scan_rows[:header_offset]

        data_start_row = header_offset + 2  # 1-indexed row after the header
        rows: list[dict[str, Any]] = []
        for values in ws.iter_rows(min_row=data_start_row, values_only=True):
            if all(v is None for v in values):
                continue
            row = {header[i]: values[i] if i < len(values) else None for i in range(len(header)) if header[i]}
            rows.append(row)
    finally:
        wb.close()

    merges = _read_pre_header_merges(file_path, header_offset)

    return SheetData(
        file_path=file_path,
        header=[h for h in header if h],
        product_details_column=product_details_column,
        rows=rows,
        pre_header_rows=pre_header_rows,
        merges=merges,
    )
