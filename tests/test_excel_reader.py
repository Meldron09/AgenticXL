from __future__ import annotations

import pytest
from openpyxl import Workbook

from agenticxl.excel.reader import (
    HeaderNotFoundError,
    ProductDetailsColumnNotFoundError,
    load_main_sheet,
)
from helpers import write_synthetic_workbook

HEADER = ["NO", "ARRIVAL DATE", "PRODUCT DETAILS", "QUANTITY", "BRAND NAME"]
SMALL_THRESHOLD = 3  # HEADER has 5 columns; real files use the default (20) against 45 columns


def test_two_row_header(tmp_path):
    path = write_synthetic_workbook(
        tmp_path / "two_row.xlsx",
        pre_header_rows=[["BANNER", None, None, None, None]],
        header=HEADER,
        rows=[[1, "2024-01-01", "widget A", 5, "Acme"], [2, "2024-01-02", "widget B", 3, "Acme"]],
    )
    sd = load_main_sheet(path, min_nonblank_header_cells=SMALL_THRESHOLD)
    assert sd.product_details_column == "PRODUCT DETAILS"
    assert len(sd.rows) == 2
    assert sd.rows[0]["PRODUCT DETAILS"] == "widget A"
    assert sd.rows[1]["BRAND NAME"] == "Acme"


def test_genie_style_three_row_header(tmp_path):
    """Genie's real file has: row1 = single stray leftover cell, row2 =
    group banner, row3 = the real header -- one row later than every other
    file. Header detection must not assume a fixed row offset."""
    path = tmp_path / "genie.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.append([None, None, "PRODUCT DETAILS", None, None])  # stray leftover cell
    ws.append(["BANNER", None, None, None, None])  # group banner
    ws.append(HEADER)  # real header
    ws.append([1, "2024-01-01", "widget A", 5, "Acme"])
    wb.save(path)

    sd = load_main_sheet(path, min_nonblank_header_cells=SMALL_THRESHOLD)
    assert sd.product_details_column == "PRODUCT DETAILS"
    assert len(sd.rows) == 1
    assert sd.rows[0]["PRODUCT DETAILS"] == "widget A"


def test_missing_product_details_column_raises(tmp_path):
    bad_header = ["NO", "ARRIVAL DATE", "DESCRIPTION", "QUANTITY", "BRAND NAME"]
    path = write_synthetic_workbook(
        tmp_path / "bad.xlsx",
        pre_header_rows=[["BANNER", None, None, None, None]],
        header=bad_header,
        rows=[[1, "2024-01-01", "x", 2, "Acme"]],
    )
    with pytest.raises(ProductDetailsColumnNotFoundError):
        load_main_sheet(path, min_nonblank_header_cells=SMALL_THRESHOLD)


def test_no_header_row_found_raises(tmp_path):
    """No row within the scanned range has enough non-blank cells --
    e.g. a file that's just a banner with no real header at all."""
    path = tmp_path / "no_header.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.append(["BANNER"])
    ws.append(["ALSO SPARSE"])
    wb.save(path)

    with pytest.raises(HeaderNotFoundError):
        load_main_sheet(path, min_nonblank_header_cells=SMALL_THRESHOLD)


def test_merged_header_banner_cells_are_skipped(tmp_path):
    """Merged banner cells (only the top-left cell of a merge holds a
    value) should read as mostly-blank and be skipped in favor of the real
    header row below."""
    path = tmp_path / "merged.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.append(["BANNER GROUP 1", None, "BANNER GROUP 2", None, None])
    ws.merge_cells("A1:B1")
    ws.merge_cells("C1:E1")
    ws.append(HEADER)
    ws.append([1, "2024-01-01", "widget A", 5, "Acme"])
    wb.save(path)

    sd = load_main_sheet(path, min_nonblank_header_cells=SMALL_THRESHOLD)
    assert sd.header == HEADER
    assert sd.product_details_column == "PRODUCT DETAILS"


def test_fully_blank_rows_are_skipped(tmp_path):
    path = write_synthetic_workbook(
        tmp_path / "blank_rows.xlsx",
        pre_header_rows=[["BANNER", None, None, None, None]],
        header=HEADER,
        rows=[
            [1, "2024-01-01", "widget A", 5, "Acme"],
            [None, None, None, None, None],
            [2, "2024-01-02", "widget B", 3, "Acme"],
        ],
    )
    sd = load_main_sheet(path, min_nonblank_header_cells=SMALL_THRESHOLD)
    assert len(sd.rows) == 2
