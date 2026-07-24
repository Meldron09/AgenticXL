from __future__ import annotations

from datetime import date

from agenticxl.excel.schema import (
    PROVENANCE_COLUMN,
    normalize_cell_value,
    normalize_column_name,
    reconcile_to_canonical,
)

CANONICAL = ["NO", "ARRIVAL DATE", "PRODUCT DETAILS", "QUANTITY"]


def test_reconcile_identical_schema_adds_provenance():
    header = ["NO", "ARRIVAL DATE", "PRODUCT DETAILS", "QUANTITY"]
    rows = [{"NO": 1, "ARRIVAL DATE": "2024-01-01", "PRODUCT DETAILS": "widget", "QUANTITY": 5}]

    result = reconcile_to_canonical(header, rows, source_file="a.xlsx", canonical=CANONICAL)

    assert result.header == CANONICAL + [PROVENANCE_COLUMN]
    assert result.rows[0][PROVENANCE_COLUMN] == "a.xlsx"
    assert result.warnings == []


def test_reconcile_case_and_whitespace_insensitive_matching():
    header = [" no ", "Arrival Date", "product details", "quantity"]
    rows = [{" no ": 1, "Arrival Date": "2024-01-01", "product details": "widget", "quantity": 5}]

    result = reconcile_to_canonical(header, rows, source_file="b.xlsx", canonical=CANONICAL)

    assert result.rows[0]["PRODUCT DETAILS"] == "widget"
    assert result.rows[0]["NO"] == 1


def test_reconcile_extra_column_is_appended_with_warning():
    header = ["NO", "ARRIVAL DATE", "PRODUCT DETAILS", "QUANTITY", "VENDOR NOTES"]
    rows = [
        {
            "NO": 1,
            "ARRIVAL DATE": "2024-01-01",
            "PRODUCT DETAILS": "widget",
            "QUANTITY": 5,
            "VENDOR NOTES": "rush order",
        }
    ]

    result = reconcile_to_canonical(header, rows, source_file="c.xlsx", canonical=CANONICAL)

    assert "VENDOR NOTES" in result.header
    assert result.rows[0]["VENDOR NOTES"] == "rush order"
    assert any("VENDOR NOTES" in w for w in result.warnings)


def test_reconcile_missing_canonical_column_is_blank():
    header = ["NO", "PRODUCT DETAILS", "QUANTITY"]  # missing ARRIVAL DATE
    rows = [{"NO": 1, "PRODUCT DETAILS": "widget", "QUANTITY": 5}]

    result = reconcile_to_canonical(header, rows, source_file="d.xlsx", canonical=CANONICAL)

    assert result.rows[0]["ARRIVAL DATE"] is None


def test_reconcile_across_two_files_with_different_extra_columns():
    header_a = CANONICAL + ["FOO"]
    rows_a = [{"NO": 1, "ARRIVAL DATE": "x", "PRODUCT DETAILS": "p1", "QUANTITY": 1, "FOO": "bar"}]
    header_b = CANONICAL + ["BAZ"]
    rows_b = [{"NO": 2, "ARRIVAL DATE": "y", "PRODUCT DETAILS": "p2", "QUANTITY": 2, "BAZ": "qux"}]

    result_a = reconcile_to_canonical(header_a, rows_a, source_file="a.xlsx", canonical=CANONICAL)
    result_b = reconcile_to_canonical(header_b, rows_b, source_file="b.xlsx", canonical=CANONICAL)

    # each file's own reconciliation only knows about its own extra column --
    # a consolidation step unions these before writing (see consolidate.py)
    assert result_a.header == CANONICAL + ["FOO", PROVENANCE_COLUMN]
    assert result_b.header == CANONICAL + ["BAZ", PROVENANCE_COLUMN]
    assert "BAZ" not in result_a.rows[0]
    assert "FOO" not in result_b.rows[0]


def test_normalize_cell_value_placeholder_and_date():
    assert normalize_cell_value("IMPORTER ID", ".") is None
    assert normalize_cell_value("IMPORTER ID", "ABC123") == "ABC123"
    # Excel serial 45658 == 2025-01-01
    assert normalize_cell_value("ARRIVAL DATE", 45658) == date(2025, 1, 1)


def test_normalize_column_name_collapses_whitespace_and_case():
    assert normalize_column_name("  product   details ") == "PRODUCT DETAILS"
