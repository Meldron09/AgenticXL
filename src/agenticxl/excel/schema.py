"""Canonical schema for consolidation.

The 45-column layout below was confirmed identical, in this exact order,
across all 14 sample files (see the implementation plan). Reconciliation is
still name-based (not positional) and tolerant of drift, so a future file
with extra/missing/reordered columns degrades gracefully instead of
crashing: unmapped columns are appended with a warning, missing canonical
columns are filled blank.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

CANONICAL_COLUMNS: list[str] = [
    "NO",
    "ARRIVAL DATE",
    "IMPORTER NAME",
    "IMPORTER ID",
    "IMPORTER ADDRESS",
    "IMPORTER COUNTRY",
    "EXPORTER NAME",
    "EXPORTER ID",
    "EXPORTER ADDRESS",
    "EXPORTER COUNTRY",
    "COUNTRY OF ORIGIN",
    "HS CODE",
    "HS CODE DESCRIPTION",
    "PRODUCT DETAILS",
    "FOB VALUE",
    "FOB CURRENCY",
    "USD FOB",
    "CIF VALUE",
    "CIF CURRENCY",
    "USD CIF",
    "UNIT PRICE",
    "GROSS WEIGHT",
    "GROSS WEIGHT UNIT",
    "NET WEIGHT",
    "NET WEIGHT UNIT",
    "QUANTITY",
    "QUANTITY UNIT",
    "NUMBER OF PACKAGES",
    "PACKAGES UNIT",
    "MANUFACTURING COMPANY",
    "BRAND NAME",
    "INCOTERMS",
    "PAYMENT TYPE",
    "REGIME",
    "TRANSPORT TYPE",
    "NOTIFY PARTY",
    "NOTIFY ADDRESS",
    "PORT OF ARRIVAL",
    "PORT OF DEPARTURE",
    "ITEM NO",
    "DECLARATION NUMBER",
    "IMPORT DATA TYPE",
    "DATA SOURCE",
    "TRANSPORT COMPANY",
    "VESSEL NAME",
]

PROVENANCE_COLUMN = "Source File"

# Placeholder values observed in IMPORTER ID / EXPORTER ID (and possibly
# elsewhere) that stand in for a genuinely blank value.
NULL_PLACEHOLDERS = {"."}

_EXCEL_EPOCH = date(1899, 12, 30)


def normalize_column_name(name: str) -> str:
    return " ".join(name.strip().upper().split())


def normalize_cell_value(column: str, value: Any) -> Any:
    """Normalize a single cell value: strip null-placeholder strings, and
    convert an Excel date serial number to a date object for date columns."""
    if isinstance(value, str) and value.strip() in NULL_PLACEHOLDERS:
        return None
    if normalize_column_name(column) == "ARRIVAL DATE" and isinstance(value, (int, float)):
        return _EXCEL_EPOCH + timedelta(days=int(value))
    if isinstance(value, datetime):
        return value.date()
    return value


@dataclass
class ReconciliationResult:
    header: list[str]
    rows: list[dict[str, Any]]
    warnings: list[str]


def reconcile_to_canonical(
    header: list[str],
    rows: list[dict[str, Any]],
    source_file: str,
    canonical: list[str] | None = None,
) -> ReconciliationResult:
    """Map a file's own header onto the canonical column list by normalized
    name, append a provenance column, and normalize known placeholder/date
    values. Columns present in `header` but not in `canonical` are appended
    at the end (schema-drift tolerance) with a warning; canonical columns
    absent from `header` are filled blank."""
    canonical = canonical if canonical is not None else CANONICAL_COLUMNS
    canonical_normalized = {normalize_column_name(c): c for c in canonical}

    source_by_normalized: dict[str, str] = {}
    extra_columns: list[str] = []
    for original in header:
        norm = normalize_column_name(original)
        if norm in canonical_normalized:
            source_by_normalized[norm] = original
        else:
            extra_columns.append(original)

    warnings: list[str] = []
    if extra_columns:
        warnings.append(
            f"{source_file}: unmapped columns appended to canonical schema: {extra_columns}"
        )

    final_header = list(canonical) + extra_columns + [PROVENANCE_COLUMN]

    reconciled_rows: list[dict[str, Any]] = []
    for row in rows:
        new_row: dict[str, Any] = {}
        for col in canonical:
            norm = normalize_column_name(col)
            original_key = source_by_normalized.get(norm)
            value = row.get(original_key) if original_key else None
            new_row[col] = normalize_cell_value(col, value)
        for col in extra_columns:
            new_row[col] = row.get(col)
        new_row[PROVENANCE_COLUMN] = source_file
        reconciled_rows.append(new_row)

    return ReconciliationResult(header=final_header, rows=reconciled_rows, warnings=warnings)
