"""Structured response schemas for deepagents subagents.

Each subagent is given a `response_format` so its single final message back
to the orchestrator (the `task` tool's ToolMessage content) is guaranteed
structured JSON, rather than relying on the orchestrator LLM to faithfully
paraphrase a subagent's free-text summary.
"""

from __future__ import annotations

from pydantic import BaseModel


class FileResult(BaseModel):
    source_file: str
    status: str  # "ok" | "error"
    error: str | None = None
    rows_in: int = 0
    rows_kept: int = 0
    output_path: str | None = None
    rationale_log_path: str | None = None


class EnrichmentSummary(BaseModel):
    status: str  # "ok" | "error"
    error: str | None = None
    output_path: str | None = None
    rows: int = 0
    source_counts: dict[str, int] = {}
