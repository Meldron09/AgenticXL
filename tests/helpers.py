from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook

from agenticxl.models.base import LLMClient


def write_synthetic_workbook(
    path: Path,
    pre_header_rows: list[list],
    header: list[str],
    rows: list[list],
    merges: list[str] | None = None,
) -> Path:
    wb = Workbook()
    ws = wb.active
    for r in pre_header_rows:
        ws.append(r)
    if merges:
        for m in merges:
            ws.merge_cells(m)
    ws.append(header)
    for row in rows:
        ws.append(row)
    wb.save(path)
    return path


class FakeLLMClient(LLMClient):
    """Deterministic LLMClient stub for tests: `responder` maps a prompt to
    the structured dict complete_structured should return, and `text` maps
    a prompt to the string complete() should return."""

    backend_name = "fake"

    def __init__(self, structured_responder=None, text_responder=None):
        self._structured_responder = structured_responder
        self._text_responder = text_responder

    @property
    def chat_model(self):
        raise NotImplementedError("FakeLLMClient has no real chat model")

    def complete(self, prompt: str) -> str:
        if self._text_responder is None:
            return "fake response"
        return self._text_responder(prompt)

    def complete_structured(self, prompt: str, schema: dict, max_retries: int = 2) -> dict:
        if self._structured_responder is None:
            raise NotImplementedError("no structured_responder configured")
        return self._structured_responder(prompt, schema)
