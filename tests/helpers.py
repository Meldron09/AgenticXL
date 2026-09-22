from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook

from agenticxl.filtering.laya_client import LayaVerdict
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


class FakeLayaClient:
    """Deterministic LayaClient stub for tests: `responder(rows, keywords, threshold)`
    returns the list[LayaVerdict] `judge_batch` should return. With no responder,
    every row comes back a non-match -- the same never-guess default LayaClient
    itself falls back to on a failed/malformed batch."""

    def __init__(self, responder=None):
        self._responder = responder

    @property
    def model_identifier(self) -> str:
        return "fake"

    def judge_batch(
        self, rows: list[tuple[int, str]], keywords: list[str], threshold: float = 0.5
    ) -> list[LayaVerdict]:
        if self._responder is None:
            return [
                LayaVerdict(row_id=row_id, match=False, score=0.0, confidence=0.0, model="fake")
                for row_id, _text in rows
            ]
        return self._responder(rows, keywords, threshold)
