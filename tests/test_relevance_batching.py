from __future__ import annotations

import re

from agenticxl.filtering.relevance import (
    dedupe_truncation_artifact,
    judge_all_rows,
    judge_batch,
    write_rationale_log,
)
from agenticxl.models.base import StructuredOutputError
from helpers import FakeLLMClient


def _ids_in_prompt(prompt: str) -> list[int]:
    return [int(m) for m in re.findall(r"Row (\d+):", prompt)]


def test_judge_all_rows_batches_and_matches_even_rows():
    def responder(prompt, schema):
        ids = _ids_in_prompt(prompt)
        return {"verdicts": [{"row_id": i, "match": i % 2 == 0, "rationale": "t"} for i in ids]}

    rows = [{"PRODUCT DETAILS": f"item {i}"} for i in range(10)]
    llm = FakeLLMClient(structured_responder=responder)

    verdicts = judge_all_rows(rows, "PRODUCT DETAILS", ["x"], llm, batch_size=4)

    assert len(verdicts) == 10
    assert [v.row_id for v in verdicts if v.match] == [0, 2, 4, 6, 8]


def test_judge_batch_splits_into_correct_batch_sizes():
    seen_batch_sizes = []

    def responder(prompt, schema):
        ids = _ids_in_prompt(prompt)
        seen_batch_sizes.append(len(ids))
        return {"verdicts": [{"row_id": i, "match": False, "rationale": "t"} for i in ids]}

    rows = [{"PRODUCT DETAILS": f"item {i}"} for i in range(9)]
    llm = FakeLLMClient(structured_responder=responder)

    judge_all_rows(rows, "PRODUCT DETAILS", ["x"], llm, batch_size=4)

    assert seen_batch_sizes == [4, 4, 1]


def test_malformed_batch_response_falls_back_to_judgment_failed():
    def responder(prompt, schema):
        raise StructuredOutputError("boom")

    rows = [{"PRODUCT DETAILS": "a"}, {"PRODUCT DETAILS": "b"}]
    llm = FakeLLMClient(structured_responder=responder)

    verdicts = judge_all_rows(rows, "PRODUCT DETAILS", ["x"], llm, batch_size=50)

    assert all(v.match is False for v in verdicts)
    assert all(v.rationale == "judgment_failed" for v in verdicts)


def test_partial_batch_response_fills_missing_rows_individually():
    def responder(prompt, schema):
        # only returns a verdict for row 0, row 1 and 2 are missing
        return {"verdicts": [{"row_id": 0, "match": True, "rationale": "ok"}]}

    llm = FakeLLMClient(structured_responder=responder)
    verdicts = judge_batch([(0, "a"), (1, "b"), (2, "c")], ["x"], llm)

    assert (verdicts[0].row_id, verdicts[0].match, verdicts[0].rationale) == (0, True, "ok")
    assert verdicts[1].match is False and verdicts[1].rationale == "judgment_failed"
    assert verdicts[2].match is False and verdicts[2].rationale == "judgment_failed"


def test_one_bad_batch_does_not_abort_remaining_batches():
    call_count = {"n": 0}

    def responder(prompt, schema):
        call_count["n"] += 1
        ids = _ids_in_prompt(prompt)
        if call_count["n"] == 1:
            raise StructuredOutputError("first batch broken")
        return {"verdicts": [{"row_id": i, "match": True, "rationale": "ok"} for i in ids]}

    rows = [{"PRODUCT DETAILS": f"item {i}"} for i in range(6)]
    llm = FakeLLMClient(structured_responder=responder)

    verdicts = judge_all_rows(rows, "PRODUCT DETAILS", ["x"], llm, batch_size=3)

    assert call_count["n"] == 2
    assert all(v.rationale == "judgment_failed" for v in verdicts[:3])
    assert all(v.match for v in verdicts[3:])


def test_dedupe_truncation_artifact():
    text = "BRAND APJ MODEL AP36J UPPER STRUCTURE ONLYBRAND APJ MODEL AP36J UPPER STRUCTURE ONLY"
    result = dedupe_truncation_artifact(text)
    assert result == "BRAND APJ MODEL AP36J UPPER STRUCTURE ONLY"


def test_dedupe_truncation_artifact_leaves_normal_text_alone():
    text = "SCISSOR LIFT 1623RD - MAKE: SINOBOOM"
    assert dedupe_truncation_artifact(text) == text


def test_duplicate_product_details_text_judged_once():
    seen_texts = []

    def responder(prompt, schema):
        seen_texts.extend(re.findall(r'Row \d+: "(.*?)"', prompt))
        ids = _ids_in_prompt(prompt)
        return {"verdicts": [{"row_id": i, "match": True, "rationale": "dup-check"} for i in ids]}

    rows = [
        {"PRODUCT DETAILS": "WIDGET A"},
        {"PRODUCT DETAILS": "WIDGET A"},
        {"PRODUCT DETAILS": "WIDGET B"},
        {"PRODUCT DETAILS": "WIDGET A"},
    ]
    llm = FakeLLMClient(structured_responder=responder)

    verdicts = judge_all_rows(rows, "PRODUCT DETAILS", ["x"], llm, batch_size=50)

    assert len(verdicts) == 4  # still one verdict per original row
    assert all(v.match for v in verdicts)
    # only 2 unique texts should ever have reached the LLM, not 4
    assert seen_texts.count("WIDGET A") == 1
    assert seen_texts.count("WIDGET B") == 1


def test_duplicate_rows_share_identical_verdict_and_rationale():
    def responder(prompt, schema):
        ids = _ids_in_prompt(prompt)
        return {"verdicts": [{"row_id": i, "match": True, "rationale": f"judged-{i}"} for i in ids]}

    rows = [{"PRODUCT DETAILS": "WIDGET A"} for _ in range(3)]
    llm = FakeLLMClient(structured_responder=responder)

    verdicts = judge_all_rows(rows, "PRODUCT DETAILS", ["x"], llm, batch_size=50)

    assert [v.row_id for v in verdicts] == [0, 1, 2]
    assert len({v.rationale for v in verdicts}) == 1  # one judgment, shared across all 3 rows


def test_concurrency_runs_batches_in_parallel():
    import time

    def responder(prompt, schema):
        time.sleep(0.2)
        ids = _ids_in_prompt(prompt)
        return {"verdicts": [{"row_id": i, "match": False, "rationale": "t"} for i in ids]}

    # 6 unique rows, batch_size=1 -> 6 batches, each taking 0.2s
    rows = [{"PRODUCT DETAILS": f"item {i}"} for i in range(6)]
    llm = FakeLLMClient(structured_responder=responder)

    start = time.monotonic()
    verdicts = judge_all_rows(rows, "PRODUCT DETAILS", ["x"], llm, batch_size=1, max_concurrency=6)
    elapsed = time.monotonic() - start

    assert len(verdicts) == 6
    # sequential would take ~1.2s; 6-way concurrent should be close to ~0.2s
    assert elapsed < 0.8, f"expected concurrent batches to run in parallel, took {elapsed:.2f}s"


def test_concurrency_default_is_sequential():
    from agenticxl.filtering.relevance import DEFAULT_RELEVANCE_CONCURRENCY

    assert DEFAULT_RELEVANCE_CONCURRENCY == 1


def test_write_rationale_log(tmp_path):
    from agenticxl.filtering.relevance import RelevanceVerdict

    verdicts = [
        RelevanceVerdict(row_id=0, match=True, rationale="matches keyword"),
        RelevanceVerdict(row_id=1, match=False, rationale="no match"),
    ]
    path = write_rationale_log(tmp_path / "rationale.jsonl", "source.xlsx", verdicts)

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    import json

    record = json.loads(lines[0])
    assert record["source_file"] == "source.xlsx"
    assert record["row_id"] == 0
    assert record["match"] is True
