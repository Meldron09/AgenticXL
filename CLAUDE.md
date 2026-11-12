# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A multi-agent (deepagents/LangChain) pipeline that filters, consolidates, and enriches Excel
trade-data exports. Input is a directory of raw `.xlsx` import/export shipment records
(`Documents/`); output is one enriched workbook plus an auditable run report.

`README.md` covers setup, config, and usage. Beyond that, the docstrings at the top of each
module are the primary design documentation; they frequently reference "the implementation
plan", a document that is **not** in the repo — treat those docstrings themselves as the
rationale of record.

Input data (`Documents/`), every generated output (`data/*/`), and all run logs (`logs/`) are
gitignored: the inputs are third-party commercial trade records and the rest derives from them.
Don't add fixtures under those paths — tests generate their own workbooks.

## Commands

```bash
# Env (already created; project is installed editable into it)
.venv/bin/pip install -e ".[dev]"                  # deps live in pyproject; requirements.txt is `-e .[dev]`
.venv/bin/python -m pytest -q                      # all tests (205, ~2s, no network/LLM needed)
.venv/bin/python -m pytest tests/test_excel_reader.py::test_two_row_header -q   # single test

.venv/bin/agenticxl run                            # full pipeline, config/run_config.yaml
.venv/bin/agenticxl run --skip-enrichment           # stop after Stage 2
.venv/bin/agenticxl run --config path/to.yaml --project-root . --env-file .env
```

No linter or formatter is configured.

## Architecture

Three stages, driven by `pipeline.py:run`. Config comes from `config/run_config.yaml` +
env vars, merged into one `RunConfig` (`config.py`) that is passed explicitly down the
call chain — there is no global config state, which is what lets tests build a `RunConfig`
directly without touching real env or files.

**Stage 1 — per-file filtering** (`agents/file_subagent.py`, `filtering/relevance.py`)
One orchestrator invocation per input `.xlsx`, fanned out with `asyncio.gather`. Each row's
`PRODUCT DETAILS` text is judged for *semantic* relevance to the configured keywords by the
LLM — deliberately no deterministic substring pre-filter (measured to miss ~31% of true
matches on this dataset). Writes `data/filtered/filtered_<stem>.xlsx` plus a per-row
`logs/<stem>_rationale.jsonl`.

**Stage 2 — consolidation** (`consolidation/consolidate.py`, `excel/schema.py`)
Pure Python, no LLM. Reconciles each filtered file's header onto the 45-column
`CANONICAL_COLUMNS` list by *normalized name* (not position), appends a `Source File`
provenance column, and writes `data/consolidated/consolidated.xlsx`. Unmapped columns are
appended with a warning; missing canonical columns are blank — schema drift degrades, never
crashes.

**Stage 3 — enrichment** (`agents/enrichment_agent.py`, `enrichment/`)
For each configured enrichment column: batched local derivation from the row's own data
(`derive_local.py`) first; only rows still unresolved fall through to Serper web search
(`web_search.py`) + snippet synthesis (`synthesize.py`). Writes
`data/output/final_enriched.xlsx` with a `<column> Source` companion column per enrichment
column recording `local` / `web_search` / `unresolved`.

Finally `reporting/report.py` writes `logs/run_report.{json,md}`.

### Agent-layer conventions

- **Fan-out is Python, not LLM-decided.** The orchestrator (`agents/orchestrator.py`) is
  deliberately thin: per invocation its only job is to `task`-call the one named subagent for
  the one named file. Looping over files and aggregating results is deterministic control flow
  in `pipeline.py`, so a file's outcome never depends on an LLM correctly iterating. Keep it
  that way when adding stages.
- **Results cross the agent boundary as structured JSON.** Subagents are given
  `response_format=ToolStrategy(...)` (`agents/schemas.py`), and `extract_last_tool_result`
  parses the last `ToolMessage` rather than trusting the orchestrator's free-text summary.
  `ToolStrategy` is forced explicitly instead of `AutoStrategy` because some backends
  (DeepSeek) reject the provider-native `response_format` mode.
- **Real logic lives in `_impl` functions.** `agents/tools.py` wraps `process_file_impl` /
  `enrich_columns_impl` in `@tool` shims; the impls are plain functions so they're unit
  testable without a live agent loop. Put new logic in the impl, not the tool.
- **xlsx is binary**, so these tools do their own openpyxl I/O and never use deepagents'
  text-oriented `read_file`/`write_file`.
- **Failures are isolated, never fabricated.** Tools return `status: "error"` dicts instead of
  raising; a malformed LLM batch response yields per-row `judgment_failed` / `derivation_failed`
  fallbacks; inconclusive search snippets yield `unresolved`, never a guessed value. Preserve
  this — several tests assert it directly.

### Model backends

One env var, `AGENTICXL_MODEL_BACKEND` (`claude` | `ollama` | `deepseek`), selects the model
used *everywhere* in a run — orchestrator, every subagent, and every direct structured call.
Everything goes through `models/get_llm_client` → an `LLMClient` (`models/base.py`), which
exposes `chat_model` for deepagents and `complete`/`complete_structured` for tools. Never
hardcode a backend in a stage. `validate_backend_ready()` fails fast on missing keys rather
than silently falling back.

Note on DeepSeek: v4 thinking mode is hard-disabled in `deepseek_client.py` because it is
incompatible with the forced `tool_choice` that `with_structured_output` needs.

### Excel reading gotchas

`excel/reader.py` detects the header row by content (first row within the first 5 with ≥20
non-blank cells), not a fixed skip count — every sample file has a merged group-banner row
above the real header, and Genie has an extra stray row above that. `load_main_sheet` also
captures `pre_header_rows` + `merges` so filtered output can be structurally identical to its
input. `min_nonblank_header_cells` is a parameter purely so tests can use small synthetic
sheets; leave it at the default in production paths.

`dedupe_truncation_artifact` (`filtering/relevance.py`) strips a self-concatenation artifact
in `PRODUCT DETAILS` **for prompt text only** — never apply it to data written to output files.

### Concurrency

Two independent knobs, both in `run_config.yaml`. `relevance_concurrency` caps relevance
batches via a `ThreadPoolExecutor` shared **globally across the whole run** (module-level
singleton in `filtering/relevance.py`), specifically so Stage 1's per-file `asyncio.gather`
doesn't multiply into unbounded simultaneous LLM calls. `judge_all_rows` defaults
`max_concurrency=1` so existing tests stay sequential — production passes the config value.
Rows with byte-identical `PRODUCT DETAILS` are judged once and share the verdict (lossless).

## Config and secrets

`config/run_config.yaml` holds the keyword list (`keywords`, referred to as `l` in the code
comments) and enrichment column list (`enrichment_columns`, `c`), batch sizes, concurrency, and
output paths. Enrichment logic is generic — it assumes nothing about what a column name means.
The committed keyword/column lists are live aerial-work-platform values, and they are the two
settings meant to be edited per job — retargeting the run at a different product category
should never require a code change.

`.env` (see `.env.example`) holds `ANTHROPIC_API_KEY` / `OLLAMA_MODEL` / `DEEPSEEK_API_KEY` and
`SERPER_API_KEY` (required only for the Stage 3 web-search fallback).

## Testing

Tests never hit a network or a real model. `tests/helpers.py` provides `FakeLLMClient` (map a
prompt → the structured dict or text to return) and `write_synthetic_workbook`. Tests import it
as `from helpers import FakeLLMClient`, relying on pytest's rootdir-relative path insertion —
there is no `conftest.py`, and `pythonpath = ["src"]` is set in `pyproject.toml`.

## Agent skills

### Issue tracker

Issues live as GitHub Issues in `Meldron09/AgenticXL`, managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five canonical roles, used as-is (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
