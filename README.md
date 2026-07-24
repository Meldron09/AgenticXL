# AgenticXL

A multi-agent pipeline that filters, consolidates, and enriches Excel trade-data exports.
Given a directory of raw import/export shipment workbooks, it keeps only the rows describing
products you care about, merges them into one canonical workbook, fills in additional columns,
and writes an auditable report of every decision it made.

The pipeline runs in three stages:

1. **Filter** — every row's `PRODUCT DETAILS` text is judged by an LLM for *semantic*
   relevance to your keyword list (synonyms, sub-components, and variants count — not just
   substring matches). One output file per input file, plus a per-row rationale log.
2. **Consolidate** — pure Python, no LLM. Filtered files are reconciled onto a canonical
   45-column schema by normalized column name and merged into one workbook, with a
   `Source File` provenance column.
3. **Enrich** — for each extra column you configure, values are derived from data already in
   the row where possible, falling back to a Serper web search only for rows that can't be
   resolved locally. Every enriched cell gets a companion `<column> Source` column recording
   whether the value came from `local`, `web_search`, or was left `unresolved`.

## Requirements

Python 3.11 or newer.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"     # or: pip install -r requirements.txt

cp .env.example .env        # then fill in the keys you need
```

## Input data

Drop your raw `.xlsx` exports into `Documents/` (or point `input_dir` in the run config
somewhere else). Every `.xlsx` in that directory is processed.

`Documents/` is **not** tracked in git — it holds third-party commercial trade records, so
you supply your own files.

Each workbook needs a `PRODUCT DETAILS` column (matched case-insensitively) on its header
row. The header row is located by content rather than a fixed offset, so merged "group
banner" rows above it are handled automatically and don't need stripping first. Only the
first sheet of each workbook is read.

## Configuration

`config/run_config.yaml` controls a run:

| Key | Meaning |
| --- | --- |
| `input_dir` | Directory of raw `.xlsx` files. Default `Documents`. |
| `keywords` | The product keywords rows are judged against in Stage 1. |
| `enrichment_columns` | Column names to add in Stage 3. Any name works — the enrichment logic assumes nothing about their meaning. |
| `relevance_batch_size` | Rows per Stage 1 LLM call. Default `50`. |
| `relevance_concurrency` | Relevance batches in flight at once, across the **whole run** rather than per file. Default `10`. |
| `enrichment_batch_size` | Rows per Stage 3 local-derivation call. Default `25`. |
| `filtered_dir`, `consolidated_dir`, `output_dir`, `logs_dir` | Output locations, relative to the project root unless absolute. |

Environment variables (see `.env.example`):

| Variable | Purpose |
| --- | --- |
| `AGENTICXL_MODEL_BACKEND` | `claude` (default), `ollama`, or `deepseek`. |
| `ANTHROPIC_API_KEY` | Required for the `claude` backend. |
| `AGENTICXL_CLAUDE_MODEL` | Optional Claude model id override. |
| `OLLAMA_MODEL`, `OLLAMA_BASE_URL` | Required / optional for the `ollama` backend. |
| `DEEPSEEK_API_KEY` | Required for the `deepseek` backend. |
| `AGENTICXL_DEEPSEEK_MODEL` | Optional DeepSeek model id override. |
| `SERPER_API_KEY` | Required only for the Stage 3 web-search fallback. Without it, rows that can't be derived locally are marked `unresolved` rather than guessed. |
| `AGENTICXL_SKIP_ENRICHMENT` | Set truthy to stop after Stage 2. |

A single `AGENTICXL_MODEL_BACKEND` value applies uniformly to the orchestrator, every
subagent, and every direct model call in a run — no stage uses a different backend. If the
selected backend is missing its required configuration, the run fails immediately with a
clear error instead of falling back to another one.

## Usage

```bash
agenticxl run                                  # full pipeline
agenticxl run --skip-enrichment                # stop after Stage 2
agenticxl run --config path/to/run_config.yaml
agenticxl run --project-root . --env-file .env
```

`--skip-enrichment` is a real skip: no enrichment agent is built, no Serper client is
constructed, and no duplicate output file is written. The consolidated workbook is the final
deliverable for that run.

## Outputs

| Path | Contents |
| --- | --- |
| `data/filtered/filtered_<name>.xlsx` | Stage 1 — relevant rows from one input file. |
| `data/consolidated/consolidated.xlsx` | Stage 2 — all filtered rows on the canonical schema. |
| `data/output/final_enriched.xlsx` | Stage 3 — the consolidated workbook plus enrichment columns. |
| `logs/<name>_rationale.jsonl` | One JSON line per row: the keep/drop verdict and why. |
| `logs/run_report.json`, `logs/run_report.md` | Per-file row counts, errors, consolidation warnings, and the enrichment source breakdown. |

A failure on one input file is captured as an error entry in the report rather than aborting
the run, so the remaining files still complete.

None of these outputs are tracked in git — they're generated, and they derive from your input
data.

## Tests

```bash
pytest
```

The suite needs no network access and no API key: model calls go through a deterministic
stub and Excel fixtures are generated on the fly (`tests/helpers.py`).

## Data handling

Input workbooks, every generated output, and all run logs are gitignored. The rationale logs
in particular contain product-description text copied from your source data, so treat them
with the same care as the inputs themselves.

## Architecture

See [CLAUDE.md](CLAUDE.md) for the internals — how fan-out is deliberately kept in Python
rather than LLM-decided, how structured results cross the agent boundary, the concurrency
model, and the Excel-parsing edge cases.
