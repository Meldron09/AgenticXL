"""Bake-off of zero-shot System 1 models as a replacement Baseline judge
(GitHub issues #10, #12).

Scores every Ground-truth set row with each candidate `Scorer`
(`scripts/bakeoff_scorers.py`), applies a FIXED 0.5 cut -- no tuning, no
calibration, no threshold sweep anywhere -- and reports accuracy / precision /
recall / F1 raw and population-reweighted (reusing `compute_classifier_metrics`
and `population_weighted_metrics` from `build_ground_truth_report.py`), AUROC
as an information-only column, and a GO / NO-GO verdict against the GO gate
(>= 95% population-reweighted accuracy AND >= 98% population-reweighted
recall). AUROC never feeds the verdict.

Input text is each sample row's `product_details_text`: the PRODUCT DETAILS
the Baseline judge saw, already passed through `dedupe_truncation_artifact`
when the sample was built. Output workbooks are never touched. Keywords come
from the run config (`keywords`), so retargeting needs no code change.

Layout (so parallel tickets touch different functions):
  * scoring     -- `score_texts` (per-row scoring: dedup / bad-score handling, #13)
  * driver      -- `evaluate_candidate`, `run_bakeoff` (per-candidate flow, #13)
  * metrics     -- `auroc`, `determine_verdict`
  * report      -- `build_report` (JSON dict), `_SECTIONS` + `render_markdown`
  * CLI         -- `main`

Usage:
    .venv/bin/pip install -e ".[bakeoff]"
    .venv/bin/python scripts/run_system1_bakeoff.py
    .venv/bin/python scripts/run_system1_bakeoff.py --candidates qwen3-reranker-0.6b
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agenticxl.config import RunConfig
from agenticxl.excel.reader import load_main_sheet
from agenticxl.filtering.relevance import dedupe_truncation_artifact

from scripts.bakeoff_scorers import CANDIDATES, Scorer, build_scorers
from scripts.build_ground_truth_report import (
    DEFAULT_LAYA_THRESHOLD,
    ClassifierMetrics,
    Quadrant,
    SampleRow,
    compute_classifier_metrics,
    compute_quadrant_pool_sizes,
    load_ground_truth_labels,
    load_ground_truth_sample,
    load_rationale_file,
    population_weighted_metrics,
)

# Fixed before any run. Do not tune, sweep, or calibrate (see CONTEXT.md: GO gate).
THRESHOLD = 0.5
GO_MIN_POPULATION_ACCURACY = 0.95
GO_MIN_RECALL = 0.98  # population-reweighted, like accuracy

Labels = dict[int, tuple[bool, str]]


@dataclass
class CandidateResult:
    name: str
    model_id: str
    revision: str
    template: str
    gate_eligible: bool
    status: str  # "ok" | "could_not_run"
    seconds: float
    raw: ClassifierMetrics
    population_weighted: ClassifierMetrics
    auroc: float | None
    verdict: str  # "GO" | "NO-GO" | "AUROC-only"
    scores: dict[int, float]  # sample_index -> probability (kept for #14 miss lists; not in the JSON)
    failed_rows: list[int] = field(default_factory=list)  # sample_index of rows with a non-finite / out-of-range score (#13)
    error: str | None = None  # reason, when status == "could_not_run" (#13)


# --- scoring -----------------------------------------------------------------


def score_texts(scorer: Scorer, texts: list[str]) -> list[float]:
    """One probability per text, in order. Byte-identical texts are scored
    once and share the score. A score that is not a finite number in [0, 1]
    comes back as NaN (a per-row failure); a wrong score count raises."""
    unique = list(dict.fromkeys(texts))
    scores = scorer.score(unique)
    if len(scores) != len(unique):
        raise ValueError(f"{scorer.name} returned {len(scores)} scores for {len(unique)} texts")
    by_text = {t: _valid_or_nan(s) for t, s in zip(unique, scores)}
    return [by_text[t] for t in texts]


def _valid_or_nan(score) -> float:
    try:
        score = float(score)
    except (TypeError, ValueError):
        return math.nan
    return score if 0.0 <= score <= 1.0 else math.nan  # NaN fails both comparisons; so does out of range


# --- metrics -----------------------------------------------------------------


def auroc(truth: list[bool], scores: list[float]) -> float | None:
    """Threshold-free separability (rank-sum / Mann-Whitney, ties count half).
    None when only one class is present. Information only."""
    pos = [s for t, s in zip(truth, scores) if t]
    neg = [s for t, s in zip(truth, scores) if not t]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def determine_verdict(population_accuracy: float, population_recall: float, gate_eligible: bool) -> str:
    """GO gate at the fixed cut. AUROC is deliberately not an input."""
    if gate_eligible is not True:  # strict: anything but literal True can never pass
        return "AUROC-only"
    if population_accuracy >= GO_MIN_POPULATION_ACCURACY and population_recall >= GO_MIN_RECALL:
        return "GO"
    return "NO-GO"


# --- driver ------------------------------------------------------------------


def evaluate_candidate(
    scorer: Scorer,
    sample_rows: list[SampleRow],
    labels: Labels,
    quadrant_pool_sizes: dict[Quadrant, int],
) -> CandidateResult:
    start = time.perf_counter()
    scores = score_texts(scorer, [r.product_details_text for r in sample_rows])
    seconds = time.perf_counter() - start

    truth = {r.sample_index: labels[r.sample_index][0] for r in sample_rows}
    all_scores = {r.sample_index: s for r, s in zip(sample_rows, scores)}
    failed = [i for i, s in all_scores.items() if math.isnan(s)]
    by_index = {i: s for i, s in all_scores.items() if i not in failed}
    # A failed row is still scored in the metrics, always counted wrong.
    predictions = {i: (not truth[i]) if i in failed else s >= THRESHOLD for i, s in all_scores.items()}

    raw = compute_classifier_metrics(truth, predictions)
    weighted = population_weighted_metrics(sample_rows, labels, quadrant_pool_sizes, predictions)
    return CandidateResult(
        name=scorer.name,
        model_id=scorer.model_id,
        revision=scorer.revision,
        template=scorer.template,
        gate_eligible=scorer.gate_eligible,
        status="ok",
        seconds=seconds,
        raw=raw,
        population_weighted=weighted,
        auroc=auroc([truth[i] for i in by_index], list(by_index.values())),
        verdict=determine_verdict(weighted.accuracy, weighted.recall, scorer.gate_eligible),
        scores=by_index,
        failed_rows=failed,
    )


def _could_not_run(scorer: Scorer, seconds: float, error: str) -> CandidateResult:
    zero = ClassifierMetrics(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)  # placeholder; the report shows n/a
    return CandidateResult(
        name=scorer.name, model_id=scorer.model_id, revision=scorer.revision, template=scorer.template,
        gate_eligible=scorer.gate_eligible, status="could_not_run", seconds=seconds, raw=zero,
        population_weighted=zero, auroc=None, verdict="COULD-NOT-RUN", scores={}, error=error,
    )


def run_bakeoff(
    scorers: list[Scorer],
    sample_rows: list[SampleRow],
    labels: Labels,
    quadrant_pool_sizes: dict[Quadrant, int],
) -> list[CandidateResult]:
    results = []
    for s in scorers:
        start = time.perf_counter()
        try:
            results.append(evaluate_candidate(s, sample_rows, labels, quadrant_pool_sizes))
        except Exception as e:  # one broken candidate must not sink the others
            results.append(_could_not_run(s, time.perf_counter() - start, f"{type(e).__name__}: {e}"))
    return results


# --- throughput (#15) --------------------------------------------------------
# A separate timing run, never part of `evaluate_candidate`: it reads no labels
# and produces no predictions, so it cannot touch the accuracy figures.

TIMING_ROWS = 500


def describe_hardware() -> str:
    try:  # macOS: "Apple M1"
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        chip = ""
    return f"{chip or platform.processor() or platform.machine()} ({platform.platform()})"


def measure_throughput(
    scorers: list[Scorer],
    sample_rows: list[SampleRow],
    timing_rows: int = TIMING_ROWS,
    hardware: str | None = None,
    clock=time.perf_counter,
) -> dict:
    """Rows per second per candidate on the first `timing_rows` distinct texts.

    One untimed single-text warm-up call first, so lazy model loading is not
    billed to throughput. Calls `scorer.score` directly (not `score_texts`), so
    this is the model's own rate regardless of any dedup layered on top. A
    scorer that raises is reported, never fatal."""
    texts = list(dict.fromkeys(r.product_details_text for r in sample_rows))[:timing_rows]
    candidates = []
    for s in scorers:
        entry = {"name": s.name, "rows": len(texts), "seconds": None, "rows_per_second": None,
                 "status": "ok", "error": None}
        try:
            s.score(texts[:1])
            start = clock()
            s.score(texts)
            entry["seconds"] = clock() - start
            entry["rows_per_second"] = len(texts) / entry["seconds"]
        except Exception as exc:
            entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
        candidates.append(entry)
    return {
        "basis": "measured",
        "hardware": hardware if hardware is not None else describe_hardware(),
        "timing_rows": timing_rows,
        "candidates": candidates,
    }


# --- report ------------------------------------------------------------------


def build_depth(results: list[CandidateResult], sample_rows: list[SampleRow], labels: Labels) -> dict:
    """Per-file metrics and miss lists per candidate (#14), from the per-row scores
    kept on each `CandidateResult`. Rows a candidate has no score for are skipped."""
    per_file: dict[str, dict] = {}
    misses: dict[str, list] = {}
    for r in results:
        if r.status != "ok" or r.gate_eligible is not True:  # no scores / AUROC-only: threshold breakdown is meaningless
            continue
        scored = [row for row in sample_rows if row.sample_index in r.scores]
        by_file: dict[str, list[SampleRow]] = {}
        for row in scored:
            by_file.setdefault(row.source_file, []).append(row)
        per_file[r.name] = {
            f: asdict(
                compute_classifier_metrics(
                    {x.sample_index: labels[x.sample_index][0] for x in rows},
                    {x.sample_index: r.scores[x.sample_index] >= THRESHOLD for x in rows},
                )
            )
            for f, rows in sorted(by_file.items())
        }
        misses[r.name] = [
            {
                "sample_index": row.sample_index,
                "source_file": row.source_file,
                "row_id": row.row_id,
                "kind": "false_negative" if labels[row.sample_index][0] else "false_positive",
                "score": r.scores[row.sample_index],
                "text": row.product_details_text,
            }
            for row in scored
            if (r.scores[row.sample_index] >= THRESHOLD) != labels[row.sample_index][0]
        ]
    return {"per_file": per_file, "misses": misses}


def build_report(
    results: list[CandidateResult],
    keywords: list[str],
    sample_size: int,
    quadrant_pool_sizes: dict[Quadrant, int],
    sample_rows: list[SampleRow] | None = None,
    labels: Labels | None = None,
) -> dict:
    """JSON-serialisable report. Add new top-level keys for new report depth (#14).
    Pass `sample_rows` + `labels` to include the per-file / miss-list depth."""
    for r in results:
        if r.status == "ok" and r.gate_eligible is not True and r.verdict != "AUROC-only":
            raise ValueError(f"{r.name} is gate-ineligible but has verdict {r.verdict!r}; refusing to report it")
    depth = build_depth(results, sample_rows, labels) if sample_rows is not None and labels is not None else {}
    return {
        **depth,
        "threshold": THRESHOLD,
        "gate": {"population_accuracy": GO_MIN_POPULATION_ACCURACY, "recall": GO_MIN_RECALL},
        "keywords": keywords,
        "sample_size": sample_size,
        "quadrant_pool_sizes": quadrant_pool_sizes,
        "candidates": [
            {
                "name": r.name,
                "model_id": r.model_id,
                "revision": r.revision,
                "template": r.template,
                "gate_eligible": r.gate_eligible,
                "status": r.status,
                "seconds": r.seconds,
                "raw": asdict(r.raw),
                "population_weighted": asdict(r.population_weighted),
                "auroc": r.auroc,
                "verdict": r.verdict,
                "failed_rows": r.failed_rows,
                "error": r.error,
            }
            for r in results
        ],
    }


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _section_header(report: dict) -> list[str]:
    gate = report["gate"]
    return [
        "# System 1 bake-off report",
        "",
        f"Fixed cut: {report['threshold']} (no tuning, no threshold sweep). "
        f"GO gate: >= {gate['population_accuracy']:.0%} population-reweighted accuracy and "
        f">= {gate['recall']:.0%} population-reweighted recall. AUROC is information only and never changes a verdict.",
        "",
        f"Keywords (from run config): {'; '.join(report['keywords'])}",
        "",
    ]


def _section_comparison(report: dict) -> list[str]:
    lines = [
        "## Comparison",
        "",
        "| candidate | verdict | pop. accuracy | pop. precision | pop. recall | pop. F1 "
        "| raw accuracy | raw precision | raw recall | raw F1 | AUROC (info only) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in report["candidates"]:
        p, r = c["population_weighted"], c["raw"]
        if c["status"] != "ok":
            lines.append(f"| {c['name']} | {c['verdict']} |" + " n/a |" * 9)
            continue
        if c["gate_eligible"] is not True:  # reference: AUROC only, no threshold metrics
            lines.append(f"| {c['name']} | {c['verdict']} |{' n/a |' * 8} {_fmt(c['auroc'])} |")
            continue
        lines.append(
            f"| {c['name']} | {c['verdict']} | {_fmt(p['accuracy'])} | {_fmt(p['precision'])} | {_fmt(p['recall'])} "
            f"| {_fmt(p['f1'])} | {_fmt(r['accuracy'])} | {_fmt(r['precision'])} | {_fmt(r['recall'])} "
            f"| {_fmt(r['f1'])} | {_fmt(c['auroc'])} |"
        )
    return [*lines, ""]


def _section_throughput(report: dict) -> list[str]:
    t = report.get("throughput")
    if not t:
        return []
    corpus_rows = sum(report.get("quadrant_pool_sizes", {}).values())
    lines = [
        "## Measured throughput",
        "",
        f"Measured on: {t['hardware']}. Timing run: up to {t['timing_rows']} distinct Ground-truth set row texts per "
        "candidate, after an untimed warm-up call. Separate from the accuracy run; does not affect any accuracy figure.",
        "",
        f"| candidate | rows timed | seconds | rows/sec (measured) | full corpus ({corpus_rows} rows), extrapolated |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in t["candidates"]:
        if c["status"] != "ok":
            lines.append(f"| {c['name']} | {c['rows']} | n/a | could not run: {c['error']} | n/a |")
            continue
        eta = f"{corpus_rows / c['rows_per_second']:.0f} s" if corpus_rows else "n/a"
        lines.append(f"| {c['name']} | {c['rows']} | {c['seconds']:.1f} | {c['rows_per_second']:.1f} | {eta} |")
    return [*lines, ""]


def _section_models(report: dict) -> list[str]:
    lines = ["## Models and frozen templates", ""]
    for c in report["candidates"]:
        lines += [
            f"### {c['name']}",
            "",
            f"- model: `{c['model_id']}` @ `{c['revision']}`",
            f"- gate-eligible: {c['gate_eligible']}",
            "- template (frozen before the first scoring run; `{keywords}` is filled from the run config):",
            "",
            "```",
            c["template"],
            "```",
            "",
        ]
    return lines


def _section_caveats(report: dict) -> list[str]:
    return [
        "## Caveats",
        "",
        f"- The Ground-truth set is Claude-labelled under human supervision and has only {report['sample_size']} rows; "
        "read close calls with caution.",
        "- The sample was stratified on Laya / Baseline-judge disagreement. Population weights keep the estimate "
        "unbiased for any classifier, but it is noisy for candidates that err in the agreement strata.",
        "- The Baseline judge's verdicts are a reference, not truth; the Ground-truth set is used only to score, "
        "never to train or tune.",
        "",
    ]


def _section_failures(report: dict) -> list[str]:
    cnr = [c for c in report["candidates"] if c["status"] != "ok"]
    bad = [c for c in report["candidates"] if c.get("failed_rows")]
    if not (cnr or bad):
        return []
    lines = ["## Failures", ""]
    lines += [f"- {c['name']}: could not run: {c['error']}" for c in cnr]
    lines += [
        f"- {c['name']}: {len(c['failed_rows'])} row(s) failed (non-finite or out-of-range score; "
        f"counted wrong in the metrics, excluded from AUROC): sample_index {c['failed_rows']}"
        for c in bad
    ]
    return [*lines, ""]


MISS_TEXT_CHARS = 200  # Markdown only; the JSON keeps the full text


def _flat(text: str, limit: int = MISS_TEXT_CHARS) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= limit else text[:limit] + "..."


def _section_per_file(report: dict) -> list[str]:
    if "per_file" not in report:
        return []
    lines = ["## Per-file breakdown", "", "Raw (unweighted) metrics at the fixed cut, per source file. Rows with a failed score are listed under "
        "Failures and not counted here.", ""]
    for name, files in report["per_file"].items():
        lines += [
            f"### {name}",
            "",
            "| file | rows | accuracy | precision | recall | F1 | FN | FP |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        lines += [
            f"| {f} | {m['total']} | {_fmt(m['accuracy'])} | {_fmt(m['precision'])} | {_fmt(m['recall'])} "
            f"| {_fmt(m['f1'])} | {m['false_negatives']} | {m['false_positives']} |"
            for f, m in files.items()
        ]
        lines.append("")
    return lines


def _section_misses(report: dict) -> list[str]:
    if "misses" not in report:
        return []
    lines = [
        "## Missed Ground-truth rows",
        "",
        f"Rows each candidate gets wrong at the fixed cut (text truncated to {MISS_TEXT_CHARS} chars here; "
        "full text in the JSON report). Failed-score rows are listed under Failures, not here.",
        "",
    ]
    for name, rows in report["misses"].items():
        lines += [f"### {name} ({len(rows)} missed)", ""]
        lines += [
            f"- {m['kind']} | score {m['score']:.3f} | {m['source_file']} row {m['row_id']}: {_flat(m['text'])}"
            for m in rows
        ]
        lines.append("")
    return lines


# Ordered report sections; add a section by adding a function here.
_SECTIONS = [
    _section_header,
    _section_comparison,
    _section_per_file,
    _section_misses,
    _section_failures,
    _section_throughput,
    _section_models,
    _section_caveats,
]


def render_markdown(report: dict) -> str:
    return "\n".join(line for section in _SECTIONS for line in section(report)) + "\n"


def write_report(report: dict, json_path: str | Path, markdown_path: str | Path) -> tuple[Path, Path]:
    json_path, markdown_path = Path(json_path), Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


# --- full-corpus agreement mode (#20; opt-in via --full-corpus) ---------------


@dataclass
class FullCorpusResult:
    name: str
    model_id: str
    revision: str
    seconds: float
    overall: ClassifierMetrics  # candidate vs the Baseline judge's verdicts; its "accuracy" is agreement
    per_file: dict[str, ClassifierMetrics]
    skipped: list[tuple[str, str]]  # (source_file, reason)


def evaluate_full_corpus(scorer: Scorer, run_config: RunConfig) -> FullCorpusResult:
    """Score every row of every `logs/*_rationale.jsonl` file at the fixed 0.5
    cut and compare with the Baseline judge's recorded verdicts. This measures
    AGREEMENT with the Baseline judge, not accuracy: the verdicts are a
    reference, not truth. No GO / NO-GO comes out of this mode."""
    start = time.perf_counter()
    per_file: dict[str, ClassifierMetrics] = {}
    skipped: list[tuple[str, str]] = []
    all_baseline: dict[tuple[str, int], bool] = {}
    all_pred: dict[tuple[str, int], bool] = {}
    for path in sorted(run_config.logs_dir_resolved.glob("*_rationale.jsonl")):
        source_file, records = load_rationale_file(path)
        baseline = {row_id: match for row_id, (match, _rationale) in records.items()}
        input_path = run_config.input_dir_resolved / source_file
        if not input_path.exists():
            skipped.append((source_file, f"input workbook not found at {input_path}"))
            continue
        sheet = load_main_sheet(input_path)
        if max(baseline) >= len(sheet.rows):
            skipped.append((source_file, "baseline references rows beyond the workbook (file changed since the baseline)"))
            continue
        row_ids = sorted(baseline)
        texts = [
            dedupe_truncation_artifact(str(sheet.rows[i].get(sheet.product_details_column) or "")) for i in row_ids
        ]
        scores = score_texts(scorer, texts)
        predictions = {i: s >= THRESHOLD for i, s in zip(row_ids, scores)}
        per_file[source_file] = compute_classifier_metrics(baseline, predictions)
        all_baseline.update({(source_file, i): baseline[i] for i in row_ids})
        all_pred.update({(source_file, i): p for i, p in predictions.items()})
    return FullCorpusResult(
        name=scorer.name,
        model_id=scorer.model_id,
        revision=scorer.revision,
        seconds=time.perf_counter() - start,
        overall=compute_classifier_metrics(all_baseline, all_pred),
        per_file=per_file,
        skipped=skipped,
    )


def _agreement_counts(m: ClassifierMetrics) -> dict:
    return {
        "rows": m.total,
        "agree": m.true_positives + m.true_negatives,
        "agreement_rate": m.accuracy,
        "candidate_match_baseline_no_match": m.false_positives,
        "candidate_no_match_baseline_match": m.false_negatives,
    }


def build_agreement_report(result: FullCorpusResult, keywords: list[str]) -> dict:
    return {
        "mode": "agreement_with_baseline_judge",
        "note": "Agreement with the Baseline judge's recorded verdicts over the full corpus. "
        "The Baseline judge is a reference, not truth; this is not a measure of correctness "
        "and does not feed the GO gate.",
        "threshold": THRESHOLD,
        "keywords": keywords,
        "candidate": result.name,
        "model_id": result.model_id,
        "revision": result.revision,
        "seconds": result.seconds,
        "rows_scored": result.overall.total,
        **_agreement_counts(result.overall),
        "per_file": {f: _agreement_counts(m) for f, m in sorted(result.per_file.items())},
        "skipped_files": [{"source_file": f, "reason": r} for f, r in result.skipped],
    }


def render_agreement_markdown(report: dict) -> str:
    lines = [
        f"# Full-corpus agreement with the Baseline judge: {report['candidate']}",
        "",
        report["note"],
        "",
        f"Fixed cut: {report['threshold']}. Rows scored: {report['rows_scored']}. "
        f"Agreement with the Baseline judge: {_fmt(report['agreement_rate'])} "
        f"({report['agree']} of {report['rows']}).",
        "",
        f"- candidate match / Baseline judge no match: {report['candidate_match_baseline_no_match']}",
        f"- candidate no match / Baseline judge match: {report['candidate_no_match_baseline_match']}",
        "",
        "| file | rows | agreement with the Baseline judge |",
        "|---|---:|---:|",
        *(f"| {f} | {c['rows']} | {_fmt(c['agreement_rate'])} |" for f, c in report["per_file"].items()),
        "",
    ]
    if report["skipped_files"]:
        lines += ["Skipped files:", *(f"- {s['source_file']}: {s['reason']}" for s in report["skipped_files"]), ""]
    return "\n".join(lines)


def _run_full_corpus(run_config: RunConfig, args: argparse.Namespace) -> int:
    if len(args.candidates) != 1:
        print("--full-corpus takes exactly one candidate (it is a slow, per-candidate run).", file=sys.stderr)
        return 2
    (scorer,) = build_scorers(args.candidates, run_config.keywords)
    report = build_agreement_report(evaluate_full_corpus(scorer, run_config), run_config.keywords)
    json_path = _resolve(run_config, args.agreement_report_path)
    md_path = _resolve(run_config, args.agreement_markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_agreement_markdown(report), encoding="utf-8")
    print(f"{report['candidate']}: agreement with the Baseline judge = {_fmt(report['agreement_rate'])} "
          f"over {report['rows_scored']} rows")
    print(f"Agreement report written to {json_path} and {md_path}")
    return 0


# --- CLI ---------------------------------------------------------------------


def _resolve(run_config: RunConfig, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else run_config.project_root / p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/run_config.yaml")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--sample-path", default="logs/ground_truth_sample.jsonl")
    parser.add_argument("--labels-path", default="logs/ground_truth_labels.jsonl")
    parser.add_argument("--laya-scores-path", default="logs/laya_predictions.jsonl")
    parser.add_argument(
        "--laya-threshold",
        type=float,
        default=DEFAULT_LAYA_THRESHOLD,
        help="The laya cut the sample was stratified at (NOT a bake-off threshold; that is fixed at 0.5).",
    )
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES), help=f"Known: {sorted(CANDIDATES)}")
    parser.add_argument("--timing-rows", type=int, default=TIMING_ROWS, help="Rows per candidate in the throughput run; 0 skips it.")
    parser.add_argument("--report-path", default="logs/system1_bakeoff_report.json")
    parser.add_argument("--markdown-path", default="logs/system1_bakeoff_report.md")
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Opt-in: score every row of every logs/*_rationale.jsonl and report agreement with the "
        "Baseline judge (not accuracy). Replaces the Ground-truth flow; pass exactly one candidate.",
    )
    parser.add_argument("--agreement-report-path", default="logs/system1_bakeoff_agreement.json")
    parser.add_argument("--agreement-markdown-path", default="logs/system1_bakeoff_agreement.md")
    args = parser.parse_args(argv)

    run_config = RunConfig.load(args.config, project_root=args.project_root)
    if args.full_corpus:
        return _run_full_corpus(run_config, args)
    labels = load_ground_truth_labels(_resolve(run_config, args.labels_path))
    sample_rows = [r for r in load_ground_truth_sample(_resolve(run_config, args.sample_path)) if r.sample_index in labels]
    if not sample_rows:
        print("No labelled Ground-truth rows found; build the sample and labels first.", file=sys.stderr)
        return 1
    pool_sizes = compute_quadrant_pool_sizes(
        run_config, _resolve(run_config, args.laya_scores_path), args.laya_threshold
    )

    scorers = build_scorers(args.candidates, run_config.keywords)
    results = run_bakeoff(scorers, sample_rows, labels, pool_sizes)
    report = build_report(results, run_config.keywords, len(sample_rows), pool_sizes, sample_rows, labels)
    if args.timing_rows > 0:
        report["throughput"] = measure_throughput(scorers, sample_rows, args.timing_rows)
    json_path, md_path = write_report(
        report, _resolve(run_config, args.report_path), _resolve(run_config, args.markdown_path)
    )

    for r in results:
        if r.status != "ok":
            print(f"{r.name}: could not run ({r.error})")
            continue
        if r.gate_eligible is not True:
            print(f"{r.name}: {r.verdict}  AUROC={_fmt(r.auroc)}")
            continue
        print(f"{r.name}: {r.verdict}  pop. accuracy={r.population_weighted.accuracy:.1%}  "
              f"pop. recall={r.population_weighted.recall:.1%}  AUROC={_fmt(r.auroc)}")
    print(f"Report written to {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
