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
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from agenticxl.config import RunConfig

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
    status: str  # "ok" (#13 adds "could_not_run")
    seconds: float
    raw: ClassifierMetrics
    population_weighted: ClassifierMetrics
    auroc: float | None
    verdict: str  # "GO" | "NO-GO" | "AUROC-only"
    scores: dict[int, float]  # sample_index -> probability (kept for #14 miss lists; not in the JSON)


# --- scoring -----------------------------------------------------------------


def score_texts(scorer: Scorer, texts: list[str]) -> list[float]:
    """One probability per text, in order. Dedup of identical texts and
    per-row bad-score handling (#13) belong here."""
    scores = scorer.score(texts)
    if len(scores) != len(texts):
        raise ValueError(f"{scorer.name} returned {len(scores)} scores for {len(texts)} texts")
    return scores


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
    if not gate_eligible:
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

    by_index = {r.sample_index: s for r, s in zip(sample_rows, scores)}
    predictions = {i: s >= THRESHOLD for i, s in by_index.items()}
    truth = {r.sample_index: labels[r.sample_index][0] for r in sample_rows}

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
    )


def run_bakeoff(
    scorers: list[Scorer],
    sample_rows: list[SampleRow],
    labels: Labels,
    quadrant_pool_sizes: dict[Quadrant, int],
) -> list[CandidateResult]:
    return [evaluate_candidate(s, sample_rows, labels, quadrant_pool_sizes) for s in scorers]


# --- report ------------------------------------------------------------------


def build_report(
    results: list[CandidateResult],
    keywords: list[str],
    sample_size: int,
    quadrant_pool_sizes: dict[Quadrant, int],
) -> dict:
    """JSON-serialisable report. Add new top-level keys for new report depth (#14)."""
    return {
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
        lines.append(
            f"| {c['name']} | {c['verdict']} | {_fmt(p['accuracy'])} | {_fmt(p['precision'])} | {_fmt(p['recall'])} "
            f"| {_fmt(p['f1'])} | {_fmt(r['accuracy'])} | {_fmt(r['precision'])} | {_fmt(r['recall'])} "
            f"| {_fmt(r['f1'])} | {_fmt(c['auroc'])} |"
        )
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


# Ordered report sections; add a section (e.g. per-file breakdown, miss lists) by adding a function here.
_SECTIONS = [_section_header, _section_comparison, _section_models, _section_caveats]


def render_markdown(report: dict) -> str:
    return "\n".join(line for section in _SECTIONS for line in section(report)) + "\n"


def write_report(report: dict, json_path: str | Path, markdown_path: str | Path) -> tuple[Path, Path]:
    json_path, markdown_path = Path(json_path), Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


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
    parser.add_argument("--report-path", default="logs/system1_bakeoff_report.json")
    parser.add_argument("--markdown-path", default="logs/system1_bakeoff_report.md")
    args = parser.parse_args(argv)

    run_config = RunConfig.load(args.config, project_root=args.project_root)
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
    report = build_report(results, run_config.keywords, len(sample_rows), pool_sizes)
    json_path, md_path = write_report(
        report, _resolve(run_config, args.report_path), _resolve(run_config, args.markdown_path)
    )

    for r in results:
        print(f"{r.name}: {r.verdict}  pop. accuracy={r.population_weighted.accuracy:.1%}  "
              f"pop. recall={r.population_weighted.recall:.1%}  AUROC={_fmt(r.auroc)}")
    print(f"Report written to {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
