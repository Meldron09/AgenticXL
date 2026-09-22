"""Compare all four `LayaClient` question_mode/payload_layout variants (issue #7)
against the existing 526-row blind-labeled ground-truth sample (issue #6:
`logs/ground_truth_sample.jsonl` + `logs/ground_truth_labels.jsonl`), and report
a GO / iterate / NO-GO verdict per variant.

`scripts/build_ground_truth_report.py` already answered "how accurate is laya,
today, at the shipped combined/legacy variant" -- 89.1% population-reweighted
accuracy at the legacy threshold of 0.65 (`logs/ground_truth_validation_report.json`),
short of the 95% gate. This script asks the follow-on question: does a different
question_mode/payload_layout combination close that gap? It never re-labels
anything -- it reuses the same ground-truth sample and labels, re-runs each
variant's `LayaClient.judge_batch` over the sample's `product_details_text`,
re-sweeps the threshold per variant (reusing `sweep_thresholds` from
`validate_laya_filtering.py`), and re-scores against ground truth (reusing
`compute_classifier_metrics` / `population_weighted_overall` from
`build_ground_truth_report.py`) -- never reimplementing either.

This script is tooling only: actually running it against the real laya-mlx
checkpoint (a ~15-20 minute run: 4 variants x 526 rows, `per_keyword` variants
one `predict()` call per row under `payload_layout="convention"`) is a separate
follow-on ticket, per issue #8.

Usage:
    .venv/bin/python scripts/build_laya_redesign_report.py
    .venv/bin/python scripts/build_laya_redesign_report.py --config config/run_config.yaml \\
        --report-path logs/laya_redesign_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from agenticxl.config import RunConfig
from agenticxl.filtering.laya_client import LayaClient, PayloadLayout, QuestionMode

from scripts.build_ground_truth_report import (
    ClassifierMetrics,
    Quadrant,
    SampleRow,
    compute_classifier_metrics,
    compute_quadrant_pool_sizes,
    load_ground_truth_labels,
    load_ground_truth_sample,
    population_weighted_overall,
)
from scripts.validate_laya_filtering import (
    DEFAULT_THRESHOLD_CANDIDATES,
    sweep_thresholds,
    write_score_dump,
)

Verdict = Literal["GO", "iterate", "NO-GO"]

VARIANTS: tuple[tuple[QuestionMode, PayloadLayout], ...] = (
    ("combined", "legacy"),
    ("combined", "convention"),
    ("per_keyword", "legacy"),
    ("per_keyword", "convention"),
)

# The threshold the sample/report were originally built at (issue #6's
# `DEFAULT_LAYA_THRESHOLD`), carried over here as the "legacy" comparison point
# alongside each variant's own re-swept threshold.
LEGACY_THRESHOLD = 0.65

# GitHub issue #6's finding for the shipped combined/legacy variant, at
# LEGACY_THRESHOLD (see logs/ground_truth_validation_report.json ->
# overall.population_weighted.laya.accuracy): the gap this script checks
# whether any variant closes.
BASELINE_POPULATION_ACCURACY = 0.891317570436289
GO_ACCURACY_THRESHOLD = 0.95

# A variant that closes less than this fraction of the (GO - baseline) gap
# hasn't moved the needle enough to justify further iteration -- NO-GO
# confirmed, per issue #8's "no real movement" criterion.
MEANINGFUL_GAP_CLOSURE_FRACTION = 0.25


def effective_batch_size(configured_batch_size: int, question_mode: QuestionMode, keyword_count: int) -> int:
    """`per_keyword` mode asks one question per (row, keyword) pair in a single
    shared-`state` `predict()` call, so its per-batch question count is
    `configured_batch_size * keyword_count` -- to keep that within the pinned
    checkpoint's `max_tokens_per_batch=4096` cap (`rl_agent_config.json`), the
    row batch size itself is divided by the keyword count, floored, minimum 1.
    `combined` mode asks one question per row regardless of keyword count, so
    its batch size is unchanged."""
    if question_mode != "per_keyword":
        return configured_batch_size
    return max(1, configured_batch_size // keyword_count)


def run_variant_scores(
    client: LayaClient,
    rows: list[tuple[int, str]],
    keywords: list[str],
    batch_size: int,
    question_mode: QuestionMode,
    payload_layout: PayloadLayout,
) -> dict[int, float]:
    """Run every (row_id, product_details_text) pair through `client.judge_batch`
    at this variant, chunked by `batch_size`, and return the raw `noul` score per
    row_id -- threshold-independent, so the sweep below never re-runs inference.
    Mirrors `validate_laya_filtering.classify_rows`, extended to pass the
    variant's `question_mode`/`payload_layout` through."""
    scores: dict[int, float] = {}
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        for verdict in client.judge_batch(
            chunk, keywords, threshold=0.0, question_mode=question_mode, payload_layout=payload_layout
        ):
            scores[verdict.row_id] = verdict.score
    return scores


@dataclass(frozen=True)
class VariantThresholdResult:
    threshold: float
    raw: ClassifierMetrics
    population_weighted: ClassifierMetrics


@dataclass(frozen=True)
class VariantReport:
    question_mode: str
    payload_layout: str
    effective_batch_size: int
    swept_best: VariantThresholdResult
    legacy: VariantThresholdResult
    verdict: Verdict


def determine_verdict(
    population_accuracy: float,
    *,
    baseline_accuracy: float = BASELINE_POPULATION_ACCURACY,
    go_threshold: float = GO_ACCURACY_THRESHOLD,
    min_gap_closure_fraction: float = MEANINGFUL_GAP_CLOSURE_FRACTION,
) -> Verdict:
    """GO if `population_accuracy` clears `go_threshold`. Otherwise, "iterate" if
    it closes at least `min_gap_closure_fraction` of the (go_threshold -
    baseline_accuracy) gap versus the shipped combined/legacy variant, else
    "NO-GO confirmed" -- no meaningful movement."""
    if population_accuracy >= go_threshold:
        return "GO"
    gap = go_threshold - baseline_accuracy
    if gap <= 0:
        return "GO"
    closed_fraction = (population_accuracy - baseline_accuracy) / gap
    if closed_fraction >= min_gap_closure_fraction:
        return "iterate"
    return "NO-GO"


def score_variant_at_threshold(
    ground_truth: dict[int, bool],
    scores: dict[int, float],
    threshold: float,
    sample_rows_by_index: dict[int, SampleRow],
    ground_truth_labels: dict[int, tuple[bool, str]],
    quadrant_pool_sizes: dict[Quadrant, int],
) -> VariantThresholdResult:
    """Score one variant's raw `scores` at one `threshold`: raw-sample metrics via
    `compute_classifier_metrics` (direct reuse), and population-reweighted metrics
    via `population_weighted_overall` (reused by building "shadow" SampleRows whose
    `laya_match` is overridden to this variant's verdict at `threshold`, keeping
    every row's original `quadrant` -- the corpus stratification is a property of
    the sample, not of which variant is being scored against it)."""
    predictions = {row_id: score >= threshold for row_id, score in scores.items()}
    raw = compute_classifier_metrics(ground_truth, predictions)

    shadow_rows = [
        replace(sample_rows_by_index[row_id], laya_match=match) for row_id, match in predictions.items()
    ]
    population_weighted = population_weighted_overall(shadow_rows, ground_truth_labels, quadrant_pool_sizes).laya

    return VariantThresholdResult(threshold=threshold, raw=raw, population_weighted=population_weighted)


def build_variant_report(
    question_mode: QuestionMode,
    payload_layout: PayloadLayout,
    variant_effective_batch_size: int,
    ground_truth: dict[int, bool],
    scores: dict[int, float],
    sample_rows_by_index: dict[int, SampleRow],
    ground_truth_labels: dict[int, tuple[bool, str]],
    quadrant_pool_sizes: dict[Quadrant, int],
    threshold_candidates: list[float] = DEFAULT_THRESHOLD_CANDIDATES,
    legacy_threshold: float = LEGACY_THRESHOLD,
) -> VariantReport:
    """Assemble one variant's full report: re-sweep the threshold against this
    variant's own score distribution (reusing `sweep_thresholds`, keyed on the
    ground truth label rather than the LLM baseline it was originally written to
    compare against -- it is a generic two-boolean-map comparison), score at both
    the empirically best swept threshold and the carried-over legacy threshold,
    and determine the GO/iterate/NO-GO verdict from the swept-best population-
    weighted accuracy (the most favorable figure this variant's design can reach)."""
    sweep = sweep_thresholds({"ground_truth_sample": (ground_truth, scores)}, threshold_candidates)
    best_threshold = max(sweep, key=lambda t: sweep[t].weighted_agreement_rate)

    swept_best = score_variant_at_threshold(
        ground_truth, scores, best_threshold, sample_rows_by_index, ground_truth_labels, quadrant_pool_sizes
    )
    legacy = score_variant_at_threshold(
        ground_truth, scores, legacy_threshold, sample_rows_by_index, ground_truth_labels, quadrant_pool_sizes
    )
    verdict = determine_verdict(swept_best.population_weighted.accuracy)

    return VariantReport(
        question_mode=question_mode,
        payload_layout=payload_layout,
        effective_batch_size=variant_effective_batch_size,
        swept_best=swept_best,
        legacy=legacy,
        verdict=verdict,
    )


def _scores_by_source_file(
    sample_rows: list[SampleRow], scores_by_sample_index: dict[int, float]
) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = {}
    for row in sample_rows:
        result.setdefault(row.source_file, {})[row.row_id] = scores_by_sample_index[row.sample_index]
    return result


def write_variant_score_dump(
    path: str | Path,
    sample_rows: list[SampleRow],
    scores_by_sample_index: dict[int, float],
    threshold: float,
    model_identifier: str,
    question_mode: QuestionMode,
    payload_layout: PayloadLayout,
) -> Path:
    """Persist this variant's raw per-row scores via `write_score_dump` (reused,
    not reimplemented), remapped from sample_index back to each row's real
    (source_file, row_id) identity for auditability. Tagged per variant two ways:
    a distinct `path` per (question_mode, payload_layout) so dumps never collide,
    and `question_mode`/`payload_layout`/`threshold` as their own fields on every
    record (via `write_score_dump`'s variant-tagging parameters) so a dump is
    self-describing even in isolation."""
    per_file_scores = _scores_by_source_file(sample_rows, scores_by_sample_index)
    per_file_data = {source_file: ({}, scores) for source_file, scores in per_file_scores.items()}
    return write_score_dump(
        path, per_file_data, threshold, model_identifier, question_mode=question_mode, payload_layout=payload_layout
    )


def run_variant_and_report(
    client: LayaClient,
    labeled_rows: list[SampleRow],
    keywords: list[str],
    configured_batch_size: int,
    question_mode: QuestionMode,
    payload_layout: PayloadLayout,
    ground_truth_labels: dict[int, tuple[bool, str]],
    quadrant_pool_sizes: dict[Quadrant, int],
    scores_dir: Path,
    threshold_candidates: list[float] = DEFAULT_THRESHOLD_CANDIDATES,
    legacy_threshold: float = LEGACY_THRESHOLD,
) -> VariantReport:
    """End-to-end for one variant: run inference over `labeled_rows`, write the
    tagged score dump, and assemble its report."""
    batch_size = effective_batch_size(configured_batch_size, question_mode, len(keywords))
    rows = [(r.sample_index, r.product_details_text) for r in labeled_rows]
    scores = run_variant_scores(client, rows, keywords, batch_size, question_mode, payload_layout)

    ground_truth = {r.sample_index: ground_truth_labels[r.sample_index][0] for r in labeled_rows}
    sample_rows_by_index = {r.sample_index: r for r in labeled_rows}

    report = build_variant_report(
        question_mode,
        payload_layout,
        batch_size,
        ground_truth,
        scores,
        sample_rows_by_index,
        ground_truth_labels,
        quadrant_pool_sizes,
        threshold_candidates,
        legacy_threshold,
    )

    scores_path = scores_dir / f"{question_mode}_{payload_layout}.jsonl"
    write_variant_score_dump(
        scores_path, labeled_rows, scores, report.swept_best.threshold, client.model_identifier,
        question_mode, payload_layout,
    )
    return report


def _resolve(run_config: RunConfig, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else run_config.project_root / p


def _variant_row(r: dict) -> str:
    return (
        f"| {r['question_mode']}/{r['payload_layout']} | {r['effective_batch_size']} | "
        f"{r['swept_best']['threshold']:.2f} | {r['swept_best']['population_weighted']['accuracy']:.1%} | "
        f"{r['swept_best']['raw']['accuracy']:.1%} | {r['legacy']['population_weighted']['accuracy']:.1%} | "
        f"{r['legacy']['raw']['accuracy']:.1%} | {r['verdict']} |"
    )


def _render_markdown(report: dict) -> str:
    lines = [
        "# Laya variant ground-truth comparison",
        "",
        f"Sample size: {report['sample_size']}"
        + (f"  |  Excluded (no label yet): {report['unlabeled_excluded']}" if report["unlabeled_excluded"] else "")
        + f"  |  Legacy threshold: {report['legacy_threshold']}",
        "",
        f"GO gate: population-reweighted accuracy >= {report['go_accuracy_threshold']:.0%}. "
        f"Baseline (combined/legacy variant @ legacy threshold): {report['baseline_population_accuracy']:.1%}.",
        "",
        "| variant | eff. batch size | swept threshold | swept pop-weighted acc | swept raw acc | "
        "legacy pop-weighted acc | legacy raw acc | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        *[_variant_row(r) for r in report["variants"]],
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/run_config.yaml", help="Path to run_config.yaml")
    parser.add_argument("--project-root", default=None, help="Project root (default: repo root)")
    parser.add_argument("--sample-path", default="logs/ground_truth_sample.jsonl")
    parser.add_argument("--labels-path", default="logs/ground_truth_labels.jsonl")
    parser.add_argument(
        "--legacy-scores-path",
        default="logs/laya_predictions.jsonl",
        help="Legacy combined/legacy-variant score dump, used only to recompute the "
        "corpus-wide quadrant pool sizes for population reweighting.",
    )
    parser.add_argument("--legacy-threshold", type=float, default=LEGACY_THRESHOLD)
    parser.add_argument("--report-path", default="logs/laya_redesign_report.json")
    parser.add_argument("--markdown-path", default="logs/laya_redesign_report.md")
    parser.add_argument("--scores-dir", default="logs/laya_redesign_scores")
    args = parser.parse_args(argv)

    run_config = RunConfig.load(args.config, project_root=args.project_root)

    sample_path = _resolve(run_config, args.sample_path)
    labels_path = _resolve(run_config, args.labels_path)
    legacy_scores_path = _resolve(run_config, args.legacy_scores_path)
    report_path = _resolve(run_config, args.report_path)
    markdown_path = _resolve(run_config, args.markdown_path)
    scores_dir = _resolve(run_config, args.scores_dir)

    sample_rows = load_ground_truth_sample(sample_path)
    if not sample_rows:
        print(f"No sample rows found at {sample_path}; run build_ground_truth_report.py 'sample' first.", file=sys.stderr)
        return 1

    ground_truth_labels = load_ground_truth_labels(labels_path)
    labeled_rows = [r for r in sample_rows if r.sample_index in ground_truth_labels]
    unlabeled_excluded = len(sample_rows) - len(labeled_rows)
    if unlabeled_excluded:
        print(
            f"WARN: {unlabeled_excluded} sampled rows have no ground-truth label yet -- excluding them.",
            file=sys.stderr,
        )
    if not labeled_rows:
        print("No labeled rows to compare.", file=sys.stderr)
        return 1

    quadrant_pool_sizes = compute_quadrant_pool_sizes(run_config, legacy_scores_path, args.legacy_threshold)

    print(f"Loading laya-mlx checkpoint ({run_config.relevance_batch_size}-row native batches)...")
    client = LayaClient(batch_size=run_config.relevance_batch_size)

    scores_dir.mkdir(parents=True, exist_ok=True)
    variant_reports: list[VariantReport] = []
    for question_mode, payload_layout in VARIANTS:
        print(f"Running variant question_mode={question_mode} payload_layout={payload_layout}...")
        report = run_variant_and_report(
            client,
            labeled_rows,
            run_config.keywords,
            run_config.relevance_batch_size,
            question_mode,
            payload_layout,
            ground_truth_labels,
            quadrant_pool_sizes,
            scores_dir,
            legacy_threshold=args.legacy_threshold,
        )
        variant_reports.append(report)
        print(
            f"  swept-best threshold={report.swept_best.threshold:.2f} "
            f"pop-weighted accuracy={report.swept_best.population_weighted.accuracy:.1%} "
            f"(legacy thr={args.legacy_threshold:.2f} -> {report.legacy.population_weighted.accuracy:.1%}) "
            f"verdict={report.verdict}"
        )

    report_dict = {
        "sample_size": len(labeled_rows),
        "unlabeled_excluded": unlabeled_excluded,
        "legacy_threshold": args.legacy_threshold,
        "go_accuracy_threshold": GO_ACCURACY_THRESHOLD,
        "baseline_population_accuracy": BASELINE_POPULATION_ACCURACY,
        "variants": [asdict(r) for r in variant_reports],
        "model": client.model_identifier,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_dict, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_markdown(report_dict), encoding="utf-8")

    print(f"\nComparison report written to {report_path} and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
