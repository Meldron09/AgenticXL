"""Independent ground-truth validation of laya vs. the LLM-baseline relevance judge.

`scripts/validate_laya_filtering.py` (issue #2) already answered "does laya agree
with the LLM baseline" -- NO-GO, 88.1% agreement vs. a 95% gate. That comparison
treats the LLM's own judgment as ground truth, which the plan behind this script
considers incomplete: it never checks whether either system is actually *right*.

This script builds a third, independent source of truth -- a human-supervised
Claude judging a stratified sample of rows directly, blind to both systems'
verdicts -- and reports each classifier's accuracy/precision/recall/F1 against
that ground truth, instead of against each other.

Workflow (two CLI subcommands plus a manual labeling phase in between, since
labeling requires live subagent dispatch, not a subprocess call):

    .venv/bin/python scripts/build_ground_truth_report.py sample
    # -- dispatch parallel labeling subagents here, collect into
    #    logs/ground_truth_labels.jsonl via append_ground_truth_labels --
    .venv/bin/python scripts/build_ground_truth_report.py report

See the plan (docs/agents or the session's plan file) for the full design
rationale: quadrant definitions, sampling weights, and the blinding guarantee.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from agenticxl.config import RunConfig
from agenticxl.excel.reader import load_main_sheet
from agenticxl.filtering.relevance import dedupe_truncation_artifact

Quadrant = Literal["both_match", "both_no_match", "laya_false_negative", "laya_false_positive"]

# The 3 files with the worst laya/LLM agreement in the original validation
# (all under 70%, see logs/laya_validation_report.json / GitHub issue #2) --
# boosted so each still gets meaningful sample coverage despite being small.
WORST_AGREEMENT_FILES = frozenset(
    {
        "Comet Offi - Export to India - 2021 to 2025.xlsx",
        "Jinwoo - Export to India - 2021 to 2025.xlsx",
        "Sinoboom - Export to India - 2021 to 2025.xlsx",
    }
)
BOOST_WEIGHT = 3.0
MIN_ROWS_PER_BOOSTED_FILE = 15
DEFAULT_LAYA_THRESHOLD = 0.65

# ~50% of the ~500-row sample deliberately goes to the two disagreement
# quadrants (a ~4x oversampling ratio vs. their ~12% share of the corpus),
# since that's the failure mode issue #1 flagged as most informative;
# agreement quadrants still get enough rows for a real accuracy check.
DEFAULT_QUADRANT_TARGETS: dict[Quadrant, int] = {
    "laya_false_negative": 150,
    "laya_false_positive": 100,
    "both_match": 150,
    "both_no_match": 100,
}


@dataclass(frozen=True)
class SampleRow:
    """One row selected into the ground-truth sample, with everything needed to
    label it (product_details_text) and everything needed to score both
    classifiers against the eventual ground-truth label (laya_score/laya_match,
    llm_match/llm_rationale) -- but a labeling subagent must only ever see the
    former, never the latter (see `blinded_batch_payload`)."""

    sample_index: int
    source_file: str
    row_id: int
    quadrant: Quadrant
    product_details_text: str
    laya_score: float
    laya_match: bool
    llm_match: bool
    llm_rationale: str


def classify_quadrant(llm_match: bool, laya_match: bool) -> Quadrant:
    """Classify one row's (LLM baseline, laya) verdict pair into an agreement/
    disagreement quadrant. A "false negative" here means laya said no-match where
    the LLM baseline said match (laya under-filtering) -- the failure mode
    `scripts/validate_laya_filtering.py` already flagged as most dangerous."""
    if llm_match and laya_match:
        return "both_match"
    if not llm_match and not laya_match:
        return "both_no_match"
    if llm_match and not laya_match:
        return "laya_false_negative"
    return "laya_false_positive"


_ALL_QUADRANTS: tuple[Quadrant, ...] = (
    "both_match",
    "both_no_match",
    "laya_false_negative",
    "laya_false_positive",
)


def build_quadrant_pools(
    baselines: dict[str, dict[int, bool]],
    scores: dict[str, dict[int, float]],
    threshold: float,
) -> dict[Quadrant, dict[str, list[int]]]:
    """Bucket every (source_file, row_id) present in `baselines` into its
    agreement/disagreement quadrant against laya's score at `threshold`.

    A row present in a baseline but missing from `scores` (e.g. a partial
    scores dump) is skipped -- never fabricated as a match or no-match, per
    this codebase's isolate-failures convention."""
    pools: dict[Quadrant, dict[str, list[int]]] = {q: {} for q in _ALL_QUADRANTS}
    for source_file, baseline in baselines.items():
        file_scores = scores.get(source_file, {})
        for row_id in sorted(baseline):
            if row_id not in file_scores:
                continue
            laya_match = file_scores[row_id] >= threshold
            quadrant = classify_quadrant(baseline[row_id], laya_match)
            pools[quadrant].setdefault(source_file, []).append(row_id)
    return pools


@dataclass(frozen=True)
class ClassifierMetrics:
    total: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1: float


def compute_classifier_metrics(
    ground_truth: dict[int, bool], predictions: dict[int, bool]
) -> ClassifierMetrics:
    """Standard accuracy/precision/recall/F1 of `predictions` against a true
    `ground_truth` label per row_id (both dicts must cover the same row_ids).

    Unlike `compute_agreement` in `validate_laya_filtering.py` (a symmetric
    agreement measure with no true label), this is directional: it answers
    "how often is this classifier actually right," not "how often do two
    classifiers agree." Precision/recall are each defined as 0.0 in the
    degenerate case where their denominator would be zero (no predicted
    positives, or no actual positives) rather than raising."""
    if set(ground_truth) != set(predictions):
        only_truth = sorted(set(ground_truth) - set(predictions))
        only_predictions = sorted(set(predictions) - set(ground_truth))
        raise ValueError(
            "ground_truth and predictions must cover the same row_ids: "
            f"only in ground_truth={only_truth[:10]}, only in predictions={only_predictions[:10]}"
        )

    total = len(ground_truth)
    tp = fp = fn = tn = 0
    for row_id, truth in ground_truth.items():
        predicted = predictions[row_id]
        if truth and predicted:
            tp += 1
        elif truth and not predicted:
            fn += 1
        elif not truth and predicted:
            fp += 1
        else:
            tn += 1

    accuracy = (tp + tn) / total if total else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return ClassifierMetrics(
        total=total,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


@dataclass(frozen=True)
class GroundTruthComparisonReport:
    laya: ClassifierMetrics
    llm_baseline: ClassifierMetrics


def build_ground_truth_sample(
    run_config: RunConfig,
    laya_predictions_path: str | Path,
    threshold: float = DEFAULT_LAYA_THRESHOLD,
    quadrant_targets: dict[Quadrant, int] | None = None,
    boosted_files: frozenset[str] | set[str] = WORST_AGREEMENT_FILES,
    boost_weight: float = BOOST_WEIGHT,
    min_rows_per_boosted_file: int = MIN_ROWS_PER_BOOSTED_FILE,
    min_nonblank_header_cells: int | None = None,
) -> list[SampleRow]:
    """Build the full stratified ground-truth sample: load every
    `logs/*_rationale.jsonl` baseline plus `laya_predictions_path`, bucket
    into quadrants, allocate row_ids per `allocate_quadrant_targets`, then
    load each involved file's PRODUCT DETAILS text once via `load_main_sheet`
    (applying `dedupe_truncation_artifact`, since that's what both laya and
    the LLM baseline actually judged) and materialize `SampleRow` records
    sorted by (source_file, row_id), with `sample_index` assigned in that
    order.

    `min_nonblank_header_cells` is passed straight through to
    `load_main_sheet` purely so tests can use small synthetic workbooks;
    production callers should leave it at `None` (the reader's own default).
    """
    if quadrant_targets is None:
        quadrant_targets = DEFAULT_QUADRANT_TARGETS

    logs_dir = run_config.logs_dir_resolved
    rationale_files = sorted(logs_dir.glob("*_rationale.jsonl"))
    if not rationale_files:
        raise ValueError(f"No *_rationale.jsonl baseline files found in {logs_dir}")

    baselines: dict[str, dict[int, bool]] = {}
    rationales: dict[str, dict[int, str]] = {}
    for path in rationale_files:
        source_file, records = load_rationale_file(path)
        baselines[source_file] = {row_id: match for row_id, (match, _rationale) in records.items()}
        rationales[source_file] = {row_id: rationale for row_id, (_match, rationale) in records.items()}

    scores = load_laya_predictions(laya_predictions_path)

    pools = build_quadrant_pools(baselines, scores, threshold)
    allocation = allocate_quadrant_targets(
        pools, quadrant_targets, boosted_files, boost_weight, min_rows_per_boosted_file
    )

    # Flatten to one quadrant per (source_file, row_id) before touching any
    # xlsx file, so each involved file is loaded exactly once regardless of
    # how many quadrants it contributed rows to.
    quadrant_by_row: dict[str, dict[int, Quadrant]] = {}
    for quadrant, by_file in allocation.items():
        for source_file, row_ids in by_file.items():
            for row_id in row_ids:
                quadrant_by_row.setdefault(source_file, {})[row_id] = quadrant

    reader_kwargs = {} if min_nonblank_header_cells is None else {
        "min_nonblank_header_cells": min_nonblank_header_cells
    }

    sample_rows: list[SampleRow] = []
    for source_file, row_quadrants in quadrant_by_row.items():
        input_path = run_config.input_dir_resolved / source_file
        sheet = load_main_sheet(input_path, **reader_kwargs)
        for row_id, quadrant in row_quadrants.items():
            text = str(sheet.rows[row_id].get(sheet.product_details_column) or "")
            sample_rows.append(
                SampleRow(
                    sample_index=-1,  # assigned below, after global sort
                    source_file=source_file,
                    row_id=row_id,
                    quadrant=quadrant,
                    product_details_text=dedupe_truncation_artifact(text),
                    laya_score=scores[source_file][row_id],
                    laya_match=scores[source_file][row_id] >= threshold,
                    llm_match=baselines[source_file][row_id],
                    llm_rationale=rationales[source_file][row_id],
                )
            )

    sample_rows.sort(key=lambda r: (r.source_file, r.row_id))
    return [
        SampleRow(
            sample_index=i,
            source_file=r.source_file,
            row_id=r.row_id,
            quadrant=r.quadrant,
            product_details_text=r.product_details_text,
            laya_score=r.laya_score,
            laya_match=r.laya_match,
            llm_match=r.llm_match,
            llm_rationale=r.llm_rationale,
        )
        for i, r in enumerate(sample_rows)
    ]


def write_ground_truth_sample(path: str | Path, sample_rows: list[SampleRow]) -> Path:
    """Persist the full sample (every field, including laya/LLM verdicts) for
    the orchestrating session to use between the `sample` and `report` CLI
    phases -- never given to a labeling subagent (see `blinded_batch_payload`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in sample_rows:
            f.write(json.dumps(asdict(row)) + "\n")
    return path


def load_ground_truth_sample(path: str | Path) -> list[SampleRow]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(SampleRow(**json.loads(line)))
    return rows


def compute_ground_truth_report(
    sample_rows: list[SampleRow],
    ground_truth_labels: dict[int, tuple[bool, str]],
) -> GroundTruthComparisonReport:
    """Score laya and the LLM baseline against the true ground-truth label for
    each sampled row -- never against each other. `ground_truth_labels` maps
    sample_index -> (ground_truth_match, rationale), as produced by
    `load_ground_truth_labels`."""
    ground_truth = {r.sample_index: ground_truth_labels[r.sample_index][0] for r in sample_rows}
    laya_predictions = {r.sample_index: r.laya_match for r in sample_rows}
    llm_predictions = {r.sample_index: r.llm_match for r in sample_rows}
    return GroundTruthComparisonReport(
        laya=compute_classifier_metrics(ground_truth, laya_predictions),
        llm_baseline=compute_classifier_metrics(ground_truth, llm_predictions),
    )


def _evenly_spaced(items: list, n: int) -> list:
    n = min(n, len(items))
    if n <= 0 or not items:
        return []
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def _allocate_one_quadrant(
    by_file: dict[str, list[int]],
    target: int,
    boosted_files: frozenset[str],
    boost_weight: float,
) -> dict[str, int]:
    """Split `target` rows across `by_file`'s files, weighted by pool size
    (boosted files counted `boost_weight` times as heavily), via largest-
    remainder rounding, capped at each file's own pool size and redistributed
    to the remaining files when a cap is hit. Deterministic: ties in the
    largest-remainder step break on file name."""
    caps = {f: len(rows) for f, rows in by_file.items()}
    active = {f for f, cap in caps.items() if cap > 0}
    allocation = {f: 0 for f in by_file}
    remaining_target = min(target, sum(caps.values()))

    while remaining_target > 0 and active:
        weights = {f: caps[f] * (boost_weight if f in boosted_files else 1.0) for f in active}
        weight_sum = sum(weights.values())
        if weight_sum <= 0:
            break
        raw = {f: remaining_target * weights[f] / weight_sum for f in active}
        proposed = {f: int(raw[f]) for f in active}
        remainder = remaining_target - sum(proposed.values())
        by_fraction = sorted(active, key=lambda f: (-(raw[f] - proposed[f]), f))
        for f in by_fraction[:remainder]:
            proposed[f] += 1

        overflow = 0
        for f in list(active):
            room = caps[f] - allocation[f]
            if proposed[f] > room:
                overflow += proposed[f] - room
                allocation[f] = caps[f]
                active.discard(f)
            else:
                allocation[f] += proposed[f]
        remaining_target = overflow

    return {f: n for f, n in allocation.items() if n > 0}


def allocate_quadrant_targets(
    pools: dict[Quadrant, dict[str, list[int]]],
    quadrant_targets: dict[Quadrant, int],
    boosted_files: frozenset[str] | set[str] = frozenset(),
    boost_weight: float = 1.0,
    min_rows_per_boosted_file: int = 0,
) -> dict[Quadrant, dict[str, list[int]]]:
    """Deterministic, no RNG: pick which row_ids from `pools` go into the
    ground-truth sample. Per quadrant, `_allocate_one_quadrant` decides how
    many rows come from each file (weighted by pool size, boosted files
    counted `boost_weight` times as heavily); `_evenly_spaced` then picks
    which specific row_ids within each (file, quadrant) cell.

    After the per-quadrant pass, any boosted file whose total allocation
    (summed across all quadrants) is still below `min_rows_per_boosted_file`
    is topped up from its own remaining (unselected) pool, prioritizing
    laya_false_negative > laya_false_positive > both_match > both_no_match,
    up to the floor or its remaining pool, whichever is smaller -- this can
    push the total sample size slightly above the nominal sum of
    `quadrant_targets`."""
    boosted_files = frozenset(boosted_files)
    remaining: dict[Quadrant, dict[str, list[int]]] = {
        q: {f: list(rows) for f, rows in by_file.items()} for q, by_file in pools.items()
    }
    allocation: dict[Quadrant, dict[str, list[int]]] = {q: {} for q in pools}

    for quadrant, target in quadrant_targets.items():
        by_file = remaining.get(quadrant, {})
        counts = _allocate_one_quadrant(by_file, target, boosted_files, boost_weight)
        for source_file, n in counts.items():
            available = remaining[quadrant][source_file]
            selected = _evenly_spaced(available, n)
            allocation[quadrant][source_file] = selected
            selected_set = set(selected)
            remaining[quadrant][source_file] = [r for r in available if r not in selected_set]

    top_up_priority: tuple[Quadrant, ...] = (
        "laya_false_negative",
        "laya_false_positive",
        "both_match",
        "both_no_match",
    )
    for source_file in boosted_files:
        total = sum(len(allocation[q].get(source_file, [])) for q in allocation)
        needed = min_rows_per_boosted_file - total
        if needed <= 0:
            continue
        for quadrant in top_up_priority:
            if needed <= 0:
                break
            available = remaining.get(quadrant, {}).get(source_file, [])
            if not available:
                continue
            take = min(needed, len(available))
            extra = _evenly_spaced(available, take)
            allocation[quadrant][source_file] = sorted(allocation[quadrant].get(source_file, []) + extra)
            extra_set = set(extra)
            remaining[quadrant][source_file] = [r for r in available if r not in extra_set]
            needed -= take

    return allocation


def _weighted_metrics(weighted_tp: float, weighted_fp: float, weighted_fn: float, weighted_tn: float) -> ClassifierMetrics:
    total = weighted_tp + weighted_fp + weighted_fn + weighted_tn
    accuracy = (weighted_tp + weighted_tn) / total if total else 1.0
    precision = weighted_tp / (weighted_tp + weighted_fp) if (weighted_tp + weighted_fp) else 0.0
    recall = weighted_tp / (weighted_tp + weighted_fn) if (weighted_tp + weighted_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return ClassifierMetrics(
        total=total,
        true_positives=weighted_tp,
        false_positives=weighted_fp,
        false_negatives=weighted_fn,
        true_negatives=weighted_tn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def population_weighted_metrics(
    sample_rows: list[SampleRow],
    ground_truth_labels: dict[int, tuple[bool, str]],
    quadrant_pool_sizes: dict[Quadrant, int],
    predictions: dict[int, bool],
) -> ClassifierMetrics:
    """Population-reweighted accuracy/precision/recall/F1 for ANY classifier.

    Standard stratified-sampling (Horvitz-Thompson-style) correction: the
    sample deliberately oversamples disagreement quadrants relative to their
    true share of the corpus, so a flat/unweighted accuracy over the raw
    sample misrepresents "how often is this classifier right across the
    whole dataset." Reweights each quadrant's confusion-matrix counts by
    `quadrant_pool_sizes[quadrant] / (sampled rows in that quadrant)`.

    `predictions` maps `sample_index` -> predicted match for the classifier
    being scored. The quadrants (and so the weights) are fixed by the sample's
    laya/baseline strata, so the correction is unbiased for any classifier."""
    rows_by_quadrant: dict[Quadrant, list[SampleRow]] = {}
    for row in sample_rows:
        rows_by_quadrant.setdefault(row.quadrant, []).append(row)

    wtp = wfp = wfn = wtn = 0.0
    for quadrant, rows in rows_by_quadrant.items():
        weight = quadrant_pool_sizes.get(quadrant, len(rows)) / len(rows)
        for row in rows:
            truth = ground_truth_labels[row.sample_index][0]
            predicted = predictions[row.sample_index]
            if truth and predicted:
                wtp += weight
            elif truth and not predicted:
                wfn += weight
            elif not truth and predicted:
                wfp += weight
            else:
                wtn += weight
    return _weighted_metrics(wtp, wfp, wfn, wtn)


def population_weighted_overall(
    sample_rows: list[SampleRow],
    ground_truth_labels: dict[int, tuple[bool, str]],
    quadrant_pool_sizes: dict[Quadrant, int],
) -> GroundTruthComparisonReport:
    """`population_weighted_metrics` for laya and the LLM baseline independently."""
    return GroundTruthComparisonReport(
        laya=population_weighted_metrics(
            sample_rows, ground_truth_labels, quadrant_pool_sizes, {r.sample_index: r.laya_match for r in sample_rows}
        ),
        llm_baseline=population_weighted_metrics(
            sample_rows, ground_truth_labels, quadrant_pool_sizes, {r.sample_index: r.llm_match for r in sample_rows}
        ),
    )


def compute_breakdowns(
    sample_rows: list[SampleRow],
    ground_truth_labels: dict[int, tuple[bool, str]],
    key_fn,
) -> dict:
    """Group `sample_rows` by `key_fn` (e.g. `lambda r: r.source_file` or
    `lambda r: r.quadrant`) and compute an independent
    `GroundTruthComparisonReport` per group -- shared by the report's
    by-file and by-quadrant breakdowns."""
    groups: dict = {}
    for row in sample_rows:
        groups.setdefault(key_fn(row), []).append(row)
    return {key: compute_ground_truth_report(rows, ground_truth_labels) for key, rows in groups.items()}


def load_rationale_file(path: str | Path) -> tuple[str, dict[int, tuple[bool, str]]]:
    """Parse one `<stem>_rationale.jsonl` LLM-baseline file into
    (source_file, {row_id: (match, rationale)}) -- like
    `validate_laya_filtering.load_baseline`, but keeps the rationale text
    too, since the final report cites it alongside each classifier's
    verdict."""
    path = Path(path)
    baseline: dict[int, tuple[bool, str]] = {}
    source_file: str | None = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            source_file = record["source_file"]
            baseline[int(record["row_id"])] = (bool(record["match"]), str(record.get("rationale", "")))
    if source_file is None:
        raise ValueError(f"{path} has no rationale records")
    return source_file, baseline


def load_laya_predictions(path: str | Path) -> dict[str, dict[int, float]]:
    """Parse `logs/laya_predictions.jsonl` (written by
    `validate_laya_filtering.write_score_dump`) into
    {source_file: {row_id: score}}."""
    scores: dict[str, dict[int, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            scores.setdefault(record["source_file"], {})[int(record["row_id"])] = float(record["score"])
    return scores


def compute_quadrant_pool_sizes(
    run_config: RunConfig, laya_scores_path: str | Path, threshold: float
) -> dict[Quadrant, int]:
    """The true corpus-wide quadrant pool sizes the ground-truth sample was
    stratified from: every `logs/*_rationale.jsonl` baseline bucketed against
    laya scores at `threshold` via `build_quadrant_pools`. Shared by
    `cmd_report` (recomputing pool sizes for its own population reweighting)
    and `scripts/build_laya_redesign_report.py` (issue #8, reweighting each
    variant's scores against this same corpus-wide stratification)."""
    logs_dir = run_config.logs_dir_resolved
    baselines: dict[str, dict[int, bool]] = {}
    for path in sorted(logs_dir.glob("*_rationale.jsonl")):
        source_file, records = load_rationale_file(path)
        baselines[source_file] = {row_id: match for row_id, (match, _rationale) in records.items()}
    scores = load_laya_predictions(laya_scores_path)
    pools = build_quadrant_pools(baselines, scores, threshold)
    return {quadrant: sum(len(rows) for rows in by_file.values()) for quadrant, by_file in pools.items()}


def append_ground_truth_labels(path: str | Path, records: list[dict]) -> None:
    """Append one batch's {"sample_index", "ground_truth_match", "rationale"}
    records as JSON lines. Called once per collected labeling-subagent batch,
    not once at the very end, so a crash partway through labeling doesn't
    lose already-collected batches."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def load_ground_truth_labels(path: str | Path) -> dict[int, tuple[bool, str]]:
    """{sample_index: (ground_truth_match, rationale)}, read back from
    everything `append_ground_truth_labels` has written so far."""
    labels: dict[int, tuple[bool, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            labels[int(record["sample_index"])] = (
                bool(record["ground_truth_match"]),
                str(record["rationale"]),
            )
    return labels


def blinded_batch_payload(rows: list[SampleRow]) -> list[dict]:
    """The ONLY fields a labeling subagent's prompt may ever be built from --
    never laya_match/laya_score/llm_match/llm_rationale/quadrant, so a
    ground-truth judgment can't anchor on either system's prior verdict."""
    return [{"sample_index": r.sample_index, "product_details_text": r.product_details_text} for r in rows]


def _resolve(run_config: RunConfig, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else run_config.project_root / p


def _print_sample_summary(
    sample_rows: list[SampleRow],
    boosted_files: frozenset[str],
    min_rows_per_boosted_file: int,
) -> None:
    counts: dict[str, dict[Quadrant, int]] = {}
    for row in sample_rows:
        counts.setdefault(row.source_file, {})[row.quadrant] = (
            counts.setdefault(row.source_file, {}).get(row.quadrant, 0) + 1
        )
    print(f"\n{'file':40s} " + " ".join(f"{q:>20s}" for q in _ALL_QUADRANTS) + "   total")
    for source_file in sorted(counts):
        by_quadrant = counts[source_file]
        total = sum(by_quadrant.values())
        marker = " *" if source_file in boosted_files and total < min_rows_per_boosted_file else ""
        row_str = " ".join(f"{by_quadrant.get(q, 0):>20d}" for q in _ALL_QUADRANTS)
        print(f"{source_file:40s} {row_str}   {total}{marker}")
    print(f"\nTotal sampled rows: {len(sample_rows)} (files marked * are below their boosted-file floor)")


def cmd_sample(args: argparse.Namespace) -> int:
    run_config = RunConfig.load(args.config, project_root=args.project_root)
    scores_path = _resolve(run_config, args.laya_scores_path)

    sample_rows = build_ground_truth_sample(run_config, scores_path, threshold=args.threshold)

    sample_path = _resolve(run_config, args.sample_path)
    write_ground_truth_sample(sample_path, sample_rows)

    _print_sample_summary(sample_rows, WORST_AGREEMENT_FILES, MIN_ROWS_PER_BOOSTED_FILE)
    print(f"\nSample written to {sample_path}")
    print(
        "Next: dispatch parallel labeling subagents over blinded_batch_payload() batches of "
        f"this sample, appending each batch's results via append_ground_truth_labels() to "
        f"{args.labels_path}, then run the 'report' subcommand."
    )
    return 0


def _metrics_row(label: str, m: dict) -> str:
    return f"| {label} | {m['accuracy']:.1%} | {m['precision']:.1%} | {m['recall']:.1%} | {m['f1']:.1%} | {m['total']:.0f} |"


def _metrics_table(rows: list[str]) -> list[str]:
    return ["| classifier | accuracy | precision | recall | f1 | n |", "|---|---:|---:|---:|---:|---:|", *rows]


def _render_markdown(report: dict) -> str:
    lines = [
        "# Ground-truth validation report",
        "",
        f"Threshold: {report['threshold']}  |  Labeled sample size: {report['sample_size']}"
        + (f"  |  Excluded (no label yet): {report['unlabeled_excluded']}" if report["unlabeled_excluded"] else ""),
        "",
        "## Overall -- raw sample",
        "",
        *_metrics_table(
            [
                _metrics_row("laya", report["overall"]["laya"]),
                _metrics_row("LLM baseline", report["overall"]["llm_baseline"]),
            ]
        ),
        "",
        "## Overall -- population-reweighted",
        "",
        "The sample deliberately oversamples laya/LLM disagreement rows relative to their "
        "true share of the full corpus; this corrects for that so the numbers below "
        "estimate accuracy across the whole dataset, not just this sample.",
        "",
        *_metrics_table(
            [
                _metrics_row("laya", report["overall"]["population_weighted"]["laya"]),
                _metrics_row("LLM baseline", report["overall"]["population_weighted"]["llm_baseline"]),
            ]
        ),
        "",
        "## By quadrant",
        "",
        "Answers: of the rows where laya and the LLM baseline disagreed, who was actually right?",
        "",
        *_metrics_table(
            [
                _metrics_row(f"{quadrant} -- laya", metrics["laya"])
                for quadrant, metrics in sorted(report["by_quadrant"].items())
            ]
            + [
                _metrics_row(f"{quadrant} -- LLM baseline", metrics["llm_baseline"])
                for quadrant, metrics in sorted(report["by_quadrant"].items())
            ]
        ),
        "",
        "## By file",
        "",
        *_metrics_table(
            [
                _metrics_row(f"{source_file} -- laya", metrics["laya"])
                for source_file, metrics in sorted(report["by_file"].items())
            ]
            + [
                _metrics_row(f"{source_file} -- LLM baseline", metrics["llm_baseline"])
                for source_file, metrics in sorted(report["by_file"].items())
            ]
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    run_config = RunConfig.load(args.config, project_root=args.project_root)
    sample_path = _resolve(run_config, args.sample_path)
    labels_path = _resolve(run_config, args.labels_path)

    sample_rows = load_ground_truth_sample(sample_path)
    if not sample_rows:
        print(f"No sample rows found at {sample_path}; run the 'sample' subcommand first.", file=sys.stderr)
        return 1

    ground_truth_labels = load_ground_truth_labels(labels_path)
    labeled_rows = [r for r in sample_rows if r.sample_index in ground_truth_labels]
    unlabeled_excluded = len(sample_rows) - len(labeled_rows)
    if unlabeled_excluded:
        print(
            f"WARN: {unlabeled_excluded} sampled rows have no ground-truth label yet -- "
            "excluding them from this report.",
            file=sys.stderr,
        )
    if not labeled_rows:
        print("No labeled rows to report on.", file=sys.stderr)
        return 1

    overall = compute_ground_truth_report(labeled_rows, ground_truth_labels)
    by_file = compute_breakdowns(labeled_rows, ground_truth_labels, key_fn=lambda r: r.source_file)
    by_quadrant = compute_breakdowns(labeled_rows, ground_truth_labels, key_fn=lambda r: r.quadrant)

    # Recompute the true corpus-wide quadrant pool sizes (cheap: no xlsx reads,
    # just the same rationale + laya-score files `sample` already read) so
    # population_weighted_overall can correct for the sample's oversampling.
    scores_path = _resolve(run_config, args.laya_scores_path)
    pool_sizes = compute_quadrant_pool_sizes(run_config, scores_path, args.threshold)
    population_weighted = population_weighted_overall(labeled_rows, ground_truth_labels, pool_sizes)

    report = {
        "threshold": args.threshold,
        "sample_size": len(labeled_rows),
        "unlabeled_excluded": unlabeled_excluded,
        "overall": {
            "laya": asdict(overall.laya),
            "llm_baseline": asdict(overall.llm_baseline),
            "population_weighted": {
                "laya": asdict(population_weighted.laya),
                "llm_baseline": asdict(population_weighted.llm_baseline),
            },
        },
        "by_file": {f: {"laya": asdict(r.laya), "llm_baseline": asdict(r.llm_baseline)} for f, r in by_file.items()},
        "by_quadrant": {
            q: {"laya": asdict(r.laya), "llm_baseline": asdict(r.llm_baseline)} for q, r in by_quadrant.items()
        },
        "quadrant_pool_sizes": pool_sizes,
        "boosted_files": sorted(WORST_AGREEMENT_FILES),
    }

    report_path = _resolve(run_config, args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    markdown_path = _resolve(run_config, args.markdown_path)
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")

    print(f"Ground-truth comparison written to {report_path} and {markdown_path}")
    print(
        f"\nOverall (raw sample, n={overall.laya.total:.0f}): "
        f"laya accuracy={overall.laya.accuracy:.1%}, LLM baseline accuracy={overall.llm_baseline.accuracy:.1%}"
    )
    print(
        "Overall (population-reweighted): "
        f"laya accuracy={population_weighted.laya.accuracy:.1%}, "
        f"LLM baseline accuracy={population_weighted.llm_baseline.accuracy:.1%}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("sample", help="Build the stratified ground-truth sample.")
    sample_parser.add_argument("--config", default="config/run_config.yaml")
    sample_parser.add_argument("--project-root", default=None)
    sample_parser.add_argument("--laya-scores-path", default="logs/laya_predictions.jsonl")
    sample_parser.add_argument("--sample-path", default="logs/ground_truth_sample.jsonl")
    sample_parser.add_argument("--labels-path", default="logs/ground_truth_labels.jsonl")
    sample_parser.add_argument("--threshold", type=float, default=DEFAULT_LAYA_THRESHOLD)

    report_parser = subparsers.add_parser(
        "report", help="Compare laya and the LLM baseline against collected ground-truth labels."
    )
    report_parser.add_argument("--config", default="config/run_config.yaml")
    report_parser.add_argument("--project-root", default=None)
    report_parser.add_argument("--laya-scores-path", default="logs/laya_predictions.jsonl")
    report_parser.add_argument("--sample-path", default="logs/ground_truth_sample.jsonl")
    report_parser.add_argument("--labels-path", default="logs/ground_truth_labels.jsonl")
    report_parser.add_argument("--report-path", default="logs/ground_truth_validation_report.json")
    report_parser.add_argument("--markdown-path", default="logs/ground_truth_validation_report.md")
    report_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_LAYA_THRESHOLD,
        help="Must match the threshold the 'sample' subcommand used, for correct quadrant pool sizes.",
    )

    args = parser.parse_args(argv)
    if args.command == "sample":
        return cmd_sample(args)
    return cmd_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
