"""Go/no-go validator for replacing LLM-based Stage 1 relevance judging with the
local `laya-mlx` classifier (`agenticxl.filtering.laya_client.LayaClient`).

Loads every `logs/*_rationale.jsonl` baseline (real prior LLM verdicts, one line
per row) plus the original row data those verdicts judged (from the matching
file in `input_dir`), runs `LayaClient` over the same rows, and reports
agreement % against the baseline -- with false negatives (classifier says
no-match where the LLM said match, i.e. the classifier would silently drop a
real record) tracked separately from false positives, since under-filtering is
the failure mode this validation exists to catch (see CLAUDE.md / GitHub issue
#1's "Further Notes").

The match/no-match threshold on the classifier's `noul` score is not assumed --
this script sweeps a range of candidate thresholds against the *raw* score
(computed once per row) and reports the empirically best one, per issue #1's
"threshold derived empirically from the validation run" decision.

Usage:
    .venv/bin/python scripts/validate_laya_filtering.py
    .venv/bin/python scripts/validate_laya_filtering.py --sample-per-file 200
    .venv/bin/python scripts/validate_laya_filtering.py --config config/run_config.yaml \\
        --report-path logs/laya_validation_report.json

Exit code 0 means the run completed (a go/no-go verdict is always printed and
written to the report regardless of outcome); exit code 1 means the script
itself failed to run (bad config, no baseline files, etc.).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from agenticxl.config import RunConfig
from agenticxl.excel.reader import load_main_sheet
from agenticxl.filtering.laya_client import LayaClient

# A false negative -- the classifier silently drops a row the LLM baseline judged
# relevant -- is weighted more heavily than a false positive (an extra row that
# still gets caught by downstream review), because it's the failure mode that
# degrades data quality without anyone noticing. See issue #1's "Further Notes".
FALSE_NEGATIVE_WEIGHT = 2.0
GO_NO_GO_AGREEMENT_THRESHOLD = 0.95
DEFAULT_THRESHOLD_CANDIDATES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


@dataclass(frozen=True)
class AgreementReport:
    total: int
    agree: int
    agreement_rate: float
    false_negatives: int
    false_negative_rate: float
    false_positives: int
    false_positive_rate: float
    weighted_agreement_rate: float


def compute_agreement(
    baseline: dict[int, bool],
    classifier: dict[int, bool],
    fn_weight: float = FALSE_NEGATIVE_WEIGHT,
) -> AgreementReport:
    """Pure comparison of two row_id->match maps that must cover the same row_ids.

    `weighted_agreement_rate` counts each false negative as `fn_weight` errors
    (default 2x a false positive) against the plain row total -- it is a ranking/
    gating signal, not a probability, and can go negative when false negatives
    dominate a small sample.
    """
    if set(baseline) != set(classifier):
        only_baseline = sorted(set(baseline) - set(classifier))
        only_classifier = sorted(set(classifier) - set(baseline))
        raise ValueError(
            "baseline and classifier must cover the same row_ids: "
            f"only in baseline={only_baseline[:10]}, only in classifier={only_classifier[:10]}"
        )

    total = len(baseline)
    if total == 0:
        return AgreementReport(0, 0, 1.0, 0, 0.0, 0, 0.0, 1.0)

    baseline_true = [row_id for row_id, match in baseline.items() if match]
    baseline_false = [row_id for row_id, match in baseline.items() if not match]

    false_negatives = sum(1 for row_id in baseline_true if not classifier[row_id])
    false_positives = sum(1 for row_id in baseline_false if classifier[row_id])
    agree = total - false_negatives - false_positives

    fn_rate = false_negatives / len(baseline_true) if baseline_true else 0.0
    fp_rate = false_positives / len(baseline_false) if baseline_false else 0.0
    weighted_agreement_rate = 1.0 - (fn_weight * false_negatives + false_positives) / total

    return AgreementReport(
        total=total,
        agree=agree,
        agreement_rate=agree / total,
        false_negatives=false_negatives,
        false_negative_rate=fn_rate,
        false_positives=false_positives,
        false_positive_rate=fp_rate,
        weighted_agreement_rate=weighted_agreement_rate,
    )


def load_baseline(path: str | Path) -> tuple[str, dict[int, bool]]:
    """Parse one `<stem>_rationale.jsonl` file into (source_file, {row_id: match})."""
    path = Path(path)
    baseline: dict[int, bool] = {}
    source_file: str | None = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            source_file = record["source_file"]
            baseline[int(record["row_id"])] = bool(record["match"])
    if source_file is None:
        raise ValueError(f"{path} has no rationale records")
    return source_file, baseline


def iter_rationale_files(logs_dir: Path) -> list[Path]:
    return sorted(logs_dir.glob("*_rationale.jsonl"))


def _evenly_spaced(items: list, n: int) -> list:
    n = min(n, len(items))
    if n <= 0 or not items:
        return []
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def stratified_sample(
    rows: list[tuple[int, str]], baseline: dict[int, bool], sample_size: int
) -> list[tuple[int, str]]:
    """Deterministically sample `rows` down to ~`sample_size`, keeping the baseline's
    true/false proportion (evenly spaced within each class) so a validation run
    capped for runtime doesn't skew toward the (usually much larger) non-match class."""
    if len(rows) <= sample_size:
        return rows
    true_rows = [r for r in rows if baseline[r[0]]]
    false_rows = [r for r in rows if not baseline[r[0]]]
    total = len(rows)
    n_true = round(sample_size * len(true_rows) / total) if true_rows else 0
    n_true = min(n_true, len(true_rows))
    n_false = min(sample_size - n_true, len(false_rows))
    sampled_ids = {row_id for row_id, _ in _evenly_spaced(true_rows, n_true)}
    sampled_ids |= {row_id for row_id, _ in _evenly_spaced(false_rows, n_false)}
    return [r for r in rows if r[0] in sampled_ids]


def classify_rows(
    client: LayaClient, keywords: list[str], rows: list[tuple[int, str]], batch_size: int
) -> dict[int, float]:
    """Run every row through `client.judge_batch`, chunked by `batch_size`, and return
    the raw `noul` score per row_id (threshold-independent, so the sweep below never
    re-runs inference)."""
    scores: dict[int, float] = {}
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        for verdict in client.judge_batch(chunk, keywords, threshold=0.0):
            scores[verdict.row_id] = verdict.score
    return scores


def sweep_thresholds(
    per_file_data: dict[str, tuple[dict[int, bool], dict[int, float]]],
    thresholds: list[float],
) -> dict[float, AgreementReport]:
    """Aggregate agreement across every file at each candidate threshold, without
    re-running inference (uses the raw scores `classify_rows` already collected)."""
    results: dict[float, AgreementReport] = {}
    for threshold in thresholds:
        combined_baseline: dict[int, bool] = {}
        combined_classifier: dict[int, bool] = {}
        offset = 0
        for baseline, scores in per_file_data.values():
            # row_ids repeat across files, so offset them to keep the combined
            # aggregate keyed uniquely per (file, row_id) pair.
            for row_id, match in baseline.items():
                combined_baseline[offset + row_id] = match
            for row_id, score in scores.items():
                combined_classifier[offset + row_id] = score >= threshold
            offset += 10_000_000
        results[threshold] = compute_agreement(combined_baseline, combined_classifier)
    return results


def write_score_dump(
    path: str | Path,
    per_file_data: dict[str, tuple[dict[int, bool], dict[int, float]]],
    threshold: float,
    model_identifier: str,
    *,
    question_mode: str | None = None,
    payload_layout: str | None = None,
) -> Path:
    """Persist every row's raw laya score across every file classified this run
    (the full corpus when --sample-per-file is omitted) as one JSON line per
    row, sorted by (source_file, row_id): laya's own output, independent of
    the LLM baseline it's compared against elsewhere in this script.

    `question_mode`/`payload_layout` are optional (issue #6/#8): when a caller
    passes them (`scripts/build_laya_redesign_report.py` scoring one of the four
    `LayaClient` variants against ground truth), each record also carries
    `question_mode`, `payload_layout`, and `threshold`, so dumps from different
    variants stay independently auditable. Omitted by every existing caller of
    this function, which keeps their output byte-identical to before."""
    if (question_mode is None) != (payload_layout is None):
        raise ValueError("question_mode and payload_layout must be given together, or not at all")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for source_file in sorted(per_file_data):
            _, scores = per_file_data[source_file]
            for row_id in sorted(scores):
                score = scores[row_id]
                record = {
                    "source_file": source_file,
                    "row_id": row_id,
                    "score": score,
                    "match": score >= threshold,
                    "model": model_identifier,
                }
                if question_mode is not None:
                    record["question_mode"] = question_mode
                    record["payload_layout"] = payload_layout
                    record["threshold"] = threshold
                f.write(json.dumps(record) + "\n")
    return path


def _print_report(source_file: str, report: AgreementReport) -> None:
    print(
        f"  {source_file}: agreement={report.agreement_rate:.1%} "
        f"({report.agree}/{report.total}), "
        f"FN={report.false_negatives} ({report.false_negative_rate:.1%} of baseline matches), "
        f"FP={report.false_positives} ({report.false_positive_rate:.1%} of baseline non-matches)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/run_config.yaml", help="Path to run_config.yaml")
    parser.add_argument("--project-root", default=None, help="Project root (default: repo root)")
    parser.add_argument(
        "--sample-per-file",
        type=int,
        default=None,
        help="Cap rows classified per file (stratified by baseline match/no-match) "
        "to bound runtime; omit to run the full baseline set.",
    )
    parser.add_argument(
        "--report-path",
        default="logs/laya_validation_report.json",
        help="Where to write the JSON validation report.",
    )
    parser.add_argument(
        "--scores-path",
        default="logs/laya_predictions.jsonl",
        help="Where to write laya's raw per-row scores (its own output, independent of "
        "the LLM baseline), one JSON line per row.",
    )
    args = parser.parse_args(argv)

    run_config = RunConfig.load(args.config, project_root=args.project_root)
    logs_dir = run_config.logs_dir_resolved
    rationale_files = iter_rationale_files(logs_dir)
    if not rationale_files:
        print(f"No *_rationale.jsonl baseline files found in {logs_dir}", file=sys.stderr)
        return 1

    print(f"Loading laya-mlx checkpoint ({run_config.relevance_batch_size}-row native batches)...")
    client = LayaClient(batch_size=run_config.relevance_batch_size)

    per_file_data: dict[str, tuple[dict[int, bool], dict[int, float]]] = {}
    for path in rationale_files:
        source_file, baseline = load_baseline(path)
        input_path = run_config.input_dir_resolved / source_file
        if not input_path.exists():
            print(f"WARN: skipping {source_file}: not found at {input_path}", file=sys.stderr)
            continue

        sheet = load_main_sheet(input_path)
        missing = [row_id for row_id in baseline if row_id >= len(sheet.rows)]
        if missing:
            print(
                f"WARN: skipping {source_file}: baseline references row_id(s) {missing[:5]} "
                f"but the file only has {len(sheet.rows)} rows (file changed since the baseline "
                "was recorded)",
                file=sys.stderr,
            )
            continue

        rows = [
            (row_id, str(sheet.rows[row_id].get(sheet.product_details_column) or ""))
            for row_id in sorted(baseline)
        ]
        if args.sample_per_file is not None:
            rows = stratified_sample(rows, baseline, args.sample_per_file)
            baseline = {row_id: baseline[row_id] for row_id, _ in rows}

        print(f"Classifying {source_file}: {len(rows)} rows...")
        scores = classify_rows(client, run_config.keywords, rows, run_config.relevance_batch_size)
        per_file_data[source_file] = (baseline, scores)

    if not per_file_data:
        print("No files could be validated (all baselines skipped).", file=sys.stderr)
        return 1

    sweep = sweep_thresholds(per_file_data, DEFAULT_THRESHOLD_CANDIDATES)
    best_threshold = max(sweep, key=lambda t: sweep[t].weighted_agreement_rate)
    best_report = sweep[best_threshold]

    print("\nPer-file agreement at the empirically best threshold:")
    per_file_reports = {}
    for source_file, (baseline, scores) in per_file_data.items():
        classifier = {row_id: score >= best_threshold for row_id, score in scores.items()}
        report = compute_agreement(baseline, classifier)
        per_file_reports[source_file] = report
        _print_report(source_file, report)

    print("\nThreshold sweep (overall, weighted against false negatives):")
    for threshold in DEFAULT_THRESHOLD_CANDIDATES:
        r = sweep[threshold]
        marker = " <-- best" if threshold == best_threshold else ""
        print(
            f"  threshold={threshold:.2f}: agreement={r.agreement_rate:.1%}, "
            f"weighted={r.weighted_agreement_rate:.1%}, FN_rate={r.false_negative_rate:.1%}, "
            f"FP_rate={r.false_positive_rate:.1%}{marker}"
        )

    go = best_report.agreement_rate >= GO_NO_GO_AGREEMENT_THRESHOLD and best_report.weighted_agreement_rate >= GO_NO_GO_AGREEMENT_THRESHOLD
    verdict = "GO" if go else "NO-GO"
    print(
        f"\n{verdict}: best threshold={best_threshold:.2f}, "
        f"overall agreement={best_report.agreement_rate:.1%}, "
        f"weighted agreement={best_report.weighted_agreement_rate:.1%} "
        f"(gate: >= {GO_NO_GO_AGREEMENT_THRESHOLD:.0%} on both, false negatives weighted "
        f"{FALSE_NEGATIVE_WEIGHT:.0f}x)"
    )

    report_path = Path(args.report_path)
    if not report_path.is_absolute():
        report_path = run_config.project_root / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "verdict": verdict,
                "best_threshold": best_threshold,
                "gate": GO_NO_GO_AGREEMENT_THRESHOLD,
                "false_negative_weight": FALSE_NEGATIVE_WEIGHT,
                "overall": asdict(best_report),
                "threshold_sweep": {str(t): asdict(r) for t, r in sweep.items()},
                "per_file": {f: asdict(r) for f, r in per_file_reports.items()},
                "sample_per_file": args.sample_per_file,
                "model": client.model_identifier,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nFull report written to {report_path}")

    scores_path = Path(args.scores_path)
    if not scores_path.is_absolute():
        scores_path = run_config.project_root / scores_path
    write_score_dump(scores_path, per_file_data, best_threshold, client.model_identifier)
    print(f"Per-row laya scores written to {scores_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
