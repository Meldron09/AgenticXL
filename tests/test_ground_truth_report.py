from __future__ import annotations

import json

import pytest

from agenticxl.config import RunConfig
from helpers import write_synthetic_workbook

from scripts.build_ground_truth_report import (
    ClassifierMetrics,
    SampleRow,
    allocate_quadrant_targets,
    append_ground_truth_labels,
    blinded_batch_payload,
    build_ground_truth_sample,
    build_quadrant_pools,
    classify_quadrant,
    compute_breakdowns,
    compute_classifier_metrics,
    compute_ground_truth_report,
    compute_quadrant_pool_sizes,
    load_ground_truth_labels,
    load_ground_truth_sample,
    load_laya_predictions,
    load_rationale_file,
    population_weighted_metrics,
    population_weighted_overall,
    write_ground_truth_sample,
)


def test_classify_quadrant_both_match():
    assert classify_quadrant(llm_match=True, laya_match=True) == "both_match"


def test_classify_quadrant_both_no_match():
    assert classify_quadrant(llm_match=False, laya_match=False) == "both_no_match"


def test_classify_quadrant_laya_false_negative():
    # LLM said relevant, laya said not -- laya under-filters
    assert classify_quadrant(llm_match=True, laya_match=False) == "laya_false_negative"


def test_classify_quadrant_laya_false_positive():
    # LLM said not relevant, laya said relevant
    assert classify_quadrant(llm_match=False, laya_match=True) == "laya_false_positive"


def test_build_quadrant_pools_buckets_rows_by_file_and_quadrant():
    baselines = {
        "a.xlsx": {0: True, 1: True, 2: False, 3: False},
        "b.xlsx": {5: True, 6: False},
    }
    scores = {
        "a.xlsx": {0: 0.9, 1: 0.1, 2: 0.9, 3: 0.1},  # row0 both_match, row1 FN, row2 FP, row3 both_no_match
        "b.xlsx": {5: 0.9, 6: 0.1},  # row5 both_match, row6 both_no_match
    }

    pools = build_quadrant_pools(baselines, scores, threshold=0.5)

    assert pools == {
        "both_match": {"a.xlsx": [0], "b.xlsx": [5]},
        "both_no_match": {"a.xlsx": [3], "b.xlsx": [6]},
        "laya_false_negative": {"a.xlsx": [1]},
        "laya_false_positive": {"a.xlsx": [2]},
    }


def test_build_quadrant_pools_sorts_row_ids_within_a_file():
    baselines = {"a.xlsx": {3: True, 1: True, 2: True}}
    scores = {"a.xlsx": {3: 0.9, 1: 0.9, 2: 0.9}}

    pools = build_quadrant_pools(baselines, scores, threshold=0.5)

    assert pools["both_match"]["a.xlsx"] == [1, 2, 3]


def test_build_quadrant_pools_skips_row_missing_a_laya_score():
    # row 1 is in the LLM baseline but was never scored by laya (e.g. a partial
    # scores dump) -- must be skipped, never fabricated as a match or no-match.
    baselines = {"a.xlsx": {0: True, 1: True}}
    scores = {"a.xlsx": {0: 0.9}}

    pools = build_quadrant_pools(baselines, scores, threshold=0.5)

    assert pools["both_match"] == {"a.xlsx": [0]}
    all_pooled_rows = {rid for by_file in pools.values() for rows in by_file.values() for rid in rows}
    assert 1 not in all_pooled_rows


def _sample_row(**overrides) -> SampleRow:
    defaults = dict(
        sample_index=0,
        source_file="a.xlsx",
        row_id=0,
        quadrant="both_match",
        product_details_text="a scissor lift",
        laya_score=0.9,
        laya_match=True,
        llm_match=True,
        llm_rationale="scissor lift matches 'scissor'",
    )
    defaults.update(overrides)
    return SampleRow(**defaults)


def test_blinded_batch_payload_exposes_only_sample_index_and_text():
    rows = [
        _sample_row(sample_index=0, product_details_text="a scissor lift"),
        _sample_row(sample_index=1, product_details_text="a USB cable", llm_match=False, laya_match=False),
    ]

    payload = blinded_batch_payload(rows)

    assert payload == [
        {"sample_index": 0, "product_details_text": "a scissor lift"},
        {"sample_index": 1, "product_details_text": "a USB cable"},
    ]
    for item in payload:
        assert set(item.keys()) == {"sample_index", "product_details_text"}


def test_compute_classifier_metrics_perfect_agreement():
    ground_truth = {0: True, 1: False, 2: True}
    predictions = {0: True, 1: False, 2: True}

    metrics = compute_classifier_metrics(ground_truth, predictions)

    assert metrics == ClassifierMetrics(
        total=3,
        true_positives=2,
        false_positives=0,
        false_negatives=0,
        true_negatives=1,
        accuracy=1.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
    )


def test_compute_classifier_metrics_mixed_confusion_matrix():
    # ground truth: 3 relevant (0,1,2), 3 not relevant (3,4,5)
    # predictions:  0 -> TP, 1 -> FN, 2 -> TP, 3 -> TN, 4 -> FP, 5 -> TN
    ground_truth = {0: True, 1: True, 2: True, 3: False, 4: False, 5: False}
    predictions = {0: True, 1: False, 2: True, 3: False, 4: True, 5: False}

    metrics = compute_classifier_metrics(ground_truth, predictions)

    assert metrics.total == 6
    assert metrics.true_positives == 2
    assert metrics.false_negatives == 1
    assert metrics.false_positives == 1
    assert metrics.true_negatives == 2
    assert metrics.accuracy == pytest.approx(4 / 6)
    assert metrics.precision == pytest.approx(2 / 3)  # 2 TP / (2 TP + 1 FP)
    assert metrics.recall == pytest.approx(2 / 3)  # 2 TP / (2 TP + 1 FN)
    assert metrics.f1 == pytest.approx(2 / 3)  # precision == recall here


def test_compute_classifier_metrics_recall_is_zero_when_ground_truth_has_no_positives():
    ground_truth = {0: False, 1: False}
    predictions = {0: True, 1: False}  # one false positive, no actual positives to recall

    metrics = compute_classifier_metrics(ground_truth, predictions)

    assert metrics.recall == 0.0
    assert metrics.precision == 0.0  # the one predicted positive was wrong


def test_compute_classifier_metrics_precision_is_zero_when_no_predicted_positives():
    ground_truth = {0: True, 1: False}
    predictions = {0: False, 1: False}  # no predicted positives at all

    metrics = compute_classifier_metrics(ground_truth, predictions)

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0  # the one actual positive was missed


def test_compute_classifier_metrics_raises_on_row_id_mismatch():
    with pytest.raises(ValueError):
        compute_classifier_metrics({0: True, 1: False}, {0: True})


def test_compute_ground_truth_report_scores_each_classifier_against_ground_truth_not_each_other():
    # sample_index 0: ground truth relevant, laya right, LLM wrong (a laya_false_negative
    #   quadrant row where laya turns out to actually be correct)
    # sample_index 1: ground truth not relevant, laya wrong, LLM right
    sample_rows = [
        _sample_row(sample_index=0, laya_match=True, llm_match=False, quadrant="laya_false_negative"),
        _sample_row(sample_index=1, laya_match=True, llm_match=False, quadrant="laya_false_positive"),
    ]
    ground_truth_labels = {
        0: (True, "relevant"),
        1: (False, "not relevant"),
    }

    report = compute_ground_truth_report(sample_rows, ground_truth_labels)

    # laya predicted [True, True] vs ground truth [True, False] -> 1 TP, 1 FP
    assert report.laya.true_positives == 1
    assert report.laya.false_positives == 1
    assert report.laya.accuracy == 0.5
    # LLM predicted [False, False] vs ground truth [True, False] -> 1 FN, 1 TN
    assert report.llm_baseline.false_negatives == 1
    assert report.llm_baseline.true_negatives == 1
    assert report.llm_baseline.accuracy == 0.5


def test_compute_breakdowns_groups_by_key_fn_and_scores_each_group_independently():
    sample_rows = [
        _sample_row(sample_index=0, source_file="a.xlsx", laya_match=True, llm_match=True),
        _sample_row(sample_index=1, source_file="a.xlsx", laya_match=False, llm_match=False),
        _sample_row(sample_index=2, source_file="b.xlsx", laya_match=True, llm_match=False),
    ]
    ground_truth_labels = {
        0: (True, "r0"),
        1: (True, "r1"),  # a.xlsx: laya/LLM both wrong on this one
        2: (True, "r2"),  # b.xlsx: laya right, LLM wrong
    }

    breakdowns = compute_breakdowns(sample_rows, ground_truth_labels, key_fn=lambda r: r.source_file)

    assert set(breakdowns) == {"a.xlsx", "b.xlsx"}
    assert breakdowns["a.xlsx"].laya.total == 2
    assert breakdowns["a.xlsx"].laya.accuracy == 0.5  # right on row0, wrong on row1
    assert breakdowns["b.xlsx"].laya.total == 1
    assert breakdowns["b.xlsx"].laya.accuracy == 1.0
    assert breakdowns["b.xlsx"].llm_baseline.accuracy == 0.0


def test_ground_truth_labels_round_trip_across_multiple_appended_batches(tmp_path):
    path = tmp_path / "ground_truth_labels.jsonl"

    append_ground_truth_labels(
        path,
        [
            {"sample_index": 0, "ground_truth_match": True, "rationale": "scissor lift"},
            {"sample_index": 1, "ground_truth_match": False, "rationale": "unrelated cable"},
        ],
    )
    append_ground_truth_labels(
        path,
        [{"sample_index": 2, "ground_truth_match": True, "rationale": "boom lift"}],
    )

    labels = load_ground_truth_labels(path)

    assert labels == {
        0: (True, "scissor lift"),
        1: (False, "unrelated cable"),
        2: (True, "boom lift"),
    }


def _pool_of_size(n: int, start: int = 0) -> list[int]:
    return list(range(start, start + n))


def test_allocate_quadrant_targets_totals_match_target_when_pools_are_ample():
    pools = {
        "both_match": {"a.xlsx": _pool_of_size(100), "b.xlsx": _pool_of_size(50), "c.xlsx": _pool_of_size(20)},
    }

    allocation = allocate_quadrant_targets(pools, {"both_match": 30})

    total_allocated = sum(len(rows) for rows in allocation["both_match"].values())
    assert total_allocated == 30


def test_allocate_quadrant_targets_never_exceeds_a_files_pool_size():
    pools = {"both_match": {"a.xlsx": _pool_of_size(5), "b.xlsx": _pool_of_size(100)}}

    allocation = allocate_quadrant_targets(pools, {"both_match": 50})

    assert len(allocation["both_match"]["a.xlsx"]) <= 5
    assert sum(len(rows) for rows in allocation["both_match"].values()) == 50


def test_allocate_quadrant_targets_boosted_file_gets_larger_share_than_equal_sized_unboosted_file():
    pools = {"both_match": {"boosted.xlsx": _pool_of_size(50), "plain.xlsx": _pool_of_size(50, start=1000)}}

    allocation = allocate_quadrant_targets(
        pools, {"both_match": 20}, boosted_files={"boosted.xlsx"}, boost_weight=3.0
    )

    assert len(allocation["both_match"]["boosted.xlsx"]) > len(allocation["both_match"]["plain.xlsx"])


def test_allocate_quadrant_targets_tops_up_boosted_file_to_the_floor():
    # A small boosted file whose proportional share would be well under the floor.
    pools = {
        "both_match": {"tiny.xlsx": _pool_of_size(3), "big.xlsx": _pool_of_size(1000, start=10_000)},
        "both_no_match": {"tiny.xlsx": _pool_of_size(2, start=100), "big.xlsx": _pool_of_size(1000, start=20_000)},
    }
    quadrant_targets = {"both_match": 10, "both_no_match": 10}

    allocation = allocate_quadrant_targets(
        pools,
        quadrant_targets,
        boosted_files={"tiny.xlsx"},
        boost_weight=3.0,
        min_rows_per_boosted_file=5,
    )

    total_for_tiny = sum(len(by_file.get("tiny.xlsx", [])) for by_file in allocation.values())
    assert total_for_tiny >= 5
    # tiny.xlsx only has 5 rows total across both quadrants (3 + 2), so the floor
    # of 5 exhausts its entire pool -- never fabricated beyond what exists.
    assert total_for_tiny == 5


def test_allocate_quadrant_targets_is_deterministic():
    pools = {"both_match": {"a.xlsx": _pool_of_size(37), "b.xlsx": _pool_of_size(41, start=1000)}}
    quadrant_targets = {"both_match": 17}

    first = allocate_quadrant_targets(pools, quadrant_targets)
    second = allocate_quadrant_targets(pools, quadrant_targets)

    assert first == second


def test_population_weighted_overall_corrects_for_oversampling_a_disagreement_quadrant():
    # True corpus: 900 rows where laya/LLM both correctly agree "match" (both_match),
    # 100 rows where laya wrongly says no-match but the LLM (and ground truth) say
    # match (laya_false_negative). The sample deliberately oversamples the rare
    # disagreement quadrant: only 9 of the 900 both_match rows, but 91 of the 100
    # laya_false_negative rows.
    sample_rows = [
        _sample_row(sample_index=i, quadrant="both_match", laya_match=True, llm_match=True)
        for i in range(9)
    ] + [
        _sample_row(sample_index=100 + i, quadrant="laya_false_negative", laya_match=False, llm_match=True)
        for i in range(91)
    ]
    ground_truth_labels = {r.sample_index: (True, "relevant") for r in sample_rows}
    quadrant_pool_sizes = {
        "both_match": 900,
        "both_no_match": 0,
        "laya_false_negative": 100,
        "laya_false_positive": 0,
    }

    report = population_weighted_overall(sample_rows, ground_truth_labels, quadrant_pool_sizes)

    # Reweighted to the true 900:100 population split, laya's real accuracy is ~90%
    # (it's only wrong on the true false-negative population, not the oversampled
    # sample fraction) -- a flat/unweighted accuracy over this sample would read
    # ~9% instead, since the sample is dominated by disagreement rows.
    assert report.laya.accuracy == pytest.approx(0.9)
    assert report.laya.recall == pytest.approx(0.9)
    assert report.laya.precision == pytest.approx(1.0)
    # LLM predicted "match" (correctly) on every row in both quadrants here.
    assert report.llm_baseline.accuracy == pytest.approx(1.0)


def test_load_rationale_file_keeps_source_file_match_and_rationale(tmp_path):
    path = tmp_path / "x_rationale.jsonl"
    lines = [
        {"source_file": "x.xlsx", "row_id": 0, "match": True, "rationale": "scissor lift"},
        {"source_file": "x.xlsx", "row_id": 1, "match": False, "rationale": "unrelated"},
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    source_file, baseline = load_rationale_file(path)

    assert source_file == "x.xlsx"
    assert baseline == {0: (True, "scissor lift"), 1: (False, "unrelated")}


def test_load_laya_predictions_nests_scores_by_source_file_and_row_id(tmp_path):
    path = tmp_path / "laya_predictions.jsonl"
    lines = [
        {"source_file": "a.xlsx", "row_id": 0, "score": 0.9, "match": True, "model": "m"},
        {"source_file": "a.xlsx", "row_id": 1, "score": 0.1, "match": False, "model": "m"},
        {"source_file": "b.xlsx", "row_id": 0, "score": 0.7, "match": True, "model": "m"},
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    scores = load_laya_predictions(path)

    assert scores == {"a.xlsx": {0: 0.9, 1: 0.1}, "b.xlsx": {0: 0.7}}


def test_compute_quadrant_pool_sizes_matches_build_quadrant_pools(tmp_path):
    (tmp_path / "logs").mkdir()
    rationale_lines = [
        {"source_file": "a.xlsx", "row_id": 0, "match": True, "rationale": "r0"},
        {"source_file": "a.xlsx", "row_id": 1, "match": True, "rationale": "r1"},
        {"source_file": "a.xlsx", "row_id": 2, "match": False, "rationale": "r2"},
    ]
    (tmp_path / "logs" / "a_rationale.jsonl").write_text(
        "\n".join(json.dumps(line) for line in rationale_lines) + "\n"
    )
    laya_lines = [
        {"source_file": "a.xlsx", "row_id": 0, "score": 0.9, "match": True, "model": "m"},  # both_match
        {"source_file": "a.xlsx", "row_id": 1, "score": 0.1, "match": False, "model": "m"},  # laya_false_negative
        {"source_file": "a.xlsx", "row_id": 2, "score": 0.1, "match": False, "model": "m"},  # both_no_match
    ]
    laya_predictions_path = tmp_path / "logs" / "laya_predictions.jsonl"
    laya_predictions_path.write_text("\n".join(json.dumps(line) for line in laya_lines) + "\n")

    run_config = RunConfig(project_root=tmp_path, input_dir="Documents", logs_dir="logs", keywords=["k"])

    pool_sizes = compute_quadrant_pool_sizes(run_config, laya_predictions_path, threshold=0.5)

    assert pool_sizes == {"both_match": 1, "both_no_match": 1, "laya_false_negative": 1, "laya_false_positive": 0}


def test_build_ground_truth_sample_end_to_end_on_a_synthetic_workbook(tmp_path):
    (tmp_path / "Documents").mkdir()
    (tmp_path / "logs").mkdir()

    write_synthetic_workbook(
        tmp_path / "Documents" / "a.xlsx",
        pre_header_rows=[],
        header=["ID", "PRODUCT DETAILS"],
        rows=[
            [0, "scissor lift A"],
            [1, "scissor lift B"],
            [2, "usb cable C"],
            [3, "usb cable D"],
        ],
    )
    rationale_lines = [
        {"source_file": "a.xlsx", "row_id": 0, "match": True, "rationale": "scissor lift A matches scissor"},
        {"source_file": "a.xlsx", "row_id": 1, "match": True, "rationale": "scissor lift B matches scissor"},
        {"source_file": "a.xlsx", "row_id": 2, "match": False, "rationale": "usb cable C is unrelated"},
        {"source_file": "a.xlsx", "row_id": 3, "match": False, "rationale": "usb cable D is unrelated"},
    ]
    (tmp_path / "logs" / "a_rationale.jsonl").write_text(
        "\n".join(json.dumps(line) for line in rationale_lines) + "\n"
    )
    laya_lines = [
        {"source_file": "a.xlsx", "row_id": 0, "score": 0.9, "match": True, "model": "m"},   # both_match
        {"source_file": "a.xlsx", "row_id": 1, "score": 0.1, "match": False, "model": "m"},  # laya_false_negative
        {"source_file": "a.xlsx", "row_id": 2, "score": 0.9, "match": True, "model": "m"},   # laya_false_positive
        {"source_file": "a.xlsx", "row_id": 3, "score": 0.1, "match": False, "model": "m"},  # both_no_match
    ]
    laya_predictions_path = tmp_path / "logs" / "laya_predictions.jsonl"
    laya_predictions_path.write_text("\n".join(json.dumps(line) for line in laya_lines) + "\n")

    run_config = RunConfig(
        project_root=tmp_path,
        input_dir="Documents",
        logs_dir="logs",
        keywords=["scissor"],
    )

    sample = build_ground_truth_sample(
        run_config,
        laya_predictions_path,
        threshold=0.5,
        quadrant_targets={
            "both_match": 1,
            "both_no_match": 1,
            "laya_false_negative": 1,
            "laya_false_positive": 1,
        },
        min_nonblank_header_cells=2,
    )

    assert len(sample) == 4
    by_row_id = {r.row_id: r for r in sample}
    assert by_row_id[0].quadrant == "both_match"
    assert by_row_id[0].product_details_text == "scissor lift A"
    assert by_row_id[0].laya_score == 0.9
    assert by_row_id[0].laya_match is True
    assert by_row_id[0].llm_match is True
    assert by_row_id[0].llm_rationale == "scissor lift A matches scissor"
    assert by_row_id[1].quadrant == "laya_false_negative"
    assert by_row_id[2].quadrant == "laya_false_positive"
    assert by_row_id[3].quadrant == "both_no_match"
    # sample_index assigned in sorted (source_file, row_id) order
    assert [r.sample_index for r in sorted(sample, key=lambda r: r.row_id)] == [0, 1, 2, 3]


def test_ground_truth_sample_round_trips_through_write_and_load(tmp_path):
    path = tmp_path / "ground_truth_sample.jsonl"
    rows = [
        _sample_row(sample_index=0, source_file="a.xlsx", row_id=0),
        _sample_row(sample_index=1, source_file="a.xlsx", row_id=1, quadrant="laya_false_negative"),
    ]

    result_path = write_ground_truth_sample(path, rows)

    assert result_path == path
    loaded = load_ground_truth_sample(path)
    assert loaded == rows


def test_population_weighted_metrics_scores_arbitrary_predictions_with_the_same_weighting():
    # Same oversampled setup as above: 9/900 both_match rows, 91/100 laya_false_negative rows.
    sample_rows = [
        _sample_row(sample_index=i, quadrant="both_match", laya_match=True, llm_match=True) for i in range(9)
    ] + [
        _sample_row(sample_index=100 + i, quadrant="laya_false_negative", laya_match=False, llm_match=True)
        for i in range(91)
    ]
    labels = {r.sample_index: (True, "relevant") for r in sample_rows}
    pool_sizes = {"both_match": 900, "both_no_match": 0, "laya_false_negative": 100, "laya_false_positive": 0}

    # A third classifier that is right on both_match but wrong on laya_false_negative == laya.
    preds = {r.sample_index: r.quadrant == "both_match" for r in sample_rows}
    m = population_weighted_metrics(sample_rows, labels, pool_sizes, preds)
    assert m.accuracy == pytest.approx(0.9)
    assert m.recall == pytest.approx(0.9)

    # The baseline judge's own verdicts reproduce population_weighted_overall exactly.
    llm_preds = {r.sample_index: r.llm_match for r in sample_rows}
    assert population_weighted_metrics(sample_rows, labels, pool_sizes, llm_preds) == population_weighted_overall(
        sample_rows, labels, pool_sizes
    ).llm_baseline
