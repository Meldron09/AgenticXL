from __future__ import annotations

import json

import pytest

from scripts.validate_laya_filtering import (
    AgreementReport,
    compute_agreement,
    load_baseline,
)


def test_compute_agreement_perfect_match():
    baseline = {0: True, 1: False, 2: True}
    classifier = {0: True, 1: False, 2: True}

    report = compute_agreement(baseline, classifier)

    assert report == AgreementReport(
        total=3,
        agree=3,
        agreement_rate=1.0,
        false_negatives=0,
        false_negative_rate=0.0,
        false_positives=0,
        false_positive_rate=0.0,
        weighted_agreement_rate=1.0,
    )


def test_compute_agreement_counts_false_negative_when_baseline_matched_but_classifier_did_not():
    # baseline said relevant, classifier said not -- the dangerous, under-filtering case
    baseline = {0: True}
    classifier = {0: False}

    report = compute_agreement(baseline, classifier)

    assert report.false_negatives == 1
    assert report.false_negative_rate == 1.0
    assert report.false_positives == 0
    assert report.agreement_rate == 0.0


def test_compute_agreement_counts_false_positive_when_baseline_no_match_but_classifier_matched():
    baseline = {0: False}
    classifier = {0: True}

    report = compute_agreement(baseline, classifier)

    assert report.false_positives == 1
    assert report.false_positive_rate == 1.0
    assert report.false_negatives == 0


def test_compute_agreement_rates_are_denominated_on_baseline_class_counts():
    # 2 baseline-true rows (1 FN), 2 baseline-false rows (1 FP)
    baseline = {0: True, 1: True, 2: False, 3: False}
    classifier = {0: True, 1: False, 2: False, 3: True}

    report = compute_agreement(baseline, classifier)

    assert report.total == 4
    assert report.agree == 2
    assert report.agreement_rate == 0.5
    assert report.false_negatives == 1
    assert report.false_negative_rate == 0.5  # 1 of 2 baseline-true rows
    assert report.false_positives == 1
    assert report.false_positive_rate == 0.5  # 1 of 2 baseline-false rows


def test_compute_agreement_weighted_rate_penalizes_false_negatives_more_than_false_positives():
    ten_fn = {i: True for i in range(10)}
    ten_fn_classifier = {i: False for i in range(10)}
    fn_report = compute_agreement(ten_fn, ten_fn_classifier)

    ten_fp = {i: False for i in range(10)}
    ten_fp_classifier = {i: True for i in range(10)}
    fp_report = compute_agreement(ten_fp, ten_fp_classifier)

    assert fn_report.agreement_rate == fp_report.agreement_rate == 0.0
    assert fn_report.weighted_agreement_rate < fp_report.weighted_agreement_rate


def test_compute_agreement_raises_on_row_id_mismatch():
    with pytest.raises(ValueError):
        compute_agreement({0: True, 1: False}, {0: True})


def test_compute_agreement_empty_returns_full_agreement():
    report = compute_agreement({}, {})
    assert report.total == 0
    assert report.agreement_rate == 1.0
    assert report.false_negative_rate == 0.0
    assert report.false_positive_rate == 0.0


def test_load_baseline_parses_jsonl_into_row_id_to_match_map(tmp_path):
    path = tmp_path / "x_rationale.jsonl"
    lines = [
        {"source_file": "x.xlsx", "row_id": 0, "match": True, "rationale": "a"},
        {"source_file": "x.xlsx", "row_id": 1, "match": False, "rationale": "b"},
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    source_file, baseline = load_baseline(path)

    assert source_file == "x.xlsx"
    assert baseline == {0: True, 1: False}


def test_load_baseline_rejects_empty_file(tmp_path):
    path = tmp_path / "empty_rationale.jsonl"
    path.write_text("")

    with pytest.raises(ValueError):
        load_baseline(path)
