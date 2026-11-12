"""Measured-throughput tests (issue #15): fake scorers only, fake clock."""

from __future__ import annotations

from scripts.build_ground_truth_report import SampleRow
from scripts.run_system1_bakeoff import (
    build_report,
    evaluate_candidate,
    measure_throughput,
    render_markdown,
    run_bakeoff,
)


class FakeScorer:
    template = "t {keywords}"
    gate_eligible = True
    model_id = "fake/m"
    revision = "r"

    def __init__(self, name="fake", fn=lambda t: 0.9):
        self.name = name
        self.fn = fn
        self.calls: list[list[str]] = []

    def score(self, texts):
        self.calls.append(list(texts))
        return [self.fn(t) for t in texts]


def _row(i, text=None):
    return SampleRow(
        sample_index=i, source_file="a.xlsx", row_id=i, quadrant="both_match",
        product_details_text=text if text is not None else f"row {i}",
        laya_score=0.0, laya_match=False, llm_match=True, llm_rationale="",
    )


def _clock(step):
    t = [0.0]

    def tick():
        t[0] += step
        return t[0]

    return tick


def test_rows_per_second_from_timed_run_with_warmup_excluded():
    scorer = FakeScorer()
    rows = [_row(i) for i in range(600)]
    res = measure_throughput([scorer], rows, timing_rows=500, hardware="Test HW", clock=_clock(2.5))

    (c,) = res["candidates"]
    assert c["status"] == "ok"
    assert c["rows"] == 500
    assert c["seconds"] == 2.5  # one start/stop pair of the fake clock
    assert c["rows_per_second"] == 200.0
    assert res["basis"] == "measured"
    assert res["hardware"] == "Test HW"
    assert res["timing_rows"] == 500
    # warm-up call (model load) happens first and is not in the timed batch
    assert len(scorer.calls) == 2 and len(scorer.calls[0]) < 500 and len(scorer.calls[1]) == 500


def test_fewer_rows_than_requested_uses_what_exists_and_duplicates_are_collapsed():
    scorer = FakeScorer()
    rows = [_row(0, "same"), _row(1, "same"), _row(2, "other")]
    res = measure_throughput([scorer], rows, timing_rows=500, hardware="h", clock=_clock(1.0))

    assert res["candidates"][0]["rows"] == 2
    assert scorer.calls[-1] == ["same", "other"]


def test_a_scorer_that_raises_is_reported_not_fatal():
    def boom(_):
        raise RuntimeError("no mps")

    res = measure_throughput(
        [FakeScorer("bad", boom), FakeScorer("good")], [_row(i) for i in range(3)], hardware="h", clock=_clock(1.0)
    )
    bad, good = res["candidates"]
    assert bad["status"] == "error" and "no mps" in bad["error"] and bad["rows_per_second"] is None
    assert good["status"] == "ok"


def test_timing_does_not_change_accuracy_figures():
    rows = [_row(i) for i in range(10)]
    labels = {i: (i < 8, "") for i in range(10)}
    pools = {"both_match": 10, "both_no_match": 0, "laya_false_negative": 0, "laya_false_positive": 0}

    def fn(t):
        return 0.9 if int(t.split()[1]) < 8 else 0.1

    before = evaluate_candidate(FakeScorer("x", fn), rows, labels, pools)
    measure_throughput([FakeScorer("x", fn)], rows, hardware="h", clock=_clock(1.0))
    after = evaluate_candidate(FakeScorer("x", fn), rows, labels, pools)

    assert before.population_weighted == after.population_weighted
    assert before.raw == after.raw and before.auroc == after.auroc
    # the accuracy report carries no throughput; it is a separate key
    report = build_report(run_bakeoff([FakeScorer("x", fn)], rows, labels, pools), ["k"], len(rows), pools)
    assert "throughput" not in report


def _report_stub(**extra):
    return {
        "threshold": 0.5, "gate": {"population_accuracy": 0.95, "recall": 0.98}, "keywords": ["k"],
        "sample_size": 10, "quadrant_pool_sizes": {"a": 40000, "b": 8000}, "candidates": [], **extra,
    }


def test_markdown_labels_figure_as_measured_with_hardware_and_extrapolation():
    throughput = measure_throughput(
        [FakeScorer("fake")], [_row(i) for i in range(500)], hardware="Apple M1 16 GB", clock=_clock(2.5)
    )
    md = render_markdown(_report_stub(throughput=throughput))

    assert "Measured throughput" in md
    assert "Apple M1 16 GB" in md
    assert "200.0" in md  # rows/sec
    assert "| fake |" in md
    assert "240" in md  # 48000 rows / 200 rows/s = 240 s, extrapolated
    assert "extrapolat" in md.lower()


def test_markdown_without_throughput_skips_the_section():
    assert "throughput" not in render_markdown(_report_stub()).lower()
