"""How the I3 fit stage assembles its report: metric classes, seed spread, floor skill, interval.

`scripts/run_geo_probes.py` is orchestration, but four of its helpers carry judgement rather than
plumbing, and each of them can be wrong in a way that changes a conclusion rather than crashing:

* `_metrics_from` decides which readings exist -- the moving subset must appear for deltas and must not
  appear for states, where "the log depth is far from zero" is not a statement about change;
* `_summarise` decides which classes get a seed spread, and a class it silently skipped would be
  reported without the noise floor it has to be read against;
* `_floor_skill` compares an arm against a feature-free predictor, so a flipped sign would turn "no
  better than a constant" into a positive result;
* `_delta_interval` is the x-axis of the interval sweep: it turns lag and recording stride into
  seconds, and the whole curve is mislabelled if that algebra is wrong;
* `_curve_point` and `_load_reports` assemble that curve out of many reports, where a mispaired floor
  or a mis-sorted x axis would produce a plausible curve of the wrong thing.

CPU-only: no teacher, no cache, no GPU. Values are hand-computable throughout.
Run:  pytest tests/test_i3_probe_report.py -v
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from starVLA.model.modules.world_model.depth_targets import TARGET_TYPE_METRIC
from starVLA.probes import geo_metrics, geo_probe, probe_cache

# Same handling as `tests/test_i3_vjepa21_weight_check.py`: `scripts/` holds standalone entrypoints and
# is not a package, so it goes on the path explicitly rather than being imported through a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_geo_probes  # noqa: E402

GRID = (2, 2)
VIEWS, HIDDEN = 2, 8
CONSTANT = torch.zeros(1, 1, 1, 1, 1, 1)  # a prediction broadcastable onto any target


def _inputs(rows: int = 3, blocks: int = 4, lag: int = 1) -> geo_probe.ProbeInputs:
    """Targets with real temporal structure; the features are unused by everything tested here."""
    generator = torch.Generator().manual_seed(0)
    states = torch.rand(rows, blocks, VIEWS, 1, *GRID, generator=generator)
    deltas = states[:, lag:] - states[:, :-lag]
    return geo_probe.ProbeInputs(
        features=torch.zeros(rows, blocks * GRID[0] * GRID[1], VIEWS * HIDDEN),
        states=states,
        states_mask=torch.ones_like(states, dtype=torch.bool),
        deltas=deltas,
        deltas_mask=torch.ones_like(deltas, dtype=torch.bool),
        grid=GRID,
        num_views=VIEWS,
        delta_lag=lag,
    )


def _target_cache(**index) -> probe_cache.TargetCache:
    """A TargetCache carrying nothing but its index; only the index fields are read here."""
    array = np.zeros((1, 1, 1, 1, 1, 1), dtype=np.float32)
    return probe_cache.TargetCache(
        root=Path("/nonexistent"),
        index=index,
        states=array,
        states_mask=array.astype(bool),
        deltas=array,
        deltas_mask=array.astype(bool),
    )


# --------------------------------------------------------------------------------------
# which readings exist for which kind
# --------------------------------------------------------------------------------------


def test_the_moving_subset_is_reported_for_deltas_and_only_for_deltas():
    """"Did it move" is a question about a transition; for a state it would mean "is it far away"."""
    data = _inputs()
    states = run_geo_probes._metrics_from(CONSTANT, data, "state", TARGET_TYPE_METRIC)
    deltas = run_geo_probes._metrics_from(CONSTANT, data, "delta", TARGET_TYPE_METRIC)

    assert set(states) == {"metric", "relative", "counts"}
    assert set(deltas) == {
        "metric",
        "relative",
        "counts",
        "metric_moving",
        "relative_moving",
        "counts_moving",
    }


def test_the_subset_reports_the_same_metrics_over_no_more_elements():
    """The subset is a mask, not a second metric family, so the two readings stay comparable."""
    deltas = run_geo_probes._metrics_from(CONSTANT, _inputs(), "delta", TARGET_TYPE_METRIC)
    assert set(deltas["metric_moving"]) == set(deltas["metric"])
    assert set(deltas["relative_moving"]) == set(deltas["relative"])
    assert deltas["counts_moving"]["states_total"] == deltas["counts"]["states_total"]
    assert deltas["counts_moving"]["states_valid"] <= deltas["counts"]["states_valid"]


def test_a_subset_with_nothing_in_it_reports_nan_rather_than_a_flattering_zero():
    data = _inputs()
    # Every delta an order of magnitude below the deadband: the transitions exist but none counts.
    scaled = geo_probe.ProbeInputs(
        features=data.features,
        states=data.states,
        states_mask=data.states_mask,
        deltas=data.deltas * 1e-3,
        deltas_mask=data.deltas_mask,
        grid=data.grid,
        num_views=data.num_views,
    )
    deltas = run_geo_probes._metrics_from(CONSTANT, scaled, "delta", TARGET_TYPE_METRIC)
    assert deltas["counts_moving"]["states_valid"] == 0
    assert math.isnan(deltas["metric_moving"]["abs_rel"])
    assert not math.isnan(deltas["metric"]["abs_rel"]), "the full-grid reading is unaffected"


# --------------------------------------------------------------------------------------
# the seed spread has to cover every class that is reported
# --------------------------------------------------------------------------------------


def test_the_summary_is_the_mean_and_the_sample_spread():
    runs = [{"metric": {"abs_rel": value}, "relative": {"gradient": value}, "counts": {}} for value in (0.1, 0.2, 0.4)]
    summary = run_geo_probes._summarise(runs)
    assert summary["metric"]["abs_rel"] == pytest.approx(0.7 / 3)
    assert summary["metric"]["abs_rel_std"] == pytest.approx(float(np.std([0.1, 0.2, 0.4], ddof=1)))
    assert summary["relative"]["gradient_std"] > 0.0


def test_a_single_seed_reports_a_zero_spread_rather_than_nan():
    """`ddof=1` on one sample is undefined; the report says "no spread measured", not NaN."""
    summary = run_geo_probes._summarise([{"metric": {"abs_rel": 0.2}, "relative": {}, "counts": {}}])
    assert summary["metric"]["abs_rel_std"] == 0.0


def test_every_class_present_in_a_run_gets_summarised_and_counts_do_not():
    """A moving class without its seed spread would be quoted with no noise floor to read it against."""
    data = _inputs()
    delta_runs = [
        run_geo_probes._metrics_from(CONSTANT + shift, data, "delta", TARGET_TYPE_METRIC) for shift in (0.0, 0.1, 0.2)
    ]
    summary = run_geo_probes._summarise(delta_runs)

    assert set(summary) == set(run_geo_probes.METRIC_CLASSES + run_geo_probes.MOVING_CLASSES)
    assert "counts" not in summary and "counts_moving" not in summary
    for metric_class, scores in summary.items():
        names = [name for name in scores if not name.endswith("_std")]
        assert names, f"{metric_class} is empty"
        assert all(f"{name}_std" in scores for name in names)

    state_summary = run_geo_probes._summarise(
        [run_geo_probes._metrics_from(CONSTANT, data, "state", TARGET_TYPE_METRIC)]
    )
    assert set(state_summary) == set(run_geo_probes.METRIC_CLASSES), "states have no moving subset to summarise"


# --------------------------------------------------------------------------------------
# skill against the feature-free floor
# --------------------------------------------------------------------------------------


def test_the_skill_score_is_positive_exactly_when_the_arm_beats_the_floor():
    """Both directions in one call: halving an error and doubling an accuracy are both +100%/+50%."""
    summary = {"metric": {"abs_rel": 0.05, "abs_rel_std": 0.01, "delta1": 0.8, "delta1_std": 0.0}}
    floor = {"metric": {"abs_rel": 0.10, "delta1": 0.4}, "relative": {}, "counts": {"states_valid": 1}}

    skill = run_geo_probes._floor_skill(summary, floor)
    assert skill["metric"]["abs_rel"] == pytest.approx(0.5)
    assert skill["metric"]["delta1"] == pytest.approx(1.0)
    assert "abs_rel_std" not in skill["metric"], "a spread has no direction to improve in"
    assert "counts" not in skill
    assert "relative" not in skill, "an empty floor class yields no skill score"


def test_a_worse_than_the_floor_arm_reports_a_negative_skill():
    summary = {"metric": {"abs_rel": 0.20}}
    skill = run_geo_probes._floor_skill(summary, {"metric": {"abs_rel": 0.10}})
    assert skill["metric"]["abs_rel"] == pytest.approx(-1.0)


def test_a_metric_with_no_declared_direction_raises_instead_of_being_guessed():
    with pytest.raises(KeyError, match="better direction"):
        run_geo_probes._floor_skill({"metric": {"invented": 1.0}}, {"metric": {"invented": 2.0}})


def test_the_skill_score_covers_the_moving_classes():
    summary = {
        "metric": {"abs_rel": 0.05},
        "metric_moving": {"abs_rel": 0.30},
        "relative": {"aligned_delta_mae": 0.05},
        "relative_moving": {"aligned_delta_mae": 0.20},
    }
    floor = {
        "metric": {"abs_rel": 0.10},
        "metric_moving": {"abs_rel": 0.60},
        "relative": {"aligned_delta_mae": 0.10},
        "relative_moving": {"aligned_delta_mae": 0.10},
    }
    skill = run_geo_probes._floor_skill(summary, floor)
    assert set(skill) == set(run_geo_probes.METRIC_CLASSES + run_geo_probes.MOVING_CLASSES)
    assert skill["metric_moving"]["abs_rel"] == pytest.approx(0.5)
    assert skill["relative_moving"]["aligned_delta_mae"] == pytest.approx(-1.0)


# --------------------------------------------------------------------------------------
# the interval a delta target spans, in frames and in seconds
# --------------------------------------------------------------------------------------


def test_a_cache_written_before_the_sweep_reproduces_the_two_frame_contract():
    """No lag and no stride recorded means both were 1, i.e. one tubelet pair at 20 Hz."""
    interval = run_geo_probes._delta_interval(_target_cache())
    assert interval == {
        "delta_lag": 1,
        "clip_stride": 1,
        "tubelet_size": 2,
        "frames": 2,
        "seconds": pytest.approx(0.10),
    }


def test_lag_and_recording_stride_multiply_into_the_interval():
    interval = run_geo_probes._delta_interval(
        _target_cache(delta_lag=3, clip_stride=4, tubelet_size=2, delta_frames=24)
    )
    assert interval["frames"] == 24, "tubelet 2 x lag 3 x stride 4"
    assert interval["seconds"] == pytest.approx(1.2)


def test_the_recorded_frame_count_agrees_with_the_derived_one():
    """The index stores `delta_frames` for readers; it must not be able to disagree with the algebra."""
    cache = _target_cache(delta_lag=2, clip_stride=2, tubelet_size=2, delta_frames=8)
    assert run_geo_probes._delta_interval(cache)["frames"] == cache.index["delta_frames"]


# --------------------------------------------------------------------------------------
# the curve: many reports read along one axis
# --------------------------------------------------------------------------------------

FLOOR, MOVING_FLOOR, SPREAD = 0.10, 0.60, 0.004
FULL_ELEMENTS, MOVING_ELEMENTS = 1000, 340


def _sweep_report(lag: int = 1, stride: int = 1, values=None, moving=None, kind: str = "delta") -> dict:
    """A fit report reduced to what the sweep reads: one metric, its floor, its skill and its counts."""
    values = values or {"A": 0.05}
    moving = moving or {arm: 0.30 for arm in values}

    def probes(arm: str) -> dict:
        summary = {"metric": {"abs_rel": values[arm], "abs_rel_std": SPREAD}}
        skill = {"metric": {"abs_rel": geo_metrics.relative_improvement(FLOOR, values[arm], "abs_rel")}}
        counts = {"counts": {"states_valid": FULL_ELEMENTS}}
        if kind == "delta":
            summary["metric_moving"] = {"abs_rel": moving[arm], "abs_rel_std": 2 * SPREAD}
            skill["metric_moving"] = {"abs_rel": geo_metrics.relative_improvement(MOVING_FLOOR, moving[arm], "abs_rel")}
            counts["counts_moving"] = {"states_valid": MOVING_ELEMENTS}
        return {
            "summary": summary,
            "summary_val": summary,
            "floor_skill": skill,
            "floor_skill_val": skill,
            "per_seed": [counts],
            "per_seed_val": [counts],
        }

    floors = {"metric": {"abs_rel": FLOOR}}
    if kind == "delta":
        floors["metric_moving"] = {"abs_rel": MOVING_FLOOR}
    baselines = {kind: {run_geo_probes.FLOOR_BASELINE: floors}}
    frames = 2 * lag * stride
    return {
        "arms": {arm: {"probes": {kind: probes(arm)}} for arm in values},
        "baselines": baselines,
        "baselines_val": baselines,
        "floor_baseline": run_geo_probes.FLOOR_BASELINE,
        "delta_interval": {
            "delta_lag": lag,
            "clip_stride": stride,
            "tubelet_size": 2,
            "frames": frames,
            "seconds": frames / run_geo_probes.CONTROL_HZ,
        },
    }


def test_each_reading_is_paired_with_its_own_floor_and_its_own_element_count():
    """The failure this guards: scoring the moving subset against the whole grid's floor.

    The subset is both harder and measured over fewer elements, so its floor is a different number;
    crossing the two would invent skill out of nothing.
    """
    point = run_geo_probes._curve_point(_sweep_report(), "A", "delta", "metric", "abs_rel", "val")
    assert (point["seconds"], point["frames"]) == (pytest.approx(0.10), 2)

    full, moving = point["readings"]["full"], point["readings"]["moving"]
    assert (full["value"], full["floor"], full["elements"]) == (0.05, FLOOR, FULL_ELEMENTS)
    assert (moving["value"], moving["floor"], moving["elements"]) == (0.30, MOVING_FLOOR, MOVING_ELEMENTS)
    assert full["skill"] == pytest.approx(0.5) and moving["skill"] == pytest.approx(0.5)
    # Noise is the seed spread on the skill axis, so it shares the skill's denominator.
    assert full["noise"] == pytest.approx(SPREAD / FLOOR)
    assert moving["noise"] == pytest.approx(2 * SPREAD / MOVING_FLOOR)


def test_a_state_probe_contributes_one_reading_only():
    point = run_geo_probes._curve_point(_sweep_report(kind="state"), "A", "state", "metric", "abs_rel", "val")
    assert set(point["readings"]) == {"full"}


def test_the_split_decides_which_numbers_the_curve_is_built_from():
    """The interval is chosen after seeing results, so the curve must be able to avoid the test set."""
    report = _sweep_report()
    report["arms"]["A"]["probes"]["delta"]["summary_val"] = {
        "metric": {"abs_rel": 0.08, "abs_rel_std": SPREAD},
    }
    on_val = run_geo_probes._curve_point(report, "A", "delta", "metric", "abs_rel", "val")
    on_test = run_geo_probes._curve_point(report, "A", "delta", "metric", "abs_rel", "test")
    assert on_val["readings"]["full"]["value"] == 0.08
    assert on_test["readings"]["full"]["value"] == 0.05
    assert "moving" not in on_val["readings"], "the val summary here carries no moving class"


def test_a_split_that_is_not_reported_is_refused():
    with pytest.raises(SystemExit, match="not reported"):
        run_geo_probes._split_suffix("train")


def _write_reports(directory: Path, specs) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for lag, stride in specs:
        report = _sweep_report(lag=lag, stride=stride)
        (directory / f"geo_probes_s{stride}_lag{lag}.json").write_text(json.dumps(report))


def test_reports_are_ordered_along_the_interval_axis_with_stride_breaking_ties(tmp_path):
    """Same duration at two strides is the H5 contrast, not a duplicate, so both stay on the curve."""
    _write_reports(tmp_path, [(2, 1), (1, 1), (1, 2)])
    loaded = run_geo_probes._load_reports([tmp_path])
    assert [
        (report["delta_interval"]["seconds"], report["delta_interval"]["clip_stride"]) for _, report in loaded
    ] == [(0.10, 1), (0.20, 1), (0.20, 2)]


def test_a_report_written_before_the_sweep_is_refused_rather_than_placed_at_zero(tmp_path):
    path = tmp_path / "geo_probes_s1_lag1.json"
    path.write_text(json.dumps({"arms": {}, "baselines": {}}))
    with pytest.raises(SystemExit, match="delta_interval"):
        run_geo_probes._load_reports([path])


def test_a_directory_contributes_only_its_sweep_reports(tmp_path):
    """`geo_probes.json` is the pre-sweep report and lives in the same directory; it is not a point."""
    _write_reports(tmp_path, [(1, 1)])
    (tmp_path / "geo_probes.json").write_text(json.dumps({"arms": {}}))
    assert run_geo_probes._report_paths([tmp_path]) == [tmp_path / "geo_probes_s1_lag1.json"]

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no reports matching"):
        run_geo_probes._report_paths([empty])
    with pytest.raises(SystemExit, match="no report at"):
        run_geo_probes._report_paths([tmp_path / "absent.json"])


def test_the_chart_draws_the_floor_and_puts_a_better_arm_higher():
    lines = run_geo_probes._ascii_curve({"A": [(0.1, -10.0)], "D": [(0.1, 20.0)]})
    floor_rows = [index for index, line in enumerate(lines) if set(line.split("|")[-1]) == {"-"}]
    assert len(floor_rows) == 1, "exactly one zero line"
    rows = {marker: [index for index, line in enumerate(lines) if marker in line] for marker in "AD"}
    assert rows["D"][0] < floor_rows[0] < rows["A"][0], "positive skill above the floor, negative below"


def test_the_chart_reports_nothing_to_plot_rather_than_an_empty_frame():
    assert run_geo_probes._ascii_curve({"A": [(0.1, float("nan"))]}) == ["(nothing to plot)"]
    assert run_geo_probes._ascii_curve({}) == ["(nothing to plot)"]


def test_the_table_reports_every_point_in_both_readings_and_names_its_sources():
    reports = {lag: _sweep_report(lag=lag, values={"A": 0.05, "D": 0.04}) for lag in (1, 2)}
    points = [
        run_geo_probes._curve_point(report, arm, "delta", "metric", "abs_rel", "val")
        for report in reports.values()
        for arm in ("A", "D")
    ]
    table = run_geo_probes._sweep_markdown(points, "abs_rel", "metric", "val", [Path("a.json"), Path("b.json")])
    assert "abs_rel (all tokens)" in table and "abs_rel (moving subset)" in table
    assert table.count("| 0.10 | 2 | 1 x 1 | A |") == 2, "one row per reading"
    assert table.count("| 0.20 | 4 | 1 x 2 | D |") == 2
    assert "- `a.json`" in table and "- `b.json`" in table
    assert str(MOVING_ELEMENTS) in table and str(FULL_ELEMENTS) in table
