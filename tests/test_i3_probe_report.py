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
  seconds, and the whole curve is mislabelled if that algebra is wrong.

CPU-only: no teacher, no cache, no GPU. Values are hand-computable throughout.
Run:  pytest tests/test_i3_probe_report.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from starVLA.model.modules.world_model.depth_targets import TARGET_TYPE_METRIC
from starVLA.probes import geo_probe, probe_cache

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
