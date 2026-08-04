"""Unit tests for the I3 probe metric math and the feature-free baselines.

CPU-only and weight-free by construction: `starVLA/probes/geo_metrics.py` is pure tensor math and
`geo_probe.constant_baselines` reads targets only, so both are testable against hand-computable
values with no teacher, no cache and no GPU (engineering-guidelines: core logic separate from IO).

The claims that matter here are the ones the I3 gate rests on:

* metric class and relative class stay separable, and the relative class is gauge-invariant while the
  metric class is not (gate condition c, AGENTS.md section 7);
* invalid elements are excluded rather than silently averaged as zeros;
* `relative_improvement` cannot report the wrong sign for a metric whose direction it does not know.

Run:  pytest tests/test_i3_geo_metrics.py -v
"""

import math

import pytest
import torch

from starVLA.model.modules.world_model import depth_targets
from starVLA.model.modules.world_model.depth_targets import (
    METRIC_TARGET_TYPES,
    TARGET_TYPE_METRIC,
    TARGET_TYPE_PSEUDO,
    TARGET_TYPE_PSEUDO_METRIC,
    TARGET_TYPES,
)
from starVLA.probes import geo_metrics, geo_probe

SHAPE = (2, 4, 2, 1, 4, 4)  # [N, Tp, V, 1, h, w], the layout depth_targets emits


def _targets(seed: int = 0) -> torch.Tensor:
    """Log-depth targets with real spatial and temporal structure, so gradient terms are non-trivial."""
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(SHAPE, generator=generator) * 0.8 - 0.4


def _all_valid() -> torch.Tensor:
    return torch.ones(SHAPE, dtype=torch.bool)


# --------------------------------------------------------------------------------------
# relative_improvement: the sign is not allowed to be guessed
# --------------------------------------------------------------------------------------


def test_lower_is_better_metrics_improve_when_the_candidate_is_smaller():
    assert geo_metrics.relative_improvement(0.4, 0.2, "abs_rel") == pytest.approx(0.5)
    assert geo_metrics.relative_improvement(0.2, 0.4, "abs_rel") == pytest.approx(-1.0)


def test_higher_is_better_metrics_improve_when_the_candidate_is_larger():
    assert geo_metrics.relative_improvement(0.5, 0.6, "delta1") == pytest.approx(0.2)
    assert geo_metrics.relative_improvement(0.6, 0.5, "delta1") == pytest.approx(-1 / 6)


def test_an_undeclared_metric_raises_instead_of_defaulting_to_a_direction():
    with pytest.raises(KeyError, match="better direction"):
        geo_metrics.relative_improvement(1.0, 0.5, "some_new_metric")


def test_every_reported_metric_declares_a_direction():
    """Guards the gate arithmetic: a metric in the report but in neither set would raise at judging time."""
    reported = set(
        geo_metrics.metric_depth_metrics(_targets(), _targets(), _all_valid())
        | geo_metrics.relative_depth_metrics(_targets(), _targets(), _all_valid()).keys()
    )
    undeclared = reported - geo_metrics.LOWER_IS_BETTER - geo_metrics.HIGHER_IS_BETTER
    assert undeclared == set()


# --------------------------------------------------------------------------------------
# metric class, against values computable by hand
# --------------------------------------------------------------------------------------


def test_a_perfect_prediction_scores_perfectly():
    target = _targets()
    scores = geo_metrics.metric_depth_metrics(target.clone(), target, _all_valid())
    assert scores["abs_rel"] == pytest.approx(0.0, abs=1e-6)
    assert scores["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert scores["log_mae"] == pytest.approx(0.0, abs=1e-6)
    assert scores["gradient"] == pytest.approx(0.0, abs=1e-6)
    assert scores["delta1"] == pytest.approx(1.0)
    assert scores["temporal_sign_agreement"] == pytest.approx(1.0)


def test_a_uniform_ten_percent_overestimate_scores_exactly_ten_percent():
    """A constant log offset scales every depth by the same factor, so AbsRel is that factor exactly.

    It also pins the gradient term's purpose: the offset cancels in every spatial difference, so
    "knows the layout" and "knows the distance" stay distinguishable.
    """
    target = _targets()
    scores = geo_metrics.metric_depth_metrics(target + math.log(1.1), target, _all_valid())
    assert scores["abs_rel"] == pytest.approx(0.1, abs=1e-5)
    assert scores["log_mae"] == pytest.approx(math.log(1.1), abs=1e-6)
    assert scores["gradient"] == pytest.approx(0.0, abs=1e-6)
    assert scores["delta1"] == pytest.approx(1.0), "a ratio of 1.1 is inside the 1.25 threshold"
    assert scores["implied_delta_mae"] == pytest.approx(0.0, abs=1e-6)


def test_delta1_counts_ratios_strictly_below_the_threshold():
    target = torch.zeros(SHAPE)
    just_inside = geo_metrics.metric_depth_metrics(target + math.log(1.249), target, _all_valid())
    just_outside = geo_metrics.metric_depth_metrics(target + math.log(1.251), target, _all_valid())
    assert just_inside["delta1"] == pytest.approx(1.0)
    assert just_outside["delta1"] == pytest.approx(0.0)


def test_boundary_f1_is_nan_when_the_target_has_no_boundary():
    """Undefined, not perfect: a flat scene cannot demonstrate that a probe finds discontinuities."""
    flat = torch.zeros(SHAPE)
    scores = geo_metrics.metric_depth_metrics(flat.clone(), flat, _all_valid())
    assert math.isnan(scores["boundary_f1"])


def test_temporal_changes_inside_the_deadband_are_dropped_not_counted_as_coin_flips():
    target = torch.zeros(SHAPE)
    # Every temporal change is an order of magnitude below the deadband, so nothing is scorable.
    target[:, 1::2] = geo_metrics.TEMPORAL_DEADBAND / 10
    agreement = geo_metrics.temporal_sign_agreement(-target, target, _all_valid())
    assert math.isnan(agreement), "sub-deadband changes must be excluded, not scored"


# --------------------------------------------------------------------------------------
# masking: invalid elements are excluded, never averaged in as zeros
# --------------------------------------------------------------------------------------


def test_invalid_elements_do_not_reach_any_metric():
    target, mask = _targets(), _all_valid()
    mask[:, :, :, :, 0, :] = False  # drop the top grid row
    prediction = target + math.log(1.1)
    corrupted = prediction.clone()
    corrupted[:, :, :, :, 0, :] = 1e3  # nonsense exactly where the mask says "do not look"

    clean_scores = geo_metrics.metric_depth_metrics(prediction, target, mask)
    corrupted_scores = geo_metrics.metric_depth_metrics(corrupted, target, mask)
    for name, value in clean_scores.items():
        assert corrupted_scores[name] == pytest.approx(value, abs=1e-5), f"{name} looked at masked elements"


def test_an_empty_mask_reports_nan_rather_than_dividing_by_zero():
    target = _targets()
    scores = geo_metrics.metric_depth_metrics(target, target, torch.zeros(SHAPE, dtype=torch.bool))
    assert math.isnan(scores["abs_rel"]) and math.isnan(scores["log_mae"])


def test_counts_separate_valid_from_invalid_elements():
    """Gate condition c wants invalid / pseudo / metric statistics separable; this is the invalid half."""
    mask = _all_valid()
    mask[0] = False
    result = geo_metrics.evaluate(_targets(), _targets(), mask, target_type=TARGET_TYPE_METRIC)
    assert result.counts["states_total"] == mask.numel()
    assert result.counts["states_valid"] == int(mask.sum())
    assert result.counts["states_invalid"] == mask.numel() - int(mask.sum())


# --------------------------------------------------------------------------------------
# the moving subset: the same metrics, over the elements that actually changed
# --------------------------------------------------------------------------------------


def test_the_moving_mask_keeps_only_the_elements_that_changed():
    target = torch.zeros(SHAPE)
    target[:, :, :, :, 0, 0] = 0.5  # one grid cell moved, the rest of the scene is static
    moving = geo_metrics.moving_mask(target, _all_valid())
    assert int(moving.sum()) == SHAPE[0] * SHAPE[1] * SHAPE[2]
    assert moving[:, :, :, :, 0, 0].all()
    assert not moving[:, :, :, :, 1, 1].any()


def test_the_moving_mask_cannot_revive_an_invalid_element():
    """It narrows a mask and never widens one: a moving element with no valid depth stays excluded."""
    target, mask = torch.full(SHAPE, 0.5), _all_valid()
    mask[0, 0] = False
    assert not geo_metrics.moving_mask(target, mask)[0, 0].any()
    assert geo_metrics.moving_mask(target, mask)[1].all()


def test_the_moving_threshold_is_the_sign_agreement_deadband_and_strict():
    """One threshold for both, so "too small to have a sign" and "did not move" cannot disagree."""
    at = torch.full(SHAPE, geo_metrics.TEMPORAL_DEADBAND)
    above = torch.full(SHAPE, geo_metrics.TEMPORAL_DEADBAND * 1.01)
    assert not geo_metrics.moving_mask(at, _all_valid()).any(), "exactly at the threshold is not moving"
    assert geo_metrics.moving_mask(above, _all_valid()).all()
    assert geo_metrics.moving_mask(-above, _all_valid()).all(), "the magnitude decides, not the sign"


def test_a_fully_static_target_leaves_an_empty_subset_and_reports_nan():
    """The empty set is a definition, not a bug: with nothing moving there is nothing to score."""
    static = torch.full(SHAPE, geo_metrics.TEMPORAL_DEADBAND / 10)
    moving = geo_metrics.moving_mask(static, _all_valid())
    assert not moving.any()
    scores = geo_metrics.metric_depth_metrics(static.clone(), static, moving)
    assert math.isnan(scores["abs_rel"]) and math.isnan(scores["log_mae"])


def test_the_subset_separates_a_predictor_that_only_gets_the_static_part_right():
    """Why the subset exists (H4): the full-grid average hides a prediction that predicts no change.

    The static predictor is right everywhere nothing moved, so over the whole grid it looks close to
    the target; on the moving subset it is exactly as wrong as the change it failed to predict.
    """
    target = torch.zeros(SHAPE)
    target[:, :, :, :, 0, 0] = 0.5
    static_prediction = torch.zeros(SHAPE)

    full = geo_metrics.metric_depth_metrics(static_prediction, target, _all_valid())
    subset = geo_metrics.metric_depth_metrics(
        static_prediction, target, geo_metrics.moving_mask(target, _all_valid())
    )
    assert full["log_mae"] == pytest.approx(0.5 / (SHAPE[-1] * SHAPE[-2]), abs=1e-6)
    assert subset["log_mae"] == pytest.approx(0.5, abs=1e-6)
    assert subset["log_mae"] > 10 * full["log_mae"], "the static background hid the entire error"


# --------------------------------------------------------------------------------------
# the two classes stay separate, and only one of them is gauge-invariant
# --------------------------------------------------------------------------------------


def test_an_affine_prediction_is_perfect_after_alignment_and_wrong_before_it():
    """The defining difference between the classes, stated on one prediction.

    An affinely transformed prediction carries the full relative structure and none of the metric
    scale, so the relative class must score it perfectly and the metric class must not.
    """
    target = _targets()
    prediction = 2.5 * target + 0.7
    relative = geo_metrics.relative_depth_metrics(prediction, target, _all_valid())
    metric = geo_metrics.metric_depth_metrics(prediction, target, _all_valid())
    assert relative["aligned_state_mae"] == pytest.approx(0.0, abs=1e-5)
    assert relative["gradient"] == pytest.approx(0.0, abs=1e-5)
    assert metric["log_mae"] > 0.1, "the metric class must not be gauge-invariant"


def test_a_pseudo_target_reports_no_metric_class_at_all():
    """Metres have no referent in MAD units, so the metric dict is empty rather than misleading."""
    target = _targets()
    pseudo = geo_metrics.evaluate(target, target, _all_valid(), target_type=TARGET_TYPE_PSEUDO)
    assert pseudo.metric == {}
    assert pseudo.relative and pseudo.target_type == TARGET_TYPE_PSEUDO

    metric = geo_metrics.evaluate(target, target, _all_valid(), target_type=TARGET_TYPE_METRIC)
    assert metric.metric and metric.relative


def test_a_metric_estimator_target_gets_the_metric_class_but_keeps_its_own_label():
    """S4's estimators predict metres, so AbsRel means something -- yet they are not the simulator."""
    target = _targets()
    pseudo_metric = geo_metrics.evaluate(target, target, _all_valid(), target_type=TARGET_TYPE_PSEUDO_METRIC)
    assert pseudo_metric.metric, "log metres from an estimator still have a metric referent"
    assert pseudo_metric.target_type == TARGET_TYPE_PSEUDO_METRIC
    assert pseudo_metric.target_type != TARGET_TYPE_METRIC, "estimator output must not look like sim GT"

    simulator = geo_metrics.evaluate(target, target, _all_valid(), target_type=TARGET_TYPE_METRIC)
    assert set(pseudo_metric.metric) == set(simulator.metric), "same metric names, so the two are comparable"


def test_the_metric_class_is_gated_on_units_not_on_one_string():
    """The gate must be `in METRIC_TARGET_TYPES`; an equality check would have silently excluded S4."""
    assert TARGET_TYPE_METRIC in METRIC_TARGET_TYPES
    assert TARGET_TYPE_PSEUDO_METRIC in METRIC_TARGET_TYPES
    assert TARGET_TYPE_PSEUDO not in METRIC_TARGET_TYPES
    assert METRIC_TARGET_TYPES < TARGET_TYPES


def test_a_relative_target_type_cannot_reach_the_metric_pipeline():
    """`log_metric_depth` produces log metres, so labelling its output relative is a contradiction."""
    with pytest.raises(ValueError, match="not one of the metric types"):
        depth_targets.log_metric_depth(torch.full(SHAPE, 0.5), target_type=TARGET_TYPE_PSEUDO)


def test_the_estimator_and_simulator_paths_are_the_same_maths():
    """Only the label may differ: a parallel implementation for estimators would be unauditable."""
    depth = torch.rand(2, 2, 4, 1, 4, 4) * 2.0 + 0.1  # [B, V, T, 1, H, W] metres
    simulator = depth_targets.build_metric_delta_targets(depth, target_type=TARGET_TYPE_METRIC)
    estimator = depth_targets.build_metric_delta_targets(depth, target_type=TARGET_TYPE_PSEUDO_METRIC)
    for from_sim, from_estimator in zip(simulator, estimator, strict=True):
        assert torch.equal(from_sim.values, from_estimator.values)
        assert torch.equal(from_sim.mask, from_estimator.mask)
        assert from_sim.units == from_estimator.units
        assert from_sim.target_type != from_estimator.target_type


def test_geo_metrics_offers_no_combined_scalar():
    result = geo_metrics.evaluate(_targets(), _targets(), _all_valid(), target_type=TARGET_TYPE_METRIC)
    assert set(result.as_rows()) == {"metric", "relative"}
    assert not [name for name in dir(result) if name in ("total", "score", "combined", "overall")]


def test_an_unknown_target_type_is_rejected():
    with pytest.raises(ValueError, match="target_type"):
        geo_metrics.evaluate(_targets(), _targets(), _all_valid(), target_type="guessed")


# --------------------------------------------------------------------------------------
# feature-free baselines: the floor the probe numbers are read against
# --------------------------------------------------------------------------------------


def _inputs(states: torch.Tensor, mask: torch.Tensor) -> geo_probe.ProbeInputs:
    """ProbeInputs carrying the given state targets; features are unused by the baselines."""
    rows, blocks = states.shape[0], states.shape[1]
    return geo_probe.ProbeInputs(
        features=torch.zeros(rows, blocks * SHAPE[-2] * SHAPE[-1], 2 * 8),
        states=states,
        states_mask=mask,
        deltas=states[:, 1:],
        deltas_mask=mask[:, 1:],
        grid=(SHAPE[-2], SHAPE[-1]),
        num_views=SHAPE[2],
    )


def test_the_global_baseline_is_the_masked_median():
    states, mask = _targets(), _all_valid()
    mask[0, 0] = False
    baselines = geo_probe.constant_baselines(_inputs(states, mask), "state")
    # `quantile`, not `median`: on an even count torch's `median` returns the lower of the two middle
    # values while the quantile interpolates. Both minimise L1; the baseline uses the interpolating one.
    expected = torch.quantile(states[mask], 0.5)
    assert baselines["global_constant"].shape == (1, 1, 1, 1, 1, 1)
    assert float(baselines["global_constant"]) == pytest.approx(float(expected), abs=1e-5)


def test_the_per_token_baseline_is_one_median_per_view_and_grid_cell():
    states, mask = _targets(), _all_valid()
    per_token = geo_probe.constant_baselines(_inputs(states, mask), "state")["per_token_constant"]
    assert per_token.shape == (1, 1, *SHAPE[2:])
    # Collapsed axes are rows and time, so cell (v, y, x) is the median over those.
    for view, row, column in ((0, 0, 0), (1, 2, 3)):
        expected = torch.quantile(states[:, :, view, 0, row, column].flatten(), 0.5)
        assert float(per_token[0, 0, view, 0, row, column]) == pytest.approx(float(expected), abs=1e-5)


def test_the_per_token_baseline_ignores_invalid_elements():
    states, mask = _targets(), _all_valid()
    states[0, 0, 0, 0, 1, 1] = 1e3
    mask[0, 0, 0, 0, 1, 1] = False
    per_token = geo_probe.constant_baselines(_inputs(states, mask), "state")["per_token_constant"]
    assert float(per_token[0, 0, 0, 0, 1, 1]) < 1.0, "an invalid outlier moved the median"


def test_the_baselines_broadcast_onto_the_target_layout():
    """The report scores them through the same `evaluate` call as a fitted head, so shapes must fit."""
    states, mask = _targets(), _all_valid()
    for prediction in geo_probe.constant_baselines(_inputs(states, mask), "state").values():
        scores = geo_metrics.evaluate(
            prediction.expand_as(states).contiguous(), states, mask, target_type=TARGET_TYPE_METRIC
        )
        assert scores.metric["log_mae"] > 0.0
