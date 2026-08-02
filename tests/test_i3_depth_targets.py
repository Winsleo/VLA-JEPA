"""Depth target contract for I3 (S1b).

Pins the maths of `docs/implementation-plan.md` section 6 and the depth path of section 4.1:
log-clipped metric depth, clip-level median/MAD normalisation for relative pseudo-depth, tubelet
alignment on the last frame, adjacent deltas, and valid-weighted pooling onto the probe token grid.

The three properties the I3 gate depends on are tested explicitly:

* determinism - the same cached clip must produce bitwise identical targets (condition b);
* separability - invalid, range-clipped and usable pixels are counted apart (condition c);
* no silent mixing - metric and relative targets refuse to be combined (AGENTS.md section 7).

Pure tensor math, CPU-only.
Run:  pytest tests/test_i3_depth_targets.py -v
"""

import math

import pytest
import torch

from starVLA.model.modules.world_model.depth_targets import (
    DEFAULT_D_MAX,
    DEFAULT_D_MIN,
    TARGET_TYPE_METRIC,
    TARGET_TYPE_PSEUDO,
    UNITS_LOG_METER,
    UNITS_LOG_METER_DELTA,
    UNITS_MAD,
    DepthTarget,
    adjacent_delta,
    build_metric_delta_targets,
    log_metric_depth,
    normalize_clip_level,
    pool_to_grid,
    range_clip_mask,
    require_same_target_type,
    tubelet_last_frame,
)

BATCH, VIEWS, FRAMES, SIZE = 2, 2, 8, 16
TUBELET = 2


def cache_clip(seed=0, size=SIZE):
    """A synthetic `[B, V, T, 1, H, W]` metric-depth clip in the measured LIBERO range."""
    generator = torch.Generator().manual_seed(seed)
    depth = 0.3 + 2.5 * torch.rand(BATCH, VIEWS, FRAMES, 1, size, size, generator=generator, dtype=torch.float32)
    return depth


class TestLogMetricDepth:
    def test_values_are_log_of_the_clipped_depth(self):
        depth = torch.tensor([[0.5, 1.0, 2.0]])
        target = log_metric_depth(depth)
        assert torch.allclose(target.values, torch.log(depth))
        assert target.target_type == TARGET_TYPE_METRIC
        assert target.units == UNITS_LOG_METER

    def test_out_of_range_depth_is_clipped_not_dropped(self):
        depth = torch.tensor([[DEFAULT_D_MIN / 2, DEFAULT_D_MAX * 2]])
        target = log_metric_depth(depth)
        assert target.mask.all(), "clipping is not invalidity"
        assert math.isclose(float(target.values[0, 0]), math.log(DEFAULT_D_MIN), rel_tol=1e-6)
        assert math.isclose(float(target.values[0, 1]), math.log(DEFAULT_D_MAX), rel_tol=1e-6)

    def test_clipped_pixels_stay_separately_countable(self):
        """Gate condition c: range clipping is reported apart from the validity mask."""
        depth = torch.tensor([[DEFAULT_D_MIN / 2, 1.0, DEFAULT_D_MAX * 2, float("nan")]])
        clipped = range_clip_mask(depth)
        valid = log_metric_depth(depth).mask
        assert clipped.tolist() == [[True, False, True, False]]
        assert valid.tolist() == [[True, True, True, False]]
        assert int((valid & ~clipped).sum()) == 1

    def test_invalid_pixels_are_masked_and_zero_filled(self):
        depth = torch.tensor([[float("nan"), float("inf"), -1.0, 1.0]])
        target = log_metric_depth(depth)
        assert target.mask.tolist() == [[False, False, False, True]]
        assert torch.isfinite(target.values).all(), "masked positions must not carry NaN/Inf"
        assert float(target.values[0, 0]) == 0.0

    def test_recorded_sensor_mask_is_intersected(self):
        depth = torch.full((1, 3), 1.0)
        valid = torch.tensor([[True, False, True]])
        assert log_metric_depth(depth, valid=valid).mask.tolist() == valid.tolist()

    def test_a_non_positive_clip_bound_is_rejected(self):
        with pytest.raises(ValueError, match="d_min must be positive"):
            log_metric_depth(torch.ones(1, 1), d_min=0.0)
        with pytest.raises(ValueError, match="must exceed"):
            log_metric_depth(torch.ones(1, 1), d_min=1.0, d_max=0.5)


class TestClipLevelNormalization:
    def test_statistics_are_clip_level_not_frame_wise(self):
        """Section 6 forbids frame-wise normalisation: a per-frame offset must survive."""
        depth = torch.ones(1, 1, 4, 1, 2, 2)
        depth[0, 0, 2] = 3.0
        target = normalize_clip_level(depth)
        assert target.target_type == TARGET_TYPE_PSEUDO
        assert target.units == UNITS_MAD
        per_frame = [float(target.values[0, 0, t].mean()) for t in range(4)]
        assert per_frame[2] != pytest.approx(per_frame[0]), "frame offsets must not be normalised away"

    def test_median_centres_the_clip(self):
        depth = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 4, 1, 1, 1)
        values = normalize_clip_level(depth).values.reshape(-1)
        assert float(values.median()) == pytest.approx(0.0, abs=1e-5)

    def test_views_are_normalised_independently(self):
        depth = torch.ones(1, 2, 4, 1, 2, 2)
        depth[0, 1] *= 10.0  # a differently scaled view must not shift the other one
        target = normalize_clip_level(depth)
        assert torch.allclose(target.values[0, 0], target.values[0, 1], atol=1e-5)

    def test_invalid_pixels_do_not_enter_the_statistics(self):
        depth = torch.ones(1, 1, 4, 1, 2, 2)
        depth[0, 0, 0, 0, 0, 0] = 1e6
        valid = torch.ones_like(depth, dtype=torch.bool)
        valid[0, 0, 0, 0, 0, 0] = False
        target = normalize_clip_level(depth, valid=valid)
        assert torch.isfinite(target.values).all()
        assert float(target.values[target.mask].abs().max()) == pytest.approx(0.0, abs=1e-5)


class TestTubeletAlignment:
    def test_each_tubelet_keeps_its_last_frame(self):
        depth = torch.arange(FRAMES, dtype=torch.float32).reshape(1, 1, FRAMES, 1, 1, 1)
        aligned = tubelet_last_frame(depth, TUBELET)
        assert aligned.shape == (1, FRAMES // TUBELET, 1, 1, 1, 1)
        assert aligned.reshape(-1).tolist() == [1.0, 3.0, 5.0, 7.0]

    def test_axis_order_follows_the_documented_depth_path(self):
        aligned = tubelet_last_frame(cache_clip(), TUBELET)
        assert aligned.shape == (BATCH, FRAMES // TUBELET, VIEWS, 1, SIZE, SIZE)

    def test_views_are_not_transposed(self):
        depth = torch.zeros(1, VIEWS, FRAMES, 1, 1, 1)
        depth[:, 1] = 5.0
        aligned = tubelet_last_frame(depth, TUBELET)
        assert float(aligned[0, 0, 0]) == 0.0 and float(aligned[0, 0, 1]) == 5.0

    def test_indivisible_frame_count_is_rejected(self):
        with pytest.raises(ValueError, match="not divisible"):
            tubelet_last_frame(torch.zeros(1, 1, 7, 1, 1, 1), TUBELET)

    def test_wrong_rank_is_rejected(self):
        with pytest.raises(ValueError, match=r"expected \[B,V,T,1,H,W\]"):
            tubelet_last_frame(torch.zeros(1, 1, 8, 4, 4), TUBELET)


class TestAdjacentDelta:
    def test_delta_is_the_forward_difference(self):
        states = DepthTarget(
            values=torch.tensor([0.0, 1.0, 3.0]).reshape(1, 3, 1, 1, 1, 1),
            mask=torch.ones(1, 3, 1, 1, 1, 1, dtype=torch.bool),
            target_type=TARGET_TYPE_METRIC,
            units=UNITS_LOG_METER,
        )
        delta = adjacent_delta(states)
        assert delta.values.reshape(-1).tolist() == [1.0, 2.0]
        assert delta.units == UNITS_LOG_METER_DELTA

    def test_masks_are_intersected_pairwise(self):
        mask = torch.ones(1, 3, 1, 1, 1, 1, dtype=torch.bool)
        mask[0, 1] = False
        states = DepthTarget(
            values=torch.zeros(1, 3, 1, 1, 1, 1),
            mask=mask,
            target_type=TARGET_TYPE_METRIC,
            units=UNITS_LOG_METER,
        )
        assert adjacent_delta(states).mask.reshape(-1).tolist() == [False, False]

    def test_a_single_state_cannot_form_a_delta(self):
        states = DepthTarget(
            values=torch.zeros(1, 1, 1, 1, 1, 1),
            mask=torch.ones(1, 1, 1, 1, 1, 1, dtype=torch.bool),
            target_type=TARGET_TYPE_METRIC,
            units=UNITS_LOG_METER,
        )
        with pytest.raises(ValueError, match="at least 2 states"):
            adjacent_delta(states)


class TestPoolToGrid:
    def test_pooling_averages_valid_pixels_only(self):
        values = torch.tensor([[1.0, 3.0], [5.0, 7.0]]).reshape(1, 1, 2, 2)
        mask = torch.tensor([[True, True], [False, False]]).reshape(1, 1, 2, 2)
        target = DepthTarget(values, mask, TARGET_TYPE_METRIC, UNITS_LOG_METER)
        pooled = pool_to_grid(target, (1, 1))
        assert float(pooled.values) == pytest.approx(2.0)
        assert bool(pooled.mask)

    def test_a_cell_without_valid_pixels_is_masked(self):
        target = DepthTarget(
            values=torch.ones(1, 1, 2, 2),
            mask=torch.zeros(1, 1, 2, 2, dtype=torch.bool),
            target_type=TARGET_TYPE_METRIC,
            units=UNITS_LOG_METER,
        )
        pooled = pool_to_grid(target, (1, 1))
        assert not bool(pooled.mask)
        assert float(pooled.values) == 0.0

    def test_pooling_is_identity_on_a_matching_grid(self):
        target = log_metric_depth(cache_clip(size=4))
        pooled = pool_to_grid(target, (4, 4))
        assert torch.equal(pooled.values, target.values)

    def test_leading_axes_and_type_survive(self):
        states, _ = build_metric_delta_targets(cache_clip(), tubelet_size=TUBELET)
        pooled = pool_to_grid(states, (4, 4))
        assert pooled.values.shape == (BATCH, FRAMES // TUBELET, VIEWS, 1, 4, 4)
        assert pooled.target_type == TARGET_TYPE_METRIC

    def test_an_indivisible_grid_is_rejected(self):
        target = log_metric_depth(cache_clip(size=6))
        with pytest.raises(ValueError, match="does not divide"):
            pool_to_grid(target, (4, 4))


class TestTypeFirewall:
    def test_metric_and_relative_targets_refuse_to_combine(self):
        depth = cache_clip()
        with pytest.raises(ValueError, match="refusing to combine target types"):
            require_same_target_type(log_metric_depth(depth), normalize_clip_level(depth))

    def test_states_and_deltas_refuse_to_combine(self):
        states, deltas = build_metric_delta_targets(cache_clip(), tubelet_size=TUBELET)
        with pytest.raises(ValueError, match="refusing to combine units"):
            require_same_target_type(states, deltas)

    def test_the_type_label_survives_every_transformation(self):
        pseudo = normalize_clip_level(cache_clip())
        aligned = DepthTarget(
            values=tubelet_last_frame(pseudo.values, TUBELET),
            mask=tubelet_last_frame(pseudo.mask, TUBELET),
            target_type=pseudo.target_type,
            units=pseudo.units,
        )
        assert adjacent_delta(pool_to_grid(aligned, (4, 4))).target_type == TARGET_TYPE_PSEUDO

    def test_a_mask_of_the_wrong_shape_is_rejected(self):
        with pytest.raises(ValueError, match="mask shape"):
            DepthTarget(torch.zeros(2, 2), torch.ones(2, dtype=torch.bool), TARGET_TYPE_METRIC, "x")

    def test_a_non_boolean_mask_is_rejected(self):
        with pytest.raises(TypeError, match="mask must be bool"):
            DepthTarget(torch.zeros(2), torch.ones(2), TARGET_TYPE_METRIC, "x")


class TestPipelineShapesAndDeterminism:
    def test_documented_state_and_delta_shapes(self):
        states, deltas = build_metric_delta_targets(cache_clip(), tubelet_size=TUBELET, grid=(4, 4))
        assert states.values.shape == (BATCH, 4, VIEWS, 1, 4, 4)
        assert deltas.values.shape == (BATCH, 3, VIEWS, 1, 4, 4)
        assert deltas.mask.shape == deltas.values.shape

    def test_targets_are_bitwise_reproducible(self):
        """I3 gate condition b: the target pipeline is deterministic on a fixed clip."""
        depth = cache_clip(seed=3)
        first = build_metric_delta_targets(depth, tubelet_size=TUBELET, grid=(4, 4))
        second = build_metric_delta_targets(depth.clone(), tubelet_size=TUBELET, grid=(4, 4))
        for left, right in zip(first, second, strict=True):
            assert torch.equal(left.values, right.values)
            assert torch.equal(left.mask, right.mask)

    def test_pseudo_targets_are_bitwise_reproducible(self):
        depth = cache_clip(seed=4)
        assert torch.equal(normalize_clip_level(depth).values, normalize_clip_level(depth).values)

    def test_no_nan_or_inf_escapes_a_fully_invalid_clip(self):
        depth = torch.full((1, 1, FRAMES, 1, 2, 2), float("nan"))
        states, deltas = build_metric_delta_targets(depth, tubelet_size=TUBELET)
        pseudo = normalize_clip_level(depth)
        for target in (states, deltas, pseudo):
            assert torch.isfinite(target.values).all()
            assert not target.mask.any()
