"""Unit tests for SpatialTokenResampler, the non-learned 24x24 -> 16x16 token pooling of I3.

Pure tensor logic, no weights and no GPU: the resampler exists so the two teacher arms can be
compared on one output grid (`docs/implementation-plan.md` section 4.1), and getting its pooling
windows wrong would silently change what every probe number means.

Run:  pytest tests/test_i3_resampler.py -v
"""

import pytest
import torch

from starVLA.model.modules.world_model.spatial_token_resampler import SpatialTokenResampler

# The pre-registered I3 arm C geometry: V-JEPA 2.1 at 384 pooled onto the V-JEPA 2 grid.
GRID_IN = (24, 24)
GRID_OUT = (16, 16)


def _ramp(grid, blocks=1, batch=1, dim=1):
    """[batch, blocks * h * w, dim] whose value is the flat token index, so pooling is readable."""
    tokens = blocks * grid[0] * grid[1]
    return torch.arange(tokens, dtype=torch.float64).reshape(1, tokens, 1).expand(batch, tokens, dim).contiguous()


# --------------------------------------------------------------------------------------
# construction contract
# --------------------------------------------------------------------------------------

def test_token_counts_follow_the_grids():
    resampler = SpatialTokenResampler(grid_in=GRID_IN, grid_out=GRID_OUT)
    assert resampler.tokens_in == 576
    assert resampler.tokens_out == 256


def test_upsampling_is_refused():
    """Adaptive pooling would repeat inputs instead of averaging, i.e. invent tokens."""
    with pytest.raises(ValueError, match="not smaller than or equal to"):
        SpatialTokenResampler(grid_in=GRID_OUT, grid_out=GRID_IN)


def test_a_grid_that_shrinks_on_one_axis_only_is_refused():
    with pytest.raises(ValueError, match="not smaller than or equal to"):
        SpatialTokenResampler(grid_in=(24, 16), grid_out=(16, 24))


@pytest.mark.parametrize("grid", [(16,), (16, 16, 16), (16.0, 16), (0, 16), (-16, 16)])
def test_malformed_grids_are_refused(grid):
    with pytest.raises(ValueError, match="two positive ints"):
        SpatialTokenResampler(grid_in=grid, grid_out=(8, 8))


def test_it_holds_no_parameters_and_stays_out_of_any_state_dict():
    """Deliberately not an nn.Module: a probe arm must not gain capacity from the resampler."""
    resampler = SpatialTokenResampler(grid_in=GRID_IN, grid_out=GRID_OUT)
    assert not isinstance(resampler, torch.nn.Module)
    assert not any(isinstance(value, torch.Tensor) for value in vars(resampler).values())


# --------------------------------------------------------------------------------------
# pooling correctness
# --------------------------------------------------------------------------------------

def test_equal_grids_are_the_identity_object():
    resampler = SpatialTokenResampler(grid_in=GRID_OUT, grid_out=GRID_OUT)
    features = _ramp(GRID_OUT, blocks=4, dim=3)
    assert resampler(features) is features


def test_a_constant_field_is_preserved_exactly():
    """Averages of a constant are that constant, whatever the window layout."""
    resampler = SpatialTokenResampler(grid_in=GRID_IN, grid_out=GRID_OUT)
    pooled = resampler(torch.full((2, 4 * 576, 3), 0.375, dtype=torch.float64))
    assert torch.equal(pooled, torch.full((2, 4 * 256, 3), 0.375, dtype=torch.float64))


def test_an_exact_ratio_pools_disjoint_2x2_windows():
    """32 -> 16 divides evenly, so every output token is the mean of one disjoint 2x2 block."""
    resampler = SpatialTokenResampler(grid_in=(32, 32), grid_out=(16, 16))
    grid = torch.arange(32 * 32, dtype=torch.float64).reshape(32, 32)
    pooled = resampler(grid.reshape(1, 32 * 32, 1))[0, :, 0].reshape(16, 16)
    expected = grid.reshape(16, 2, 16, 2).mean(dim=(1, 3))
    assert torch.equal(pooled, expected)


def test_the_24_to_16_windows_are_the_documented_uneven_overlapping_ones():
    """Spot-check against the adaptive-pooling window formula stated in the module docstring."""
    resampler = SpatialTokenResampler(grid_in=GRID_IN, grid_out=GRID_OUT)
    grid = torch.arange(24 * 24, dtype=torch.float64).reshape(24, 24)
    pooled = resampler(grid.reshape(1, 24 * 24, 1))[0, :, 0].reshape(16, 16)

    def window(index):
        return (index * 24) // 16, -(-((index + 1) * 24) // 16)

    for row in (0, 1, 7, 15):
        for col in (0, 1, 8, 15):
            top, bottom = window(row)
            left, right = window(col)
            assert pooled[row, col] == grid[top:bottom, left:right].mean()
    # Row 1 reads rows 1..3 and row 2 reads rows 3..4: adjacent windows share input row 3.
    assert window(1) == (1, 3) and window(2) == (3, 5)


def test_row_major_token_order_is_assumed_and_preserved():
    """A single hot token must land in the pooled cell its row/column maps to, not elsewhere."""
    resampler = SpatialTokenResampler(grid_in=GRID_IN, grid_out=GRID_OUT)
    features = torch.zeros(1, 576, 1, dtype=torch.float64)
    features[0, 3 * 24 + 20, 0] = 1.0  # input row 3, column 20
    pooled = resampler(features)[0, :, 0].reshape(16, 16)
    assert torch.nonzero(pooled).tolist() == [[2, 13]]  # 3 in rows 3..5, 20 in columns 20..22


# --------------------------------------------------------------------------------------
# independence across the axes the probes rely on
# --------------------------------------------------------------------------------------

def test_temporal_blocks_do_not_mix():
    """Pooling a 4-block clip must equal pooling each block on its own."""
    resampler = SpatialTokenResampler(grid_in=GRID_IN, grid_out=GRID_OUT)
    blocks = [_ramp(GRID_IN) + offset for offset in (0.0, 100.0, 200.0, 300.0)]
    together = resampler(torch.cat(blocks, dim=1))
    apart = torch.cat([resampler(block) for block in blocks], dim=1)
    assert torch.equal(together, apart)


def test_channels_do_not_mix_so_view_fusion_commutes_with_pooling():
    """AGENTS section 7: views are fused on the channel axis, and pooling must not touch it."""
    resampler = SpatialTokenResampler(grid_in=GRID_IN, grid_out=GRID_OUT)
    generator = torch.Generator().manual_seed(1234)
    views = [torch.randn(2, 4 * 576, 5, dtype=torch.float64, generator=generator) for _ in range(2)]
    fused_then_pooled = resampler(torch.cat(views, dim=2))
    pooled_then_fused = torch.cat([resampler(view) for view in views], dim=2)
    assert torch.equal(fused_then_pooled, pooled_then_fused)


def test_it_is_deterministic_and_leaves_its_input_alone():
    resampler = SpatialTokenResampler(grid_in=GRID_IN, grid_out=GRID_OUT)
    features = torch.randn(2, 4 * 576, 3, generator=torch.Generator().manual_seed(7))
    before = features.clone()
    assert torch.equal(resampler(features), resampler(features))
    assert torch.equal(features, before)


def test_dtype_and_device_survive():
    resampler = SpatialTokenResampler(grid_in=GRID_IN, grid_out=GRID_OUT)
    features = torch.zeros(1, 576, 3, dtype=torch.float32)
    pooled = resampler(features)
    assert pooled.dtype == features.dtype and pooled.device == features.device


# --------------------------------------------------------------------------------------
# shape contract
# --------------------------------------------------------------------------------------

def test_a_token_count_that_is_not_a_whole_number_of_blocks_is_refused():
    resampler = SpatialTokenResampler(grid_in=GRID_IN, grid_out=GRID_OUT)
    with pytest.raises(ValueError, match="not a multiple of"):
        resampler(torch.zeros(1, 576 + 1, 3))


@pytest.mark.parametrize("shape", [(576, 3), (1, 4, 576, 3)])
def test_non_three_dimensional_features_are_refused(shape):
    resampler = SpatialTokenResampler(grid_in=GRID_IN, grid_out=GRID_OUT)
    with pytest.raises(ValueError, match=r"\[B, tokens, D\]"):
        resampler(torch.zeros(shape))
