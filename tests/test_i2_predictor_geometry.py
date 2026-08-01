"""Explicit token geometry of VisionTransformerPredictorAC (I2 C2).

The predictor re-derives its grid from `img_size / patch_size / num_frames / tubelet_size`, which
has to agree with the teacher that produces its input states. The optional `grid_size` and
`num_temporal_blocks` arguments let the caller state the measured teacher geometry so a mismatch
raises at construction instead of surfacing as a shape error inside the attention mask.

CPU-only and tiny: this is pure construction logic, no weights involved.
"""

import pytest
import torch

from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC

# Small enough to build in milliseconds; 32 / 16 = a 2x2 grid, 2 temporal blocks.
BASE_KWARGS = dict(
    img_size=(32, 32),
    patch_size=16,
    num_frames=2,
    tubelet_size=1,
    embed_dim=8,
    predictor_embed_dim=8,
    depth=1,
    num_heads=1,
    action_embed_dim=4,
    num_add_tokens=2,
)
GRID = (2, 2)
BLOCKS = 2


def _build(**overrides):
    return VisionTransformerPredictorAC(**{**BASE_KWARGS, **overrides})


def test_explicit_geometry_is_accepted_and_recorded():
    predictor = _build(grid_size=GRID, num_temporal_blocks=BLOCKS)
    assert (predictor.grid_height, predictor.grid_width) == GRID
    assert predictor.grid_depth == BLOCKS


def test_derived_geometry_is_unchanged_when_nothing_is_passed():
    """Backwards compatibility: upstream call sites pass neither argument."""
    explicit = _build(grid_size=GRID, num_temporal_blocks=BLOCKS)
    derived = _build()
    assert (derived.grid_height, derived.grid_width) == (explicit.grid_height, explicit.grid_width)
    assert derived.grid_depth == explicit.grid_depth
    assert derived.attn_mask.shape == explicit.attn_mask.shape


def test_attention_mask_covers_every_token_plus_the_action_tokens():
    predictor = _build(grid_size=GRID, num_temporal_blocks=BLOCKS)
    tokens_per_block = GRID[0] * GRID[1]
    expected = BLOCKS * (tokens_per_block + BASE_KWARGS["num_add_tokens"])
    assert predictor.attn_mask.shape == (expected, expected)


def test_mismatched_grid_size_raises():
    with pytest.raises(ValueError, match="grid_size"):
        _build(grid_size=(4, 4))


def test_mismatched_num_temporal_blocks_raises():
    with pytest.raises(ValueError, match="num_temporal_blocks"):
        _build(num_temporal_blocks=BLOCKS + 1)


def test_non_square_grid_raises_under_rope():
    """RoPE blocks receive a single scalar grid size, so a non-square grid must not pass silently."""
    with pytest.raises(ValueError, match="square grid"):
        _build(img_size=(32, 64))


def test_frame_causal_mask_is_not_a_parameter():
    """attn_mask is a plain attribute, so making the geometry explicit cannot alter checkpoints."""
    predictor = _build(grid_size=GRID, num_temporal_blocks=BLOCKS)
    assert isinstance(predictor.attn_mask, torch.Tensor)
    assert not any("attn_mask" in key for key in predictor.state_dict())
