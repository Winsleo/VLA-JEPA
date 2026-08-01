"""Unit tests for VJBackboneAdapter, the frozen V-JEPA teacher seam introduced in I2.

Built directly from the local encoder weights instead of the full VLA_JEPA framework: the
adapter is the unit under test, and keeping it independently testable is the reason it exists
(engineering-guidelines, D-019). The framework-level parity of the same code path is covered by
tests/test_i2_parity.py.

Requires one visible GPU and the local V-JEPA weights; skipped otherwise.
Run:  CUDA_VISIBLE_DEVICES=0 pytest tests/test_i2_adapter.py -v
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from parity_probe import CONFIG_PATH, SEED

NUM_VIEWS = 2


@pytest.fixture(scope="module")
def cfg():
    return OmegaConf.load(CONFIG_PATH)


@pytest.fixture(scope="module")
def adapter(cfg):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    encoder_path = cfg.framework.vj2_model.base_encoder
    if not Path(encoder_path).exists():
        pytest.skip(f"missing local weights: {encoder_path}")

    from transformers import AutoModel, AutoVideoProcessor

    from starVLA.model.modules.world_model.vj_backbone_adapter import VJBackboneAdapter

    encoder = AutoModel.from_pretrained(encoder_path).to("cuda")
    processor = AutoVideoProcessor.from_pretrained(encoder_path)
    return VJBackboneAdapter(
        encoder=encoder,
        processor=processor,
        num_frames=cfg.framework.vj2_model.num_frames,
    )


def _clips(cfg, seeds):
    """[1, len(seeds), T, C, H, W] uint8: one batch element, one distinct clip per view."""
    size = cfg.datasets.vla_data.video_resolution_size
    frames = cfg.framework.vj2_model.num_frames
    views = [
        np.random.default_rng(seed).integers(0, 255, (frames, 3, size, size), dtype=np.uint8) for seed in seeds
    ]
    return np.stack(views)[None]


# --------------------------------------------------------------------------------------
# firewall
# --------------------------------------------------------------------------------------

def test_encoder_is_frozen_and_in_eval(adapter):
    assert [n for n, p in adapter.encoder.named_parameters() if p.requires_grad] == []
    assert not adapter.encoder.training


def test_enforce_frozen_restores_the_firewall(adapter):
    adapter.encoder.train()
    for param in adapter.encoder.parameters():
        param.requires_grad_(True)
    adapter.enforce_frozen()
    assert not adapter.encoder.training
    assert [n for n, p in adapter.encoder.named_parameters() if p.requires_grad] == []


def test_features_carry_no_gradient(adapter, cfg):
    features = adapter.encode_video(_clips(cfg, (SEED,) * NUM_VIEWS))
    assert not features.requires_grad


# --------------------------------------------------------------------------------------
# geometry contract
# --------------------------------------------------------------------------------------

def test_geometry_matches_the_encoder_config(adapter, cfg):
    config = adapter.encoder.config
    grid = config.image_size // config.patch_size
    assert adapter.grid_size == (grid, grid)
    assert adapter.tokens_per_block == grid * grid
    assert adapter.num_temporal_blocks == cfg.framework.vj2_model.num_frames // config.tubelet_size
    assert adapter.hidden_size == config.hidden_size


def test_encode_video_shape_follows_the_geometry(adapter, cfg):
    features = adapter.encode_video(_clips(cfg, (1, 2)))
    assert features.shape == (
        1,
        adapter.num_temporal_blocks * adapter.tokens_per_block,
        NUM_VIEWS * adapter.hidden_size,
    )


def test_encode_video_rejects_a_wrong_clip_length(adapter, cfg):
    clips = _clips(cfg, (1, 2))
    with pytest.raises(ValueError, match="frames"):
        adapter.encode_video(clips[:, :, :-2])


# --------------------------------------------------------------------------------------
# multi-view fusion (AGENTS §7: view order must be consistent)
# --------------------------------------------------------------------------------------

def test_views_are_concatenated_in_order_on_the_feature_axis(adapter, cfg):
    """Swapping the two input views must swap the two halves of the feature axis, exactly."""
    dim = adapter.hidden_size
    straight = adapter.encode_video(_clips(cfg, (1, 2)))
    swapped = adapter.encode_video(_clips(cfg, (2, 1)))
    assert torch.equal(straight[..., :dim], swapped[..., dim:])
    assert torch.equal(straight[..., dim:], swapped[..., :dim])
    assert not torch.equal(straight[..., :dim], straight[..., dim:]), "distinct clips fused identically"


# --------------------------------------------------------------------------------------
# teacher forcing split
# --------------------------------------------------------------------------------------

def test_split_teacher_forcing_shifts_by_one_temporal_block(adapter):
    """Block i of the input must line up with block i+1 of the target."""
    tokens, blocks = adapter.tokens_per_block, adapter.num_temporal_blocks
    # Feature value == block index, so the shift is directly readable.
    features = torch.arange(blocks, dtype=torch.float32).repeat_interleave(tokens)[None, :, None]

    input_states, gt_states = adapter.split_teacher_forcing(features)
    assert input_states.shape == gt_states.shape == (1, tokens * (blocks - 1), 1)
    assert torch.equal(gt_states, input_states + 1)
    assert input_states[0, 0, 0] == 0
    assert gt_states[0, -1, 0] == blocks - 1


def test_split_teacher_forcing_rejects_a_wrong_token_count(adapter):
    tokens, blocks = adapter.tokens_per_block, adapter.num_temporal_blocks
    with pytest.raises(AssertionError, match="feature tokens"):
        adapter.split_teacher_forcing(torch.zeros(1, tokens * blocks - 1, 1))


def test_split_matches_the_upstream_arithmetic(adapter, cfg):
    """Explicit geometry must reproduce the implicit `shape[1] // T` slicing it replaced."""
    features = adapter.encode_video(_clips(cfg, (1, 2)))
    input_states, gt_states = adapter.split_teacher_forcing(features)

    blocks = adapter.num_temporal_blocks
    per_state = features.shape[1] // blocks
    assert torch.equal(input_states, features[:, : per_state * (blocks - 1), :])
    assert torch.equal(gt_states, features[:, per_state:, :])
