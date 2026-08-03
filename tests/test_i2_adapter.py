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
# batched fusion: upstream's is only correct at batch size 1
# --------------------------------------------------------------------------------------

def _batch(cfg, clip_seeds):
    """[B, V, T, C, H, W] with both views of a clip identical, so correct fusion is checkable exactly.

    Per-clip identical views make the invariant bitwise: view v of clip b must land in channel block
    v, so the two blocks of a row are equal exactly when the row was paired with its own clip.
    """
    return np.concatenate([_clips(cfg, (seed,) * NUM_VIEWS) for seed in clip_seeds], axis=0)


@pytest.fixture
def corrected(adapter):
    """The same adapter with per-clip view pairing, restored afterwards (the fixture is module-wide)."""
    adapter.correct_view_fusion = True
    yield adapter
    adapter.correct_view_fusion = False


def test_the_two_fusions_agree_at_batch_size_one(adapter, cfg):
    """The pinned path is exact at batch 1, which is why the defect never showed up in I2."""
    clips = _clips(cfg, (1, 2))
    upstream = adapter.encode_video(clips)
    adapter.correct_view_fusion = True
    try:
        assert torch.equal(upstream, adapter.encode_video(clips))
    finally:
        adapter.correct_view_fusion = False


def test_the_default_fusion_mispairs_clips_and_views_above_batch_one(adapter, cfg):
    """Pins the upstream defect rather than hiding it: identical views per clip still fuse unequal.

    `chunk(dim=0)` cuts the batch axis in half, but views are the minor axis of the `[B, V] -> [B*V]`
    flatten, so row b gets one view from clip b and one from another clip entirely.
    """
    dim = adapter.hidden_size
    features = adapter.encode_video(_batch(cfg, (1, 2)))
    assert features.shape[0] == 2
    for row in range(2):
        assert not torch.equal(features[row, :, :dim], features[row, :, dim:]), (
            f"row {row} fused two equal views: upstream's defect is gone, so this test is stale"
        )


def test_correct_view_fusion_pairs_every_row_with_its_own_clip(corrected, cfg):
    dim = corrected.hidden_size
    features = corrected.encode_video(_batch(cfg, (1, 2, 3)))
    assert features.shape[0] == 3
    for row in range(3):
        assert torch.equal(features[row, :, :dim], features[row, :, dim:])
    # Distinct clips must still be distinct rows: equal blocks alone would also pass on a constant.
    assert not torch.equal(features[0], features[1])


def test_correct_view_fusion_reproduces_per_clip_forwards(corrected, cfg):
    """End-to-end statement of the same claim, against clips whose two views really differ.

    `allclose` rather than `equal`: batching changes the encoder's GEMM shapes, so the two runs
    differ in the low bits. The mispairing it rules out is three orders of magnitude larger.
    """
    batch = np.concatenate([_clips(cfg, seeds) for seeds in ((1, 2), (3, 4), (5, 6))], axis=0)
    batched = corrected.encode_video(batch)
    per_clip = torch.cat([corrected.encode_video(batch[row : row + 1]) for row in range(3)], dim=0)
    assert torch.allclose(batched, per_clip, atol=1e-3, rtol=0.0)

    corrected.correct_view_fusion = False
    assert (corrected.encode_video(batch) - per_clip).abs().max() > 1.0


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
