"""Unit tests for the I3 additions to VJBackboneAdapter: geometry fallback + optional resampler.

Two halves. The first needs no weights and no GPU: it pins the geometry-source rule that lets one
adapter serve V-JEPA 2 (`config.image_size`) and V-JEPA 2.1 (`config.crop_size` only, `image_size`
absent), and the wiring of the optional `SpatialTokenResampler`. The second runs the vendored V-JEPA
2.1 classes on the local weights and measures the 384 path end to end.

The 256 path is deliberately untouched by all of this and stays covered by tests/test_i2_adapter.py.

Run:  CUDA_VISIBLE_DEVICES=0 pytest tests/test_i3_adapter_vjepa21.py -v
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from starVLA.model.modules.world_model.spatial_token_resampler import SpatialTokenResampler
from starVLA.model.modules.world_model.vj_backbone_adapter import VJBackboneAdapter, resolve_input_size

# Vendored-port weights, kept outside the repository (docs/provenance/teachers.md).
VJEPA21_WEIGHTS = Path("/vepfs/wangshilong/models/dynaweave/vjepa21/port_apiantonio")
# Matched field of view: 438 -> 384 and 292 -> 256 share the crop ratio 0.8767, so both teacher
# arms see the same crop through the same pinned processor class (see the module's VENDOR.md).
SHORTEST_EDGE_384 = 438

NUM_FRAMES = 8
NUM_VIEWS = 2
SOURCE_SIZE = 256  # LIBERO records 256x256; the processor does the resize to the teacher's grid.

NATIVE_GRID = (24, 24)
MATCHED_GRID = (16, 16)


def _stub_encoder(**config_fields):
    """Minimal stand-in: the adapter only reads `.config` and freezes the module in `__init__`."""
    encoder = torch.nn.Linear(2, 2)
    encoder.config = SimpleNamespace(patch_size=16, tubelet_size=2, hidden_size=1024, **config_fields)
    return encoder


def _adapter(resampler=None, **config_fields):
    return VJBackboneAdapter(
        encoder=_stub_encoder(**config_fields),
        processor=None,
        num_frames=NUM_FRAMES,
        resampler=resampler,
    )


def _clips(seeds, size=SOURCE_SIZE):
    """[1, len(seeds), T, C, H, W] uint8: one batch element, one distinct clip per view."""
    views = [
        np.random.default_rng(seed).integers(0, 255, (NUM_FRAMES, 3, size, size), dtype=np.uint8) for seed in seeds
    ]
    return np.stack(views)[None]


# --------------------------------------------------------------------------------------
# geometry source: `image_size`, else `crop_size`, else an explicit error
# --------------------------------------------------------------------------------------

def test_image_size_is_preferred_when_both_keys_exist():
    assert resolve_input_size(SimpleNamespace(image_size=256, crop_size=384)) == 256


def test_crop_size_is_the_fallback_for_the_2_1_config():
    """V-JEPA 2.1 states no `image_size` at all, which must not surface as an AttributeError."""
    assert resolve_input_size(SimpleNamespace(crop_size=384)) == 384


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(),
        SimpleNamespace(patch_size=16),
        # A dict-valued crop_size is what the *video processor* uses; a config stating it that way
        # carries no unambiguous square edge, so it must be reported rather than guessed at.
        SimpleNamespace(crop_size={"height": 384, "width": 384}),
    ],
)
def test_a_config_without_a_usable_input_size_is_reported(config):
    with pytest.raises(ValueError, match="states no input size"):
        resolve_input_size(config)


def test_the_2_1_geometry_is_derived_from_crop_size():
    adapter = _adapter(crop_size=384)
    assert adapter.image_size == 384
    assert adapter.native_grid_size == NATIVE_GRID
    assert adapter.grid_size == NATIVE_GRID
    assert adapter.tokens_per_block == 576
    assert adapter.num_temporal_blocks == NUM_FRAMES // 2


def test_the_2_path_is_unchanged_by_the_fallback():
    adapter = _adapter(image_size=256, crop_size=256)
    assert (adapter.image_size, adapter.grid_size, adapter.tokens_per_block) == (256, MATCHED_GRID, 256)
    assert adapter.resampler is None


def test_a_non_divisible_input_size_is_still_refused():
    with pytest.raises(ValueError, match="multiple of patch_size"):
        _adapter(crop_size=380)


# --------------------------------------------------------------------------------------
# optional resampler wiring
# --------------------------------------------------------------------------------------

def test_without_a_resampler_the_geometry_is_exactly_the_encoder_geometry():
    adapter = _adapter(crop_size=384)
    assert adapter.grid_size == adapter.native_grid_size
    assert adapter.tokens_per_block == adapter.native_tokens_per_block


def test_a_resampler_moves_the_published_grid_but_not_the_native_one():
    adapter = _adapter(SpatialTokenResampler(NATIVE_GRID, MATCHED_GRID), crop_size=384)
    assert (adapter.native_grid_size, adapter.native_tokens_per_block) == (NATIVE_GRID, 576)
    assert (adapter.grid_size, adapter.tokens_per_block) == (MATCHED_GRID, 256)


def test_a_resampler_that_does_not_match_the_encoder_grid_is_refused():
    """Silently pooling the wrong grid would reshape tokens into meaningless positions."""
    with pytest.raises(ValueError, match="encoder emits"):
        _adapter(SpatialTokenResampler(MATCHED_GRID, (8, 8)), crop_size=384)


def test_the_teacher_forcing_split_follows_the_post_resample_grid():
    adapter = _adapter(SpatialTokenResampler(NATIVE_GRID, MATCHED_GRID), crop_size=384)
    blocks, tokens = adapter.num_temporal_blocks, adapter.tokens_per_block
    features = torch.arange(blocks, dtype=torch.float32).repeat_interleave(tokens)[None, :, None]

    input_states, gt_states = adapter.split_teacher_forcing(features)
    assert input_states.shape == gt_states.shape == (1, tokens * (blocks - 1), 1)
    assert torch.equal(gt_states, input_states + 1)


# --------------------------------------------------------------------------------------
# the vendored V-JEPA 2.1 teacher on its real weights
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def encoder_and_processor():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if not (VJEPA21_WEIGHTS / "model.safetensors").exists():
        pytest.skip(f"missing local weights: {VJEPA21_WEIGHTS}")

    from transformers import VJEPA2VideoProcessor

    from starVLA.model.modules.world_model.vjepa21 import VJEPA21Config, VJEPA21Model

    config = VJEPA21Config.from_pretrained(VJEPA21_WEIGHTS)
    encoder = VJEPA21Model.from_pretrained(VJEPA21_WEIGHTS, config=config).to("cuda")
    processor = VJEPA2VideoProcessor(
        size={"shortest_edge": SHORTEST_EDGE_384},
        crop_size={"height": config.crop_size, "width": config.crop_size},
    )
    return encoder, processor


@pytest.fixture(scope="module")
def native_adapter(encoder_and_processor):
    encoder, processor = encoder_and_processor
    return VJBackboneAdapter(encoder=encoder, processor=processor, num_frames=NUM_FRAMES)


@pytest.fixture(scope="module")
def matched_adapter(encoder_and_processor):
    encoder, processor = encoder_and_processor
    return VJBackboneAdapter(
        encoder=encoder,
        processor=processor,
        num_frames=NUM_FRAMES,
        resampler=SpatialTokenResampler(NATIVE_GRID, MATCHED_GRID),
    )


def test_the_vendored_config_states_the_expected_geometry(native_adapter):
    config = native_adapter.encoder.config
    assert not hasattr(config, "image_size"), "config gained an image_size; the fallback is now untested"
    assert (config.crop_size, config.patch_size, config.tubelet_size, config.hidden_size) == (384, 16, 2, 1024)


def test_arm_b_encodes_onto_the_native_24x24_grid(native_adapter):
    features = native_adapter.encode_video(_clips((1, 2)))
    assert features.shape == (1, 4 * 576, NUM_VIEWS * 1024)


def test_arm_c_encodes_onto_the_matched_16x16_grid(matched_adapter):
    features = matched_adapter.encode_video(_clips((1, 2)))
    assert features.shape == (1, 4 * 256, NUM_VIEWS * 1024)


def test_arm_c_is_exactly_arm_b_pooled(native_adapter, matched_adapter):
    """The resampler must be the only difference between the two arms, applied after view fusion."""
    resampler = SpatialTokenResampler(NATIVE_GRID, MATCHED_GRID)
    clips = _clips((1, 2))
    assert torch.equal(matched_adapter.encode_video(clips), resampler(native_adapter.encode_video(clips)))


def test_views_are_concatenated_in_order_on_the_feature_axis(matched_adapter):
    """AGENTS section 7: swapping the input views must swap the two halves of the feature axis."""
    dim = matched_adapter.hidden_size
    straight = matched_adapter.encode_video(_clips((1, 2)))
    swapped = matched_adapter.encode_video(_clips((2, 1)))
    assert torch.equal(straight[..., :dim], swapped[..., dim:])
    assert torch.equal(straight[..., dim:], swapped[..., :dim])
    assert not torch.equal(straight[..., :dim], straight[..., dim:]), "distinct clips fused identically"


def test_the_2_1_teacher_is_frozen_and_stays_in_eval(matched_adapter):
    assert [name for name, param in matched_adapter.encoder.named_parameters() if param.requires_grad] == []
    assert not matched_adapter.encoder.training

    matched_adapter.encoder.train()
    for param in matched_adapter.encoder.parameters():
        param.requires_grad_(True)
    matched_adapter.enforce_frozen()
    assert not matched_adapter.encoder.training
    assert [name for name, param in matched_adapter.encoder.named_parameters() if param.requires_grad] == []


def test_features_carry_no_gradient(matched_adapter):
    assert not matched_adapter.encode_video(_clips((1, 2))).requires_grad


def test_encoding_the_same_clip_twice_is_bitwise_identical(matched_adapter):
    """Probe features are cached once and reused across seeds, so drift here would be invisible."""
    clips = _clips((1, 2))
    assert torch.equal(matched_adapter.encode_video(clips), matched_adapter.encode_video(clips))
