"""I1-S1 smoke tests for the pinned VLA-JEPA baseline.

Scope (see DynaWeave/docs/implementation-plan.md §10 and AGENTS.md §11):
  - runtime shape contract of the world-model path
  - gradient/eval firewall state of the frozen target encoder (AGENTS §6)
  - deterministic forward, no NaN/Inf
  - future-substitution invariance: swapping the future clip must not move action outputs
  - Fast Policy graph excludes the world/depth modules

These tests characterise the BASELINE as it is. Baseline gaps were recorded as strict xfail
instead of being weakened, so they stayed visible until the iteration allowed to fix them; the
two teacher-freeze gaps were closed in I2 by VJBackboneAdapter and now assert normally.

Requires one visible GPU and the local checkpoints; skipped otherwise.
Run:  CUDA_VISIBLE_DEVICES=4 pytest tests/test_i1_smoke.py -v
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from parity_probe import (
    BATCH_SIZE,
    CONFIG_PATH,
    NUM_VIEWS,
    PUBLISHED_LIBERO_CKPT,
    SEED,
    make_examples,
    seeded_forward,
    split_losses,
)


def _require_gpu_and_weights(cfg):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    for key in (cfg.framework.qwenvl.base_vlm, cfg.framework.vj2_model.base_encoder):
        if not Path(key).exists():
            pytest.skip(f"missing local weights: {key}")


@pytest.fixture(scope="module")
def cfg():
    return OmegaConf.load(CONFIG_PATH)


@pytest.fixture(scope="module")
def model(cfg):
    _require_gpu_and_weights(cfg)
    from starVLA.model.framework.VLM4A.VLA_JEPA import VLA_JEPA

    torch.manual_seed(SEED)
    model = VLA_JEPA(cfg).to("cuda")
    model.eval()
    return model


# --------------------------------------------------------------------------------------
# config contract
# --------------------------------------------------------------------------------------

def test_config_matches_published_baseline(cfg):
    """Guards the fixtures above: these are the published LIBERO hyper-parameters."""
    assert cfg.framework.name == "VLA_JEPA"
    assert cfg.framework.qwenvl.vl_hidden_dim == 2048
    assert cfg.framework.action_model.action_dim == 7
    assert cfg.framework.action_model.state_dim == 8
    assert cfg.framework.action_model.action_horizon == 7
    assert cfg.framework.action_model.future_action_window_size == 6
    assert cfg.framework.vj2_model.num_frames == 8
    assert cfg.framework.vj2_model.num_action_tokens_per_timestep == 8
    # Matches the published config: flash-attn 2 is installed from a prebuilt wheel (no nvcc in the
    # image, so building from source is impossible). See docs/provenance/environment.md.
    assert cfg.framework.qwenvl.attn_implementation == "flash_attention_2"


# --------------------------------------------------------------------------------------
# shape contract
# --------------------------------------------------------------------------------------

def test_world_path_shapes(model, cfg):
    """Record the real world-model shapes (implementation-plan §4 is a reference, not a fact)."""
    frames = cfg.framework.vj2_model.num_frames
    vid = cfg.datasets.vla_data.video_resolution_size
    examples = make_examples(cfg, video_seed=1)

    videos = np.stack([e["video"] for e in examples]).transpose(0, 1, 2, 5, 3, 4)
    b, v, t, c, h, w = videos.shape
    assert (b, v, t, c, h, w) == (BATCH_SIZE, NUM_VIEWS, frames, 3, vid, vid)

    flat = videos.reshape(b * v, t, c, h, w)
    processed = torch.cat(
        [
            model.vj_processor(videos=flat[i], return_tensors="pt")["pixel_values_videos"].to("cuda")
            for i in range(b * v)
        ],
        dim=0,
    )
    with torch.no_grad():
        feats = model.vj_encoder.get_vision_features(pixel_values_videos=processed)
        fused = torch.cat(torch.chunk(feats, chunks=v, dim=0), dim=2)

    tubelet = model.vj_encoder.config.tubelet_size
    enc_dim = model.vj_encoder.config.hidden_size
    grid = model.vj_encoder.config.image_size // model.vj_encoder.config.patch_size
    t_states = frames // tubelet

    # per-view dense tokens flattened over (time, grid) -> [B*V, t_states*grid*grid, D]
    assert feats.shape == (b * v, t_states * grid * grid, enc_dim), feats.shape
    # multi-view fusion is a concat on the FEATURE axis -> [B, t_states*grid*grid, V*D]
    assert fused.shape == (b, t_states * grid * grid, v * enc_dim), fused.shape

    per_state = fused.shape[1] // t_states
    input_states = fused[:, : per_state * (t_states - 1), :]
    gt_states = fused[:, per_state:, :]
    assert input_states.shape == gt_states.shape == (b, per_state * (t_states - 1), v * enc_dim)

    print(
        f"\n[shape] tubelet={tubelet} grid={grid}x{grid} t_states={t_states} "
        f"enc_dim={enc_dim} per_state_tokens={per_state} "
        f"encoder_out={tuple(feats.shape)} fused={tuple(fused.shape)} "
        f"world_in/target={tuple(input_states.shape)}"
    )


def test_forward_losses_are_named_and_finite(model, cfg):
    out = seeded_forward(model, make_examples(cfg, video_seed=1))
    # I2 C3 adds log-only diagnostics to the same dict; the optimized half is what is named here.
    losses, metrics = split_losses(out)
    assert set(losses) == {"action_loss", "wm_loss"}
    for name, value in out.items():
        assert value.ndim == 0
        assert torch.isfinite(value), f"{name} is not finite: {value}"
        print(f"\n[{'metric' if name in metrics else 'loss'}] {name} = {value.item():.6f}")


def test_action_free_batch_masks_only_action_loss(model, cfg):
    """AGENTS §7 / D-012: an action-free sample must drop the action loss, not zero-pad actions.

    Run under the robot-only LIBERO config, i.e. without `datasets.video_data`. I1 needed that node
    injected because the action-free branch read it unconditionally; I2 C5 selects the prompt with a
    default, so this is now the real routing rather than a fixture-only one.
    """
    assert "video_data" not in cfg.datasets, "fixture no longer exercises the robot-only config"
    examples = make_examples(cfg, video_seed=1)
    for e in examples:
        e.pop("action")
        e.pop("state")
    out = seeded_forward(model, examples)

    losses, _ = split_losses(out)
    assert set(losses) == {"wm_loss"}
    assert torch.isfinite(out["wm_loss"])


# --------------------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------------------

def test_forward_is_deterministic_under_seed(model, cfg):
    examples = make_examples(cfg, video_seed=1)
    first = seeded_forward(model, examples)
    second = seeded_forward(model, examples)
    for name in first:
        assert torch.equal(first[name], second[name]), (
            f"{name} not bit-wise reproducible: {first[name].item()} vs {second[name].item()}"
        )


# --------------------------------------------------------------------------------------
# information firewall
# --------------------------------------------------------------------------------------

def test_future_substitution_does_not_move_action_loss(model, cfg):
    """Different future clip, identical deployment inputs => identical action loss, different wm loss.

    implementation-plan §10 future-substitution invariant.
    """
    a = seeded_forward(model, make_examples(cfg, video_seed=1))
    b = seeded_forward(model, make_examples(cfg, video_seed=999))
    assert torch.equal(a["action_loss"], b["action_loss"]), (
        f"action loss leaked future frames: {a['action_loss'].item()} vs {b['action_loss'].item()}"
    )
    assert not torch.equal(a["wm_loss"], b["wm_loss"]), "world loss ignored the future clip"


def test_fast_policy_graph_excludes_world_modules(model, cfg):
    """predict_action must not touch the V-JEPA encoder or the world predictor (I1 gate)."""
    examples = make_examples(cfg, video_seed=1)

    calls = []

    def tripwire(name):
        def hook(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"Fast Policy path ran {name}")

        return hook

    enc_handle = model.vj_encoder.register_forward_pre_hook(tripwire("vj_encoder"))
    pred_handle = model.vj_predictor.register_forward_pre_hook(tripwire("vj_predictor"))
    try:
        out = model.predict_action(
            batch_images=[e["image"] for e in examples],
            instructions=[e["lang"] for e in examples],
            state=[e["state"] for e in examples],
        )
    finally:
        enc_handle.remove()
        pred_handle.remove()

    assert calls == []
    actions = out["normalized_actions"]
    chunk = cfg.framework.action_model.future_action_window_size + 1
    assert actions.shape == (BATCH_SIZE, chunk, cfg.framework.action_model.action_dim), actions.shape
    assert np.isfinite(actions).all()


# --------------------------------------------------------------------------------------
# published checkpoint
# --------------------------------------------------------------------------------------

def test_published_libero_checkpoint_loads_strict(model, cfg):
    """The published LIBERO checkpoint must load into the model built from this config.

    This is what makes the config-derived spec (D-027) verifiable rather than inferred: upstream
    loads with strict=True (trainer_tools.load_pretrained_backbones), so any key or shape drift
    would surface here instead of during S2 evaluation.
    """
    if not PUBLISHED_LIBERO_CKPT.exists():
        pytest.skip(f"missing published checkpoint: {PUBLISHED_LIBERO_CKPT}")

    checkpoint = torch.load(PUBLISHED_LIBERO_CKPT, map_location="cpu")
    state = model.state_dict()
    missing = sorted(set(state) - set(checkpoint))
    unexpected = sorted(set(checkpoint) - set(state))
    mismatched = [k for k in set(state) & set(checkpoint) if state[k].shape != checkpoint[k].shape]

    assert not missing, f"{len(missing)} keys absent from checkpoint, e.g. {missing[:3]}"
    assert not unexpected, f"{len(unexpected)} extra checkpoint keys, e.g. {unexpected[:3]}"
    assert not mismatched, f"{len(mismatched)} shape mismatches, e.g. {mismatched[:3]}"
    print(f"\n[ckpt] strict load OK, {len(state)} keys")

    # Restore the freshly-initialised weights: later tests in this module share the fixture and
    # must not silently switch to the fine-tuned checkpoint.
    del checkpoint


# --------------------------------------------------------------------------------------
# gradient firewall (AGENTS §6) — the two freeze gaps were closed in I2 (VJBackboneAdapter)
# --------------------------------------------------------------------------------------

def test_target_encoder_receives_no_gradient(model, cfg):
    """The no_grad() around the teacher forward is the one firewall rule the baseline does keep."""
    model.zero_grad(set_to_none=True)
    out = seeded_forward(model, make_examples(cfg, video_seed=1))
    (out["wm_loss"] + out["action_loss"]).backward()
    grads = [n for n, p in model.vj_encoder.named_parameters() if p.grad is not None]
    assert grads == [], f"gradient reached the frozen teacher: {grads[:5]}"
    model.zero_grad(set_to_none=True)


def test_target_encoder_params_are_frozen(model):
    trainable = [n for n, p in model.vj_encoder.named_parameters() if p.requires_grad]
    assert trainable == [], f"{len(trainable)} teacher params are trainable, e.g. {trainable[:3]}"


def test_target_encoder_stays_in_eval_after_parent_train(model):
    model.train()
    try:
        assert not model.vj_encoder.training, "teacher switched to train() with the parent"
    finally:
        model.eval()
