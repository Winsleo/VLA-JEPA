"""I2 parity gate: the interface clean-up must not move a single number.

`I2 Interface Parity` is allowed to rename, wrap, document and log; it is not allowed to change
forward outputs, the trainable-parameter updates, or the optimizer/scheduler step semantics
(AGENTS.md §9, D-011). This module compares a deterministic probe of the training path against
`tests/data/i2_parity_baseline.json`, generated on the revision that opened the iteration:

    CUDA_VISIBLE_DEVICES=0 python tests/tools/gen_parity_golden.py

Comparison is bit-wise (floats are stored as `float.hex()`), not tolerance-based: the frozen
teacher config has all dropout/drop-path rates at 0.0 and `no_grad()` was already in place, so
the planned changes are numerically identity operations. A difference here means a real
behavioural change and must be reported, not absorbed by widening a threshold (D-028).

The two documented exceptions are asserted as such rather than compared blindly:
  - the trainable set loses the V-JEPA teacher (C1, AGENTS §6)
  - the `base` optimizer group loses the same parameters

Multi-GPU DeepSpeed runs are not bit-wise reproducible (D-034) and are compared separately.

Requires one visible GPU, the local weights and the published LIBERO checkpoint; skipped
otherwise. Run:  CUDA_VISIBLE_DEVICES=0 pytest tests/test_i2_parity.py -v
"""

import ast
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from parity_probe import (
    CONFIG_PATH,
    PUBLISHED_LIBERO_CKPT,
    collect_probe,
    env_fingerprint,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "tests" / "data" / "i2_parity_baseline.json"

# Upstream optimisation semantics of the two trainers, pinned so a later commit cannot change
# them silently. train_starvla.py is the entry used by scripts/vlajepa_robot_ft.sh (S3).
EXPECTED_STEPS_PER_BATCH = {
    "train_starvla.py": ("VLATrainer", 1),
    "train_vlajepa_cotrain.py": ("VLAMTrainer", 2),
}

# Effective value of the flow-matching batch repeat. The config declares 8 under
# framework.action_model, but VLA_JEPA.forward() reads config.trainer, so the default applies.
# Locked here; the mismatch is scheduled for I4.5 Upstream Realignment (D-041).
EFFECTIVE_REPEATED_DIFFUSION_STEPS = 4


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f"missing parity golden: {GOLDEN_PATH}")
    payload = json.loads(GOLDEN_PATH.read_text())
    current = env_fingerprint()
    if current != payload["env"]:
        pytest.skip(f"golden captured on a different environment: {payload['env']} != {current}")
    return payload


@pytest.fixture(scope="module")
def cfg():
    return OmegaConf.load(CONFIG_PATH)


@pytest.fixture(scope="module")
def probe(golden, cfg):
    """One probe run shared by every comparison below (model build + short run is minutes)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    for path in (cfg.framework.qwenvl.base_vlm, cfg.framework.vj2_model.base_encoder):
        if not Path(path).exists():
            pytest.skip(f"missing local weights: {path}")
    if not PUBLISHED_LIBERO_CKPT.exists():
        pytest.skip(f"missing published checkpoint: {PUBLISHED_LIBERO_CKPT}")
    return collect_probe()


def _assert_hex_equal(label, actual, expected):
    if actual == expected:
        return
    a, e = float.fromhex(actual), float.fromhex(expected)
    rel = abs(a - e) / abs(e) if e else float("inf")
    raise AssertionError(f"{label} moved: {a!r} vs golden {e!r} (relative {rel:.3e})")


# --------------------------------------------------------------------------------------
# forward and inference parity
# --------------------------------------------------------------------------------------

def test_forward_losses_match_golden(probe, golden):
    """Both named losses, on both probe batches, bit-wise."""
    assert set(probe["forward"]) == set(golden["forward"])
    for video_seed, losses in golden["forward"].items():
        assert set(probe["forward"][video_seed]) == set(losses), (
            f"loss names changed for video_seed={video_seed}: "
            f"{sorted(probe['forward'][video_seed])} vs {sorted(losses)}"
        )
        for name, expected in losses.items():
            _assert_hex_equal(f"forward[{video_seed}].{name}", probe["forward"][video_seed][name], expected)


def test_predict_action_matches_golden(probe, golden):
    """Fast Policy output, byte-identical. I2 does not touch the action head."""
    assert probe["predict_action"] == golden["predict_action"]


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------

def test_world_path_geometry_matches_golden(probe, golden):
    """C2 makes grid/temporal-block geometry explicit; the derived numbers may not change."""
    assert probe["geometry"] == golden["geometry"]


def test_predictor_geometry_agrees_with_encoder(probe):
    """The parity assertion C2 turns into a runtime assert: both paths must derive the same grid."""
    encoder = probe["geometry"]["encoder"]
    predictor = probe["geometry"]["predictor"]
    assert predictor["grid"] == encoder["grid"], (encoder["grid"], predictor["grid"])
    assert encoder["grid"][0] == encoder["grid"][1], "non-square grid: Block(grid_size=...) is scalar"
    assert predictor["num_frames"] == encoder["num_temporal_blocks"]
    assert encoder["tokens_per_block"] == encoder["grid"][0] * encoder["grid"][1]


# --------------------------------------------------------------------------------------
# checkpoint compatibility and the trainable set
# --------------------------------------------------------------------------------------

def test_state_dict_keys_match_golden(probe, golden):
    """Checkpoint compatibility. The adapter must not re-prefix or duplicate any parameter."""
    for field in ("state_dict_num_keys", "state_dict_keys_sha256", "num_parameters"):
        assert probe["parameters"][field] == golden["parameters"][field], field


def test_trainable_set_is_all_params_or_all_but_teacher(probe, golden):
    """Freezing the V-JEPA teacher is the one documented change to the trainable set (C1).

    Written as an invariant instead of a golden comparison so it holds on both sides of C1:
    before, everything is trainable; after, exactly the teacher is gone and nothing else.
    """
    params = probe["parameters"]
    total = params["num_parameters"]
    teacher = params["teacher_num_parameters"]
    trainable = params["num_trainable_parameters"]
    assert total == golden["parameters"]["num_parameters"]
    assert teacher == golden["parameters"]["teacher_num_parameters"]
    assert trainable in (total, total - teacher), (
        f"trainable={trainable} is neither all ({total}) nor all-but-teacher ({total - teacher})"
    )
    if trainable == total - teacher:
        assert params["num_teacher_trainable_parameters"] == 0


def test_optimizer_param_groups_match_golden(probe, golden):
    """Group names, order and learning rates are parity; only the frozen teacher may leave."""
    actual = probe["param_groups"]["groups"]
    expected = golden["param_groups"]["groups"]
    assert [g["name"] for g in actual] == [g["name"] for g in expected]
    for got, want in zip(actual, expected):
        _assert_hex_equal(f"lr[{want['name']}]", got["lr"], want["lr"])

    teacher = probe["parameters"]["teacher_num_parameters"]
    for got, want in zip(actual, expected):
        allowed = (want["numel"], want["numel"] - teacher)
        assert got["numel"] in allowed, (
            f"group {want['name']} numel={got['numel']}, expected {want['numel']} "
            f"or {want['numel'] - teacher} (teacher frozen)"
        )


# --------------------------------------------------------------------------------------
# short optimisation run
# --------------------------------------------------------------------------------------

def test_short_run_step_counts_match_golden(probe, golden):
    """One optimizer step and one scheduler step per batch, and the same warmup position."""
    for field in ("steps", "optimizer_steps", "scheduler_last_epoch"):
        assert probe["short_run"][field] == golden["short_run"][field], field
    assert probe["short_run"]["optimizer_steps"] == probe["short_run"]["steps"]
    assert probe["short_run"]["scheduler_last_epoch"] == probe["short_run"]["steps"]
    for index, (got, want) in enumerate(zip(probe["short_run"]["last_lr"], golden["short_run"]["last_lr"])):
        _assert_hex_equal(f"last_lr[{index}]", got, want)


def test_short_run_losses_match_golden(probe, golden):
    """Per-step losses, bit-wise: the same batch must take the same optimisation trajectory."""
    actual = probe["short_run"]["step_losses"]
    expected = golden["short_run"]["step_losses"]
    assert len(actual) == len(expected)
    for step, (got, want) in enumerate(zip(actual, expected)):
        assert set(got["losses"]) == set(want["losses"]), f"step {step} loss names changed"
        for name, value in want["losses"].items():
            _assert_hex_equal(f"step{step}.{name}", got["losses"][name], value)
        _assert_hex_equal(f"step{step}.total", got["total"], want["total"])


def test_short_run_parameter_updates_match_golden(probe, golden):
    """Per-module parameter fingerprints after the run: the actual parameter-update parity check."""
    assert set(probe["short_run"]["modules"]) == set(golden["short_run"]["modules"])
    for module, want in golden["short_run"]["modules"].items():
        got = probe["short_run"]["modules"][module]
        assert got["num_params"] == want["num_params"], module
        _assert_hex_equal(f"{module}.param_sum", got["param_sum"], want["param_sum"])
        _assert_hex_equal(f"{module}.param_abs_sum", got["param_abs_sum"], want["param_abs_sum"])


# --------------------------------------------------------------------------------------
# pinned upstream semantics (no GPU needed)
# --------------------------------------------------------------------------------------

def _train_step_ast(filename, class_name):
    tree = ast.parse((REPO_ROOT / "starVLA" / "training" / filename).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == "_train_step":
                    return member
    raise AssertionError(f"{filename}: {class_name}._train_step not found")


def _count_step_calls(node, attribute):
    return sum(
        1
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "step"
        and isinstance(child.func.value, ast.Attribute)
        and child.func.value.attr == attribute
    )


@pytest.mark.parametrize(("filename", "class_name", "expected"), [
    (name, *value) for name, value in EXPECTED_STEPS_PER_BATCH.items()
])
def test_train_step_optimizer_semantics_are_pinned(filename, class_name, expected):
    """D-011: I2 must not change how many optimizer/scheduler steps a batch takes.

    Checked on the source instead of by running the trainer, because importing either module
    constructs a global `Accelerator` at import time. The counts are what the probe reproduces.
    """
    node = _train_step_ast(filename, class_name)
    assert _count_step_calls(node, "optimizer") == expected, f"{filename}: optimizer.step() count"
    assert _count_step_calls(node, "lr_scheduler") == expected, f"{filename}: lr_scheduler.step() count"


def test_repeated_diffusion_steps_effective_value_is_pinned(cfg):
    """The config value and the value the code reads live in different nodes (D-041).

    `framework.action_model.repeated_diffusion_steps` is declared but never read; VLA_JEPA reads
    `trainer.repeated_diffusion_steps`, which no config sets, so the literal default wins. Fixing
    the mismatch would change the effective batch repeat and is deferred to I4.5.
    """
    assert cfg.framework.action_model.repeated_diffusion_steps == 8
    assert cfg.trainer.get("repeated_diffusion_steps") is None
    effective = cfg.trainer.get("repeated_diffusion_steps", EFFECTIVE_REPEATED_DIFFUSION_STEPS)
    assert effective == EFFECTIVE_REPEATED_DIFFUSION_STEPS

    source = (REPO_ROOT / "starVLA" / "model" / "framework" / "VLA_JEPA.py").read_text()
    assert 'self.config.trainer.get("repeated_diffusion_steps", 4)' in source
