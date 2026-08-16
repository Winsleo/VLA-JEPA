"""Gradient logging helpers and the teacher firewall check (I2 C4).

AGENTS.md section 10 step 8 asks for key gradient norms to be recorded, and section 6 asks for the
frozen teacher to be provably gradient-free. Both are read-only helpers, so they are tested here on
a toy module instead of the 2.8B framework; the trainer only orchestrates them.

Pure logic, CPU-only.
"""

import ast

import pytest
import torch
import torch.nn as nn

from starVLA.training.trainer_utils.trainer_tools import (
    METRIC_PREFIX,
    frozen_module_has_no_gradient,
    global_grad_norm,
    module_grad_norms,
)


class _Toy(nn.Module):
    """Two trainable submodules plus a frozen one, mirroring the framework's layout."""

    def __init__(self):
        super().__init__()
        self.action_model = nn.Linear(2, 2, bias=False)
        self.vj_predictor = nn.Linear(2, 2, bias=False)
        self.vj_encoder = nn.Linear(2, 2, bias=False)
        self.vj_encoder.requires_grad_(False)


def _toy_with_grads():
    model = _Toy()
    for module in (model.action_model, model.vj_predictor):
        module.weight.grad = torch.full_like(module.weight, 0.5)  # norm = sqrt(4 * 0.25) = 1.0
    return model


PREFIXES = ("action_model", "vj_predictor")


def test_reports_one_norm_and_one_count_per_module():
    metrics = module_grad_norms(_toy_with_grads(), PREFIXES)
    assert sorted(metrics) == sorted(
        f"{METRIC_PREFIX}{field}/{prefix}" for prefix in PREFIXES for field in ("grad_norm", "grad_tensors")
    )
    for prefix in PREFIXES:
        assert metrics[f"{METRIC_PREFIX}grad_norm/{prefix}"] == 1.0
        assert metrics[f"{METRIC_PREFIX}grad_tensors/{prefix}"] == 1.0


def test_norm_matches_torch_over_the_matched_parameters():
    model = _toy_with_grads()
    model.action_model.weight.grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    metrics = module_grad_norms(model, PREFIXES)
    expected = model.action_model.weight.grad.norm(2).item()
    assert metrics[f"{METRIC_PREFIX}grad_norm/action_model"] == pytest.approx(expected, rel=0, abs=1e-12)


def test_a_module_without_gradients_reports_zero_and_no_tensors():
    """A frozen or not-yet-backwarded module must read as 0.0 / 0 tensors, not go missing."""
    metrics = module_grad_norms(_Toy(), PREFIXES)
    for prefix in PREFIXES:
        assert metrics[f"{METRIC_PREFIX}grad_norm/{prefix}"] == 0.0
        assert metrics[f"{METRIC_PREFIX}grad_tensors/{prefix}"] == 0.0


def test_wrapper_prefixes_do_not_hide_a_module():
    """DDP/DeepSpeed/compile add `module.` / `_orig_mod.`; plain module paths must still match."""
    wrapped = nn.ModuleDict({"module": _toy_with_grads()})
    metrics = module_grad_norms(wrapped, PREFIXES)
    assert metrics[f"{METRIC_PREFIX}grad_tensors/action_model"] == 1.0


def test_prefixes_match_whole_module_names_only():
    """`vj` must not absorb `vj_predictor`, otherwise a norm silently covers the wrong module."""
    metrics = module_grad_norms(_toy_with_grads(), ("vj", "vj_predictor"))
    assert metrics[f"{METRIC_PREFIX}grad_tensors/vj"] == 0.0
    assert metrics[f"{METRIC_PREFIX}grad_tensors/vj_predictor"] == 1.0


# --------------------------------------------------------------------------------------
# total norm across backends
# --------------------------------------------------------------------------------------

class _FakeAccelerator:
    """Only the attribute path `global_grad_norm` reads; no accelerate import needed."""

    def __init__(self, engine_norm=None):
        if engine_norm is not None:
            self.deepspeed_engine_wrapped = type("W", (), {"engine": type("E", (), {
                "get_global_grad_norm": staticmethod(lambda: engine_norm)
            })()})()


def test_clip_return_is_used_when_the_backend_provides_one():
    assert global_grad_norm(_FakeAccelerator(), torch.tensor(2.5)) == 2.5


def test_deepspeed_engine_norm_is_used_when_clipping_returns_none():
    """accelerate returns None under DeepSpeed because the engine clips internally."""
    assert global_grad_norm(_FakeAccelerator(engine_norm=3.25), None) == 3.25


def test_no_total_norm_is_reported_when_no_backend_exposes_one():
    """None, not 0.0: a wrong zero would read as a vanished gradient."""
    assert global_grad_norm(_FakeAccelerator(), None) is None
    assert global_grad_norm(_FakeAccelerator(engine_norm=None), None) is None


# --------------------------------------------------------------------------------------
# gradient firewall
# --------------------------------------------------------------------------------------

def test_frozen_module_check_passes_on_a_frozen_teacher():
    assert frozen_module_has_no_gradient(_toy_with_grads().vj_encoder)


def test_frozen_module_check_fails_when_the_teacher_is_trainable_again():
    model = _Toy()
    model.vj_encoder.requires_grad_(True)
    assert not frozen_module_has_no_gradient(model.vj_encoder)


def test_frozen_module_check_fails_when_a_stale_gradient_is_attached():
    """requires_grad=False alone is not enough: a leftover buffer would still be optimized."""
    model = _Toy()
    model.vj_encoder.weight.grad = torch.zeros_like(model.vj_encoder.weight)
    assert not frozen_module_has_no_gradient(model.vj_encoder)


# --------------------------------------------------------------------------------------
# trainer wiring
# --------------------------------------------------------------------------------------

def _train_starvla_source():
    """Read the source: importing train_starvla.py constructs a global Accelerator."""
    from pathlib import Path

    return (Path(__file__).resolve().parents[1] / "starVLA" / "training" / "train_starvla.py").read_text()


def test_gradient_diagnostics_run_before_the_optimizer_consumes_the_gradients():
    source = _train_starvla_source()
    order = [
        "clip_result = self.accelerator.clip_grad_norm_(",
        "self._gradient_metrics(clip_result) if self._is_logging_step() else {}",
        "self.optimizer.step()",
    ]
    positions = [source.find(fragment) for fragment in order]
    assert all(position > 0 for position in positions), positions
    assert positions == sorted(positions), "gradient norms must be read before optimizer.step()"


def test_the_frozen_teacher_is_checked_rather_than_logged():
    """The teacher must never appear in the grad-norm list; a trainable module legitimately may.

    Asserted as membership rather than as the literal tuple. The literal form broke the moment I4
    added `depth_delta_head`, which is exactly the kind of change this test should allow, while the
    thing it must forbid -- logging a norm for the frozen encoder instead of asserting it has none --
    was only pinned by accident.
    """
    source = _train_starvla_source()
    assignment = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "GRAD_NORM_MODULES" for target in node.targets)
    )
    modules = set(ast.literal_eval(assignment.value))
    assert "vj_encoder" not in modules, "the frozen teacher must be checked, not logged"
    assert {"qwen_vl_interface", "action_model", "vj_predictor"}.issubset(modules)
    assert "frozen_module_has_no_gradient(teacher)" in source
