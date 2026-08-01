"""Loss/metric partition and the configurable world-model weights (I2 C3).

AGENTS.md §10 (step 8) asks for raw and weighted losses to be logged separately, which means the
framework output carries terms that must never reach the backward pass. `split_loss_terms` is that
seam, and every trainer sums only its `losses` half.

Pure logic, CPU-only, no weights: the framework-level parity of the same code path is covered by
tests/test_i2_parity.py.
"""

import torch
from omegaconf import OmegaConf

from starVLA.training.trainer_utils.trainer_tools import METRIC_PREFIX, split_loss_terms

# Upstream literals the config defaults must reproduce: VLA_JEPA weighted its world-model loss by
# 0.1 on the action branch and returned it unweighted (i.e. 1.0) on the action-free branch. The
# asymmetry is pinned behaviour, not a bug to tidy up inside I2.
UPSTREAM_WM_WEIGHT = 0.1
UPSTREAM_WM_ACTION_FREE_WEIGHT = 1.0


def _weights(loss_scale):
    """Read the weights exactly as VLA_JEPA.__init__ does, without building the framework."""
    cfg = OmegaConf.create({"trainer": {"loss_scale": loss_scale} if loss_scale is not None else {}})
    scale = cfg.trainer.get("loss_scale", {})
    return scale.get("wm", UPSTREAM_WM_WEIGHT), scale.get("wm_action_free", UPSTREAM_WM_ACTION_FREE_WEIGHT)


def test_defaults_reproduce_the_upstream_literals():
    """No existing config sets `trainer.loss_scale.wm`, so the defaults decide parity."""
    assert _weights(None) == (UPSTREAM_WM_WEIGHT, UPSTREAM_WM_ACTION_FREE_WEIGHT)
    assert _weights({})[0] == UPSTREAM_WM_WEIGHT


def test_configured_weights_override_the_defaults():
    assert _weights({"wm": 0.25, "wm_action_free": 0.5}) == (0.25, 0.5)


def test_weighting_a_loss_by_the_default_is_bit_wise_identical_to_the_literal():
    """`x * cfg_weight` must be the same float as the removed `x * 0.1` / unweighted return."""
    loss = torch.tensor([1.234567, 0.0, 1e-7, 3.4e30])
    wm_weight, action_free_weight = _weights(None)
    assert torch.equal(loss * wm_weight, loss * 0.1)
    assert torch.equal(loss * action_free_weight, loss)


def test_split_routes_prefixed_keys_to_metrics_only():
    output = {
        "action_loss": torch.tensor(1.0),
        "wm_loss": torch.tensor(2.0),
        f"{METRIC_PREFIX}wm_loss_raw": torch.tensor(20.0),
        f"{METRIC_PREFIX}wm_loss_weight": torch.tensor(0.1),
    }
    losses, metrics = split_loss_terms(output)
    assert sorted(losses) == ["action_loss", "wm_loss"]
    assert sorted(metrics) == [f"{METRIC_PREFIX}wm_loss_raw", f"{METRIC_PREFIX}wm_loss_weight"]
    assert torch.equal(sum(losses.values()), torch.tensor(3.0))


def test_split_is_identity_on_a_pre_c3_output():
    """Outputs without diagnostics keep summing exactly as before, no key left behind."""
    output = {"action_loss": torch.tensor(1.5), "wm_loss": torch.tensor(0.5)}
    losses, metrics = split_loss_terms(output)
    assert losses == output
    assert metrics == {}


def test_no_trainer_sums_a_raw_framework_output():
    """The added keys reach every trainer; a missed `sum(...values())` would optimize a metric.

    train_starvla_cotrain.py indexes the loss it wants by name and is listed to keep it that way.
    """
    from pathlib import Path

    training = Path(__file__).resolve().parents[1] / "starVLA" / "training"
    for filename in (
        "train_starvla.py",
        "train_vlajepa_video.py",
        "train_vlajepa_cotrain.py",
        "train_starvla_cotrain.py",
    ):
        source = (training / filename).read_text()
        for forbidden in ("sum(output_dict.values())", "sum(vlm_output.values())"):
            assert forbidden not in source, f"{filename}: {forbidden} would sum log-only metrics"
