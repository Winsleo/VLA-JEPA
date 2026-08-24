"""Frozen depth-head condition ablations for the I4 causal gate.

The depth head receives a target-only current depth state and action-token states.  This module
holds the latter fixed except for the prescribed condition interventions, so its errors isolate
whether the head uses the action-token condition rather than the current depth map alone.
"""

from typing import Dict

import torch
from torch import nn


def _validate_inputs(
    current: torch.Tensor,
    tokens: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if current.ndim != 6 or current.shape[3] != 1:
        raise ValueError(f"current must be [B,T,V,1,H,W], got {tuple(current.shape)}")
    if tokens.ndim != 4:
        raise ValueError(f"tokens must be [B,T,Q,H], got {tuple(tokens.shape)}")
    if target.shape != current.shape or mask.shape != current.shape:
        raise ValueError("current, target, and mask must share [B,T,V,1,H,W]")
    if tokens.shape[:2] != current.shape[:2]:
        raise ValueError("tokens and current depth disagree on batch or transition axes")
    if current.shape[0] < 2:
        raise ValueError("cross-sample permutation requires at least two examples")


@torch.no_grad()
def condition_error_sums(
    head: nn.Module,
    current: torch.Tensor,
    tokens: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return globally aggregable absolute-error sums for I4's three depth conditions.

    Args:
        current: Target-only log-depth states `[B, T, V, 1, H, W]`.
        tokens: Policy-facing action-token states `[B, T, Q, H]`.
        target / mask: Predictor-aligned depth deltas and their validity mask.

    The shuffled condition is a deterministic cross-sample roll.  The zero condition retains the
    head's learned biases but removes every sample-specific action-token value.
    """
    _validate_inputs(current, tokens, target, mask)
    batch, transitions, views = current.shape[:3]

    def predict(condition_tokens: torch.Tensor) -> torch.Tensor:
        condition = condition_tokens[:, :, None].expand(batch, transitions, views, *condition_tokens.shape[2:])
        return head(
            current.reshape(batch * transitions * views, 1, *current.shape[-2:]),
            condition.reshape(batch * transitions * views, *condition.shape[-2:]),
        ).reshape_as(target)

    valid = mask.to(dtype=target.dtype)
    errors = {}
    for name, condition_tokens in {
        "correct": tokens,
        "shuffled": tokens.roll(shifts=1, dims=0),
        "zero": torch.zeros_like(tokens),
    }.items():
        errors[f"{name}_absolute_error_sum"] = ((predict(condition_tokens) - target).abs() * valid).sum(
            dtype=torch.float64
        )
    errors["valid_count"] = valid.sum(dtype=torch.float64)
    return errors
