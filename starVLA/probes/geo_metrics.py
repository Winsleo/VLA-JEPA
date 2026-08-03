# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""Depth probe metrics, kept in the two classes `docs/implementation-plan.md` section 11 defines.

    metric class    AbsRel, RMSE, delta1, delta MAE, gradient, boundary
    relative class  aligned delta MAE, gradient, temporal rank consistency

The two classes are returned in separate dictionaries and there is deliberately no combined scalar:
metric depth is in metres and relative pseudo-depth is in MAD units, so any weighted sum of the two
would encode an arbitrary exchange rate between physically different quantities (AGENTS.md section 7,
gate condition c). `GeoMetrics` therefore holds both and offers no aggregation.

All predictions and targets are log-depth on a token grid, shaped `[N, Tp, V, 1, h, w]` for states and
`[N, Tp - 1, V, 1, h, w]` for deltas, matching `depth_targets.build_metric_delta_targets`. Every
metric is masked; a metric whose mask selects nothing is reported as NaN rather than silently
averaging over zero elements.

Two temporal quantities exist and are not the same thing:

* `delta_mae` scores the dedicated delta head against the target deltas -- can a probe predict change?
* `implied_delta_mae` / `temporal_sign_agreement` score the differences *implied* by the state head's
  own consecutive predictions -- are the states it produces temporally coherent at all?

Pure tensor math: no filesystem, no model, no logger.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

from starVLA.model.modules.world_model.depth_targets import (
    METRIC_TARGET_TYPES,
    TARGET_TYPES,
)

# delta1 is the standard 1.25 depth-accuracy threshold.
DELTA1_THRESHOLD = 1.25
# A log-depth step of 0.05 is a 5% depth change between neighbouring tokens: above this a boundary is
# called present. Stated as a constant because the boundary F1 is meaningless without it.
BOUNDARY_LOG_THRESHOLD = 0.05
# Temporal changes below this log magnitude carry no reliable sign, so they are excluded from the sign
# agreement instead of contributing coin flips.
TEMPORAL_DEADBAND = 0.01

# Which direction counts as better, so that "relative improvement" is never computed with the wrong
# sign. Any metric absent from both sets is a bug rather than a neutral quantity.
LOWER_IS_BETTER = frozenset(
    {
        "abs_rel",
        "rmse",
        "log_mae",
        "gradient",
        "implied_delta_mae",
        "delta_mae",
        "delta_gradient",
        "aligned_state_mae",
        "aligned_delta_mae",
    }
)
HIGHER_IS_BETTER = frozenset({"delta1", "boundary_f1", "temporal_sign_agreement", "temporal_rank_consistency"})


def relative_improvement(baseline: float, candidate: float, metric: str) -> float:
    """Signed improvement of `candidate` over `baseline`, positive when the candidate is better.

    Expressed as a fraction of the baseline so it can be read against the >=5% gate threshold.
    """
    if metric in LOWER_IS_BETTER:
        signed = baseline - candidate
    elif metric in HIGHER_IS_BETTER:
        signed = candidate - baseline
    else:
        raise KeyError(f"metric {metric!r} declares no better direction")
    return signed / abs(baseline) if baseline else float("nan")


def _selected(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """`values` where `mask`, zero elsewhere.

    `torch.where` rather than `values * mask`: an `inf` or `NaN` sitting at an *excluded* element
    survives the multiply as `NaN` and poisons every sum it reaches, which would make the mask
    decorative exactly where it matters most (invalid or missing depth).
    """
    return torch.where(mask, values, torch.zeros_like(values))


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean of `values` over `mask`, or NaN when the mask is empty."""
    count = mask.sum()
    if count == 0:
        return float("nan")
    return float(_selected(values, mask).sum() / count)


def _spatial_differences(values: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """First differences along the two grid axes, with masks requiring both endpoints valid."""
    d_width = values[..., :, 1:] - values[..., :, :-1]
    m_width = mask[..., :, 1:] & mask[..., :, :-1]
    d_height = values[..., 1:, :] - values[..., :-1, :]
    m_height = mask[..., 1:, :] & mask[..., :-1, :]
    return d_width, m_width, d_height, m_height


def gradient_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean absolute error of the spatial log-depth gradient, averaged over the two axes.

    Sensitive to structure rather than offset: a prediction that is uniformly too far away scores
    zero here, which is what separates "knows the layout" from "knows the distance".
    """
    pd_w, pm_w, pd_h, pm_h = _spatial_differences(pred, mask)
    td_w, _, td_h, _ = _spatial_differences(target, mask)
    return 0.5 * (_masked_mean((pd_w - td_w).abs(), pm_w) + _masked_mean((pd_h - td_h).abs(), pm_h))


def boundary_f1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """F1 between predicted and target depth discontinuities, thresholded on gradient magnitude.

    Both axes are pooled into one edge set. Returns NaN when the target has no boundary at all, since
    F1 is undefined there rather than perfect.
    """
    pd_w, pm_w, pd_h, pm_h = _spatial_differences(pred, mask)
    td_w, _, td_h, _ = _spatial_differences(target, mask)

    predicted = torch.cat([(pd_w.abs() > BOUNDARY_LOG_THRESHOLD)[pm_w], (pd_h.abs() > BOUNDARY_LOG_THRESHOLD)[pm_h]])
    actual = torch.cat([(td_w.abs() > BOUNDARY_LOG_THRESHOLD)[pm_w], (td_h.abs() > BOUNDARY_LOG_THRESHOLD)[pm_h]])

    true_positive = float((predicted & actual).sum())
    if actual.sum() == 0:
        return float("nan")
    precision_denominator = float(predicted.sum())
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / float(actual.sum())
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def temporal_sign_agreement(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Fraction of temporal changes whose direction the prediction gets right.

    This is the rank-consistency notion that is well defined at `Tp = 4`: a per-token Spearman
    correlation over four time steps is dominated by ties, whereas the sign of each adjacent change is
    exactly the "is this surface approaching or receding" question. Changes inside
    `TEMPORAL_DEADBAND` are dropped, so the score is not diluted by numerically arbitrary signs.
    """
    pred_change = pred[:, 1:] - pred[:, :-1]
    target_change = target[:, 1:] - target[:, :-1]
    valid = (mask[:, 1:] & mask[:, :-1]) & (target_change.abs() > TEMPORAL_DEADBAND)
    agree = torch.sign(pred_change) == torch.sign(target_change)
    return _masked_mean(agree.to(pred.dtype), valid)


def align_scale_shift(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    per_sample: bool = True,
) -> torch.Tensor:
    """Least-squares scale and shift of `pred` onto `target` over the masked elements.

    Relative depth is only defined up to an affine transform in log space, so a relative-target probe
    must be scored after alignment or it is scored on an arbitrary gauge. Alignment is per clip by
    default, matching the clip-level normalisation `depth_targets.normalize_clip_level` applies.
    """
    dims = tuple(range(1, pred.ndim)) if per_sample else tuple(range(pred.ndim))
    weight = mask.to(pred.dtype)
    count = weight.sum(dim=dims, keepdim=True).clamp_min(1.0)

    # Excluded elements are zeroed before any sum, for the reason `_selected` documents: the solved
    # scale and shift must not depend on what happens to sit at an invalid pixel.
    pred, target = _selected(pred, mask), _selected(target, mask)
    pred_mean = (pred * weight).sum(dim=dims, keepdim=True) / count
    target_mean = (target * weight).sum(dim=dims, keepdim=True) / count
    centred_pred = _selected(pred - pred_mean, mask)
    covariance = (centred_pred * (target - target_mean) * weight).sum(dim=dims, keepdim=True)
    variance = (centred_pred * centred_pred * weight).sum(dim=dims, keepdim=True)

    # A degenerate (constant) prediction has no scale to solve for; leave it at unit scale.
    scale = torch.where(variance > 0, covariance / variance.clamp_min(1e-12), torch.ones_like(variance))
    return scale * centred_pred + target_mean


@dataclass(frozen=True)
class GeoMetrics:
    """Both metric classes side by side, with no way to collapse them into one number.

    Attributes:
        metric: metric-depth numbers in metres / log metres. Empty for relative targets.
        relative: gauge-invariant numbers, computed after scale-shift alignment.
        target_type: which depth contract the targets came from.
        counts: element counts behind the averages, including how many were masked out.
    """

    metric: Dict[str, float]
    relative: Dict[str, float]
    target_type: str
    counts: Dict[str, int] = field(default_factory=dict)

    def as_rows(self) -> Dict[str, Dict[str, float]]:
        """Reporting view: the two classes stay separate columns of one table."""
        return {"metric": dict(self.metric), "relative": dict(self.relative)}


def metric_depth_metrics(
    pred_states: torch.Tensor,
    target_states: torch.Tensor,
    states_mask: torch.Tensor,
    pred_deltas: Optional[torch.Tensor] = None,
    target_deltas: Optional[torch.Tensor] = None,
    deltas_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Metric-class numbers from log-depth predictions. AbsRel / RMSE / delta1 are in metres.

    Args:
        pred_states / target_states: `[N, Tp, V, 1, h, w]` log metres.
        pred_deltas / target_deltas: optional `[N, Tp - 1, ...]` log-depth deltas from the delta head.
    """
    pred_metres = pred_states.exp()
    target_metres = target_states.exp()
    mask = states_mask

    ratio = torch.maximum(pred_metres / target_metres.clamp_min(1e-6), target_metres / pred_metres.clamp_min(1e-6))
    results = {
        "abs_rel": _masked_mean((pred_metres - target_metres).abs() / target_metres.clamp_min(1e-6), mask),
        "rmse": _masked_mean((pred_metres - target_metres) ** 2, mask) ** 0.5,
        "delta1": _masked_mean((ratio < DELTA1_THRESHOLD).to(pred_states.dtype), mask),
        "log_mae": _masked_mean((pred_states - target_states).abs(), mask),
        "gradient": gradient_error(pred_states, target_states, mask),
        "boundary_f1": boundary_f1(pred_states, target_states, mask),
        "implied_delta_mae": _masked_mean(
            ((pred_states[:, 1:] - pred_states[:, :-1]) - (target_states[:, 1:] - target_states[:, :-1])).abs(),
            mask[:, 1:] & mask[:, :-1],
        ),
        "temporal_sign_agreement": temporal_sign_agreement(pred_states, target_states, mask),
    }

    if pred_deltas is not None:
        if target_deltas is None or deltas_mask is None:
            raise ValueError("pred_deltas given without target_deltas / deltas_mask")
        results["delta_mae"] = _masked_mean((pred_deltas - target_deltas).abs(), deltas_mask)
        results["delta_gradient"] = gradient_error(pred_deltas, target_deltas, deltas_mask)

    return results


def relative_depth_metrics(
    pred_states: torch.Tensor,
    target_states: torch.Tensor,
    states_mask: torch.Tensor,
    pred_deltas: Optional[torch.Tensor] = None,
    target_deltas: Optional[torch.Tensor] = None,
    deltas_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Relative-class numbers: everything is computed after per-clip scale-shift alignment."""
    aligned_states = align_scale_shift(pred_states, target_states, states_mask)
    results = {
        "aligned_state_mae": _masked_mean((aligned_states - target_states).abs(), states_mask),
        "gradient": gradient_error(aligned_states, target_states, states_mask),
        "temporal_rank_consistency": temporal_sign_agreement(aligned_states, target_states, states_mask),
    }

    if pred_deltas is not None:
        if target_deltas is None or deltas_mask is None:
            raise ValueError("pred_deltas given without target_deltas / deltas_mask")
        aligned_deltas = align_scale_shift(pred_deltas, target_deltas, deltas_mask)
        results["aligned_delta_mae"] = _masked_mean((aligned_deltas - target_deltas).abs(), deltas_mask)
        results["delta_gradient"] = gradient_error(aligned_deltas, target_deltas, deltas_mask)

    return results


def evaluate(
    pred_states: torch.Tensor,
    target_states: torch.Tensor,
    states_mask: torch.Tensor,
    target_type: str,
    pred_deltas: Optional[torch.Tensor] = None,
    target_deltas: Optional[torch.Tensor] = None,
    deltas_mask: Optional[torch.Tensor] = None,
) -> GeoMetrics:
    """Both metric classes for one prediction set, with the metric class gated on the target contract.

    A pseudo-relative target is normalised to MAD units, where AbsRel and RMSE in "metres" would be
    numbers without a referent, so the metric class is left empty for it rather than computed and
    labelled misleadingly. A *metric* estimator's target is in log metres and therefore does get the
    metric class -- gated on `METRIC_TARGET_TYPES`, i.e. on the units, never on the string itself.
    """
    if target_type not in TARGET_TYPES:
        raise ValueError(f"unknown target_type {target_type!r}")

    shared = {
        "pred_deltas": pred_deltas,
        "target_deltas": target_deltas,
        "deltas_mask": deltas_mask,
    }
    metric = (
        metric_depth_metrics(pred_states, target_states, states_mask, **shared)
        if target_type in METRIC_TARGET_TYPES
        else {}
    )
    relative = relative_depth_metrics(pred_states, target_states, states_mask, **shared)

    counts = {
        "states_total": int(states_mask.numel()),
        "states_valid": int(states_mask.sum()),
        "states_invalid": int((~states_mask).sum()),
    }
    if deltas_mask is not None:
        counts.update(
            deltas_total=int(deltas_mask.numel()),
            deltas_valid=int(deltas_mask.sum()),
            deltas_invalid=int((~deltas_mask).sum()),
        )

    return GeoMetrics(metric=metric, relative=relative, target_type=target_type, counts=counts)
