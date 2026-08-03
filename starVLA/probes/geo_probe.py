# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""The I3 probe head and its two fitting routines.

The probe is deliberately the weakest readout that can express the task: a linear map from one token's
channel vector to that token's log-depth, shared across views, tokens and temporal blocks. That is the
`1x1 conv` of the pre-registered protocol. Being linear is the whole point -- a stronger head can
compensate for a worse representation, which is exactly the difference the probe is supposed to
measure.

Head capacity is identical across all four arms by construction: V-JEPA 2 and 2.1 both have
`hidden_size = 1024`, so the state head is 1025 parameters and the delta head 2049 parameters in every
arm. Nothing about the comparison depends on matching capacity by hand.

Per-view sharing is a firewall as much as a parameter saving. Views are fused on the channel axis, so a
head reading all `V * hidden` channels at once could predict the wrist camera's depth from the
agentview channels. Slicing each view out and applying the same weights makes that impossible.

Two fitting routines, both reported:

* `fit_sgd` is the pre-registered protocol: masked L1, Adam, an lr/epoch grid selected on val, three
  seeds, test numbers reported. The three seeds are what the >=5% gate threshold is compared against.
* `fit_ridge` solves the same linear head in closed form. It has no optimiser, no seed and no
  schedule, so it answers "was one arm's optimisation merely luckier" -- but it minimises squared error
  rather than the masked L1 the protocol declared, so it is reported as a secondary check and never
  substituted for the pre-registered number.

Masked L1 on log-depth matches the loss family of `docs/implementation-plan.md` section 6 minus its
gradient term: a probe should be a readout, not a shaped objective.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# Selection happens on val and on the primary metric, declared here so it is not chosen per arm.
SELECTION_METRIC = "abs_rel"
DEFAULT_LR_GRID = (3e-3, 1e-3, 3e-4)
# Widened once, uniformly, after the first full run: the original `(10, 25, 50)` and `(1e-4 .. 1e2)`
# grids had *every* arm selecting their upper edge, so the numbers would have been budget-limited
# rather than representation-limited. The wider grids select in their interior and reproduce the
# original arm ordering and effect size (see the I3 results table), so this is a protocol repair and
# not a post-hoc search: both runs are reported and the same grid is applied to every arm.
DEFAULT_EPOCHS = (50, 100, 200)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_RIDGE_GRID = (1e2, 1e4, 1e6, 1e8)


def tokens_to_views(features: torch.Tensor, grid: Tuple[int, int], num_views: int) -> torch.Tensor:
    """`[N, blocks * h * w, V * D] -> [N, blocks, V, h, w, D]`, undoing the adapter's fusion.

    The adapter concatenates view `v` into channels `[v * D, (v + 1) * D)` and lays tokens out
    block-major then row-major, so this is a pure reshape plus permute; it copies nothing that a later
    `contiguous()` would not.
    """
    num_rows, tokens, dim = features.shape
    height, width = grid
    if dim % num_views:
        raise ValueError(f"{dim} channels do not split into {num_views} views")
    if tokens % (height * width):
        raise ValueError(f"{tokens} tokens are not a whole number of {height}x{width} blocks")
    hidden = dim // num_views
    blocks = tokens // (height * width)
    unpacked = features.reshape(num_rows, blocks, height, width, num_views, hidden)
    return unpacked.permute(0, 1, 4, 2, 3, 5)


def delta_inputs(views: torch.Tensor) -> torch.Tensor:
    """Pair up adjacent temporal blocks: `[N, Tp, V, h, w, D] -> [N, Tp - 1, V, h, w, 2D]`.

    The delta head sees both endpoints of a transition, which is the least it needs to predict change
    without being handed the change itself.
    """
    return torch.cat([views[:, :-1], views[:, 1:]], dim=-1)


def to_target_layout(predictions: torch.Tensor) -> torch.Tensor:
    """`[N, T, V, h, w, 1] -> [N, T, V, 1, h, w]`, the axis order of `depth_targets`."""
    return predictions.permute(0, 1, 2, 5, 3, 4)


class LinearReadout(nn.Module):
    """One shared affine map from a token's channels to its scalar depth quantity."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        """Args: `[N, T, V, h, w, in_dim]`. Returns: `[N, T, V, 1, h, w]`."""
        return to_target_layout(self.linear(views))


def masked_l1(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 over the valid elements only; zero-weighted elements contribute no gradient."""
    weight = mask.to(predictions.dtype)
    count = weight.sum().clamp_min(1.0)
    return ((predictions - targets).abs() * weight).sum() / count


@dataclass(frozen=True)
class ProbeInputs:
    """Features and targets for one arm and one split, all resident on the same device.

    Attributes:
        features: `[N, blocks * h * w, V * D]`, the cached frozen teacher features. Kept in float16 as
            stored and cast per batch, so a whole arm fits in device memory.
        states / states_mask: `[N, Tp, V, 1, h, w]` log-depth targets and their validity.
        deltas / deltas_mask: `[N, Tp - 1, V, 1, h, w]` adjacent-delta targets.
        grid: token grid the features and targets share.
    """

    features: torch.Tensor
    states: torch.Tensor
    states_mask: torch.Tensor
    deltas: torch.Tensor
    deltas_mask: torch.Tensor
    grid: Tuple[int, int]
    num_views: int = 2

    def __post_init__(self) -> None:
        counts = {self.features.shape[0], self.states.shape[0], self.deltas.shape[0]}
        if len(counts) != 1:
            raise ValueError(f"features and targets disagree on row count: {counts}")

    @property
    def num_rows(self) -> int:
        return self.features.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.features.shape[-1] // self.num_views

    def views(self, rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Unpacked float32 features for `rows` (all rows when None)."""
        block = self.features if rows is None else self.features[rows]
        return tokens_to_views(block.to(torch.float32), self.grid, self.num_views)

    def target(self, kind: str, rows: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        values, mask = (self.states, self.states_mask) if kind == "state" else (self.deltas, self.deltas_mask)
        if rows is None:
            return values, mask
        return values[rows], mask[rows]


def head_inputs(views: torch.Tensor, kind: str) -> torch.Tensor:
    """Per-token head input for a probe kind: the block itself, or a pair of adjacent blocks."""
    if kind == "state":
        return views
    if kind == "delta":
        return delta_inputs(views)
    raise ValueError(f"unknown probe kind {kind!r}")


def head_input_dim(hidden_size: int, kind: str) -> int:
    return hidden_size if kind == "state" else 2 * hidden_size


@dataclass
class FitConfig:
    """The identical budget every arm is fitted under."""

    lr_grid: Sequence[float] = field(default_factory=lambda: list(DEFAULT_LR_GRID))
    epochs: Sequence[int] = field(default_factory=lambda: list(DEFAULT_EPOCHS))
    batch_rows: int = 32
    weight_decay: float = 0.0
    ridge_grid: Sequence[float] = field(default_factory=lambda: list(DEFAULT_RIDGE_GRID))

    @property
    def max_epochs(self) -> int:
        return max(self.epochs)


@torch.no_grad()
def predict(head: nn.Module, data: ProbeInputs, kind: str, batch_rows: int = 64) -> torch.Tensor:
    """Predictions for every row of `data`, evaluated in batches to bound peak memory."""
    head.eval()
    outputs: List[torch.Tensor] = []
    for start in range(0, data.num_rows, batch_rows):
        rows = torch.arange(start, min(start + batch_rows, data.num_rows), device=data.features.device)
        outputs.append(head(head_inputs(data.views(rows), kind)))
    return torch.cat(outputs, dim=0)


def fit_sgd(
    train: ProbeInputs,
    val: ProbeInputs,
    kind: str,
    seed: int,
    config: FitConfig,
    device: str = "cuda",
) -> Dict:
    """Fit the linear head under the pre-registered protocol for one seed.

    The lr/epoch grid is swept by training at each lr for `max_epochs` and snapshotting val error at
    every epoch in `config.epochs`, so the grid costs one run per lr rather than one per pair. Both
    the seed and the shuffling order are derived from `seed`, so a rerun reproduces the fit.

    Returns:
        The selected `(lr, epoch)`, the val loss achieved, and the selected head's `state_dict`.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best: Optional[Dict] = None

    for lr in config.lr_grid:
        torch.manual_seed(seed)
        head = LinearReadout(head_input_dim(train.hidden_size, kind)).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=config.weight_decay)

        for epoch in range(1, config.max_epochs + 1):
            head.train()
            order = torch.randperm(train.num_rows, generator=generator)
            for start in range(0, train.num_rows, config.batch_rows):
                rows = order[start : start + config.batch_rows].to(device)
                targets, mask = train.target(kind, rows)
                loss = masked_l1(head(head_inputs(train.views(rows), kind)), targets, mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            if epoch in config.epochs:
                targets, mask = val.target(kind)
                val_loss = float(masked_l1(predict(head, val, kind), targets, mask))
                if best is None or val_loss < best["val_loss"]:
                    best = {
                        "lr": lr,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "state_dict": {name: tensor.detach().clone() for name, tensor in head.state_dict().items()},
                    }

    if best is None:
        raise RuntimeError("empty lr/epoch grid")
    return best


def fit_ridge(
    train: ProbeInputs,
    val: ProbeInputs,
    kind: str,
    config: FitConfig,
    device: str = "cuda",
    batch_rows: int = 64,
) -> Dict:
    """Closed-form least-squares fit of the same head, with the ridge weight selected on val.

    Normal equations are accumulated in float64 over batches, so the 1024x1024 (or 2048x2048) system is
    solved once at full precision without ever materialising the design matrix. Secondary by design:
    this minimises squared error, not the protocol's masked L1.
    """
    in_dim = head_input_dim(train.hidden_size, kind)
    gram = torch.zeros(in_dim + 1, in_dim + 1, dtype=torch.float64, device=device)
    moment = torch.zeros(in_dim + 1, 1, dtype=torch.float64, device=device)

    for start in range(0, train.num_rows, batch_rows):
        rows = torch.arange(start, min(start + batch_rows, train.num_rows), device=device)
        inputs = head_inputs(train.views(rows), kind)
        targets, mask = train.target(kind, rows)
        # `[N, T, V, h, w, D] -> [rows, D]`, dropping invalid tokens instead of weighting them.
        flat_inputs = inputs.reshape(-1, in_dim).to(torch.float64)
        flat_targets = targets.permute(0, 1, 2, 4, 5, 3).reshape(-1, 1).to(torch.float64)
        keep = mask.permute(0, 1, 2, 4, 5, 3).reshape(-1)
        flat_inputs, flat_targets = flat_inputs[keep], flat_targets[keep]

        design = torch.cat([flat_inputs, torch.ones_like(flat_targets)], dim=1)
        gram += design.T @ design
        moment += design.T @ flat_targets

    identity = torch.eye(in_dim + 1, dtype=torch.float64, device=device)
    identity[-1, -1] = 0.0  # never penalise the bias: it carries the scene's mean depth
    best: Optional[Dict] = None
    for ridge in config.ridge_grid:
        solution = torch.linalg.solve(gram + ridge * identity, moment).squeeze(1)
        state_dict = {
            "linear.weight": solution[:-1].to(torch.float32).reshape(1, in_dim),
            "linear.bias": solution[-1:].to(torch.float32),
        }
        head = LinearReadout(in_dim).to(device)
        head.load_state_dict(state_dict)
        targets, mask = val.target(kind)
        val_loss = float(masked_l1(predict(head, val, kind), targets, mask))
        if best is None or val_loss < best["val_loss"]:
            best = {"ridge": ridge, "val_loss": val_loss, "state_dict": state_dict}

    if best is None:
        raise RuntimeError("empty ridge grid")
    return best


def constant_baselines(train: ProbeInputs, kind: str) -> Dict[str, torch.Tensor]:
    """Feature-free predictors fitted on `train`, as the floor the probes are read against.

    Medians rather than means, because the median is what minimises the protocol's masked L1:

    * `global_constant` -- one number for the entire split. The weakest predictor that exists.
    * `per_token_constant` -- one number per (view, row, column), i.e. the static scene layout, which
      needs no features whatsoever. The shared `LinearReadout` cannot express this: it has a single
      bias for all tokens. So a probe scoring worse than this baseline is not a contradiction, but the
      pair does bound how much of a probe's number comes from the representation rather than from the
      fixed camera geometry -- which is why both are reported next to the arms.

    Returns:
        Predictor name -> a tensor broadcastable onto `[N, T, V, 1, h, w]`.
    """
    values, mask = train.target(kind)
    masked = torch.where(mask, values.to(torch.float32), torch.full_like(values, float("nan"), dtype=torch.float32))
    per_token = torch.nanquantile(masked.flatten(0, 1).flatten(-2).permute(1, 2, 3, 0), 0.5, dim=-1)
    return {
        "global_constant": torch.nanquantile(masked.reshape(-1), 0.5).reshape(1, 1, 1, 1, 1, 1),
        # `[V, 1, h*w] -> [1, 1, V, 1, h, w]`: the axes the median collapsed come back as size 1.
        "per_token_constant": per_token.reshape(1, 1, *values.shape[2:]),
    }


def load_head(state_dict: Dict[str, torch.Tensor], in_dim: int, device: str = "cuda") -> LinearReadout:
    head = LinearReadout(in_dim).to(device)
    head.load_state_dict(state_dict)
    return head
