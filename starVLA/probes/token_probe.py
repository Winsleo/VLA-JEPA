"""Linear readout of action-conditioned depth change from the policy-facing action tokens (I4).

This is deliberately *not* `geo_probe.LinearReadout`. That head maps one spatial token's channels to
that token's scalar depth, which only makes sense when the features carry a `h x w` grid. The tokens
probed here are `<|action_i|>` positions in the Qwen sequence: `[N, T, tokens, hidden]` with no
spatial axis at all. So the readout is one dense map per transition, from every action token of that
transition to the whole `V x h x w` depth-delta field.

Why these tokens and not the predictor states: `VLA_JEPA.forward` feeds exactly this tensor to
`depth_delta_head` as its `condition`, and `predict_action` reads the same Qwen last-hidden layer for
the policy. They are the one representation that the depth loss touches *and* the policy consumes, so
they are where "did geometry supervision change what the policy can see" is a well-posed question.

The probe reads action tokens only, never the current depth state that `depth_delta_head` also gets.
Handing it `current` would let arm C re-fit the head it was trained with, and a positive result would
say little; withholding it asks both arms the same harder question and keeps the comparison matched.
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn

from starVLA.probes.geo_probe import masked_l1


class TokenReadout(nn.Module):
    """One affine map per transition: all of a transition's action tokens -> its depth-delta field.

    Shapes:
        input  `[N, T, tokens, hidden]`
        output `[N, T, V, 1, h, w]`, the layout `depth_targets` uses.

    The map is shared across transitions, so the probe cannot memorise a per-transition constant
    that a feature-free baseline would also reach.

    Two deliberate departures from a stock linear head, both so that the number being reported is a
    property of the representation rather than of the optimiser's luck:

    - The weight starts at zero and the bias at the feature-free constant, so the probe *begins*
      exactly at the reference it is scored against and can only move away from it by using the
      features. Measured on real action tokens, a default `nn.Linear` over a 16384-wide fan-in emits
      values around 35 while the targets have a standard deviation of 0.077; a fixed epoch budget
      then mostly measures how far each arm crawled back from its initialisation.
    - Features are standardised by statistics fitted on the training split alone. A linear map could
      absorb the scale in principle, so this changes conditioning rather than expressivity.
    """

    def __init__(self, tokens: int, hidden: int, num_views: int, grid: Tuple[int, int]) -> None:
        super().__init__()
        self.tokens, self.hidden, self.num_views, self.grid = tokens, hidden, num_views, grid
        height, width = grid
        self.linear = nn.Linear(tokens * hidden, num_views * height * width)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.register_buffer("feature_mean", torch.zeros(tokens * hidden))
        self.register_buffer("feature_scale", torch.ones(tokens * hidden))

    def set_normalizer(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        """Adopt training-split feature statistics; `scale` is clamped so a dead channel is a no-op."""
        self.feature_mean.copy_(mean.reshape(-1))
        self.feature_scale.copy_(scale.reshape(-1).clamp_min(1e-6))

    def set_constant(self, constant: torch.Tensor) -> None:
        """Start the probe at the feature-free per-cell mean, averaged over transitions.

        The readout is shared across transitions, so the one bias it owns is the transition mean.
        """
        self.linear.bias.data.copy_(constant.mean(dim=0).reshape(-1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        rows, transitions = features.shape[:2]
        flat = features.reshape(rows, transitions, self.tokens * self.hidden)
        flat = (flat - self.feature_mean) / self.feature_scale
        height, width = self.grid
        out = self.linear(flat)  # [N, T, V * h * w]
        return out.reshape(rows, transitions, self.num_views, 1, height, width)


@dataclass(frozen=True)
class TokenProbeInputs:
    """One arm and one split, all resident on the same device.

    Attributes:
        features: `[N, T, tokens, hidden]` action-token states, stored float16 and cast per batch.
        deltas / deltas_mask: `[N, T, V, 1, h, w]` depth-delta targets and their validity.
    """

    features: torch.Tensor
    deltas: torch.Tensor
    deltas_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.features.shape[:2] != self.deltas.shape[:2]:
            raise ValueError(
                f"features {tuple(self.features.shape)} and targets {tuple(self.deltas.shape)} "
                "disagree on rows or transitions"
            )
        if self.deltas.shape != self.deltas_mask.shape:
            raise ValueError("target and mask shapes differ")

    @property
    def rows(self) -> int:
        return self.features.shape[0]


def constant_baseline(train: TokenProbeInputs) -> torch.Tensor:
    """The feature-free reference: the per-cell mean delta over the training split.

    A probe that does not beat this has extracted nothing from the tokens, so every improvement in
    the report is quoted relative to it rather than to zero.
    """
    weight = train.deltas_mask.to(train.deltas.dtype)
    total = (train.deltas * weight).sum(dim=0)
    count = weight.sum(dim=0).clamp_min(1.0)
    return total / count


@torch.no_grad()
def predict(head: nn.Module, data: TokenProbeInputs, batch_rows: int = 64) -> torch.Tensor:
    out = []
    for start in range(0, data.rows, batch_rows):
        block = data.features[start : start + batch_rows].to(torch.float32)
        out.append(head(block))
    return torch.cat(out) if out else torch.empty(0)


@torch.no_grad()
def evaluate(head: nn.Module, data: TokenProbeInputs, batch_rows: int = 64) -> float:
    return float(masked_l1(predict(head, data, batch_rows), data.deltas, data.deltas_mask))


def fit_ridge(
    train: TokenProbeInputs,
    val: TokenProbeInputs,
    *,
    tokens: int,
    hidden: int,
    num_views: int,
    grid: Tuple[int, int],
    penalties: Sequence[float] = (1e2, 1e3, 1e4, 1e5, 1e6, 1e7),
    device: str = "cuda",
) -> Dict[str, object]:
    """Closed-form ridge readout, penalty selected on val, never on test.

    Ridge rather than SGD because this probe is badly overdetermined: 8.4M weights against 2880
    training rows. Measured on real action tokens, an unregularised SGD fit selected on val still
    landed six times *worse* than the feature-free constant, which says the fit overfits from the
    first step rather than that the tokens carry no geometry.

    Solved in the dual, `W = X^T (X X^T + lambda I)^-1 Y`, because `n << p`: that inverts a
    2880x2880 matrix instead of a 16384x16384 one. The largest penalty in the grid drives the weight
    to zero, so the selected probe can never be worse than the constant it starts at.
    """
    flat_train = train.features.reshape(-1, tokens * hidden).to(device=device, dtype=torch.float32)
    mean, scale = flat_train.mean(dim=0), flat_train.std(dim=0).clamp_min(1e-6)
    constant = constant_baseline(train).to(device)
    centre = constant.mean(dim=0).reshape(-1)

    features = (flat_train - mean) / scale
    targets = train.deltas.reshape(features.shape[0], -1).to(device=device, dtype=torch.float32) - centre
    gram = features @ features.T
    eye = torch.eye(gram.shape[0], device=device, dtype=gram.dtype)

    best: Dict[str, object] = {"val": float("inf")}
    for penalty in penalties:
        dual = torch.linalg.solve(gram + penalty * eye, targets)
        weight = (features.T @ dual).T  # [out, p]
        head = TokenReadout(tokens, hidden, num_views, grid).to(device)
        head.set_normalizer(mean, scale)
        head.set_constant(constant)
        head.linear.weight.data.copy_(weight)
        score = evaluate(head, val)
        if score < best["val"]:
            best = {
                "val": score,
                "penalty": penalty,
                "state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            }
    return best


def fit(
    train: TokenProbeInputs,
    val: TokenProbeInputs,
    *,
    tokens: int,
    hidden: int,
    num_views: int,
    grid: Tuple[int, int],
    seed: int,
    lr_grid: Sequence[float] = (3e-3, 1e-3, 3e-4),
    epochs: Sequence[int] = (1, 2, 4, 8),
    batch_rows: int = 32,
    device: str = "cuda",
) -> Dict[str, object]:
    """Fit one probe under a fixed budget and select `(lr, epoch)` on val, never on test.

    Kept as the SGD reference next to `fit_ridge`, which is what the report uses: at 8.4M weights
    against 2880 rows this path overfits from the first step.

    Every arm is fitted under the identical grid so the comparison is of representations, not of
    tuning effort. Returns the selected point and the fitted head's state dict.
    """
    flat = train.features.reshape(-1, tokens * hidden).to(torch.float32)
    mean, scale = flat.mean(dim=0), flat.std(dim=0)
    constant = constant_baseline(train)

    best: Dict[str, object] = {"val": float("inf")}
    for lr in lr_grid:
        torch.manual_seed(seed)
        head = TokenReadout(tokens, hidden, num_views, grid).to(device)
        head.set_normalizer(mean.to(device), scale.to(device))
        head.set_constant(constant.to(device))
        optimiser = torch.optim.Adam(head.parameters(), lr=lr)
        for epoch in range(1, max(epochs) + 1):
            generator = torch.Generator(device="cpu").manual_seed(seed * 1000 + epoch)
            order = torch.randperm(train.rows, generator=generator)
            for start in range(0, train.rows, batch_rows):
                rows = order[start : start + batch_rows]
                loss = masked_l1(
                    head(train.features[rows].to(torch.float32)),
                    train.deltas[rows],
                    train.deltas_mask[rows],
                )
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
            if epoch in epochs:
                score = evaluate(head, val)
                if score < best["val"]:
                    best = {
                        "val": score,
                        "lr": lr,
                        "epoch": epoch,
                        "state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                    }
    return best
