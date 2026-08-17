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
    """

    def __init__(self, tokens: int, hidden: int, num_views: int, grid: Tuple[int, int]) -> None:
        super().__init__()
        self.tokens, self.hidden, self.num_views, self.grid = tokens, hidden, num_views, grid
        height, width = grid
        self.linear = nn.Linear(tokens * hidden, num_views * height * width)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        rows, transitions = features.shape[:2]
        flat = features.reshape(rows, transitions, self.tokens * self.hidden)
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

    Every arm is fitted under the identical grid so the comparison is of representations, not of
    tuning effort. Returns the selected point and the fitted head's state dict.
    """
    best: Dict[str, object] = {"val": float("inf")}
    for lr in lr_grid:
        torch.manual_seed(seed)
        head = TokenReadout(tokens, hidden, num_views, grid).to(device)
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
