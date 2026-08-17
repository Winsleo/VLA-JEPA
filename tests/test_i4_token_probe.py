"""Contracts for the I4 action-token depth probe.

The probe answers one question -- do the tokens the policy reads encode action-conditioned depth
change -- so the things that must not drift are: what it is allowed to see, that it beats a
feature-free reference only by using the features, and that nothing future-dated leaks into it.

Pure logic, CPU-only.
"""

import torch

from starVLA.probes.token_probe import (
    TokenProbeInputs,
    TokenReadout,
    constant_baseline,
    evaluate,
    fit,
    predict,
)

TOKENS, HIDDEN, VIEWS, GRID = 8, 16, 2, (4, 4)


def _inputs(rows: int, transitions: int = 3, seed: int = 0) -> TokenProbeInputs:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(rows, transitions, TOKENS, HIDDEN, generator=generator)
    deltas = torch.randn(rows, transitions, VIEWS, 1, *GRID, generator=generator)
    return TokenProbeInputs(features=features, deltas=deltas, deltas_mask=torch.ones_like(deltas, dtype=torch.bool))


def test_readout_maps_action_tokens_onto_the_target_layout():
    """The probe output must be shaped like `depth_targets`, not like the token sequence."""
    head = TokenReadout(TOKENS, HIDDEN, VIEWS, GRID)
    out = head(torch.randn(5, 3, TOKENS, HIDDEN))
    assert out.shape == (5, 3, VIEWS, 1, *GRID)


def test_readout_is_shared_across_transitions():
    """One map for every transition, so a per-transition constant is not learnable as a shortcut."""
    head = TokenReadout(TOKENS, HIDDEN, VIEWS, GRID)
    features = torch.randn(1, 1, TOKENS, HIDDEN)
    repeated = features.expand(1, 3, TOKENS, HIDDEN)
    out = head(repeated)
    assert torch.allclose(out[:, 0], out[:, 1]) and torch.allclose(out[:, 0], out[:, 2])


def test_probe_sees_action_tokens_only():
    """No depth state reaches the readout: its parameters admit exactly the token fan-in.

    `depth_delta_head` also receives the current depth map. Handing that to the probe would let a
    depth-supervised arm re-fit its own head, so the contract is pinned here.
    """
    head = TokenReadout(TOKENS, HIDDEN, VIEWS, GRID)
    assert head.linear.in_features == TOKENS * HIDDEN
    assert head.linear.out_features == VIEWS * GRID[0] * GRID[1]
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    assert trainable == head.linear.weight.numel() + head.linear.bias.numel()


def test_mismatched_rows_or_transitions_are_rejected():
    generator = torch.Generator().manual_seed(3)
    features = torch.randn(4, 3, TOKENS, HIDDEN, generator=generator)
    deltas = torch.randn(4, 2, VIEWS, 1, *GRID, generator=generator)
    try:
        TokenProbeInputs(features=features, deltas=deltas, deltas_mask=torch.ones_like(deltas, dtype=torch.bool))
    except ValueError:
        return
    raise AssertionError("a transition-count mismatch must not be accepted silently")


def test_constant_baseline_is_the_masked_per_cell_mean():
    """The feature-free reference must ignore invalid cells rather than average zeros into them."""
    deltas = torch.zeros(2, 1, VIEWS, 1, *GRID)
    deltas[0] = 4.0
    deltas[1] = 8.0
    mask = torch.ones_like(deltas, dtype=torch.bool)
    mask[1] = False  # only the first row is valid anywhere
    data = TokenProbeInputs(features=torch.zeros(2, 1, TOKENS, HIDDEN), deltas=deltas, deltas_mask=mask)
    assert torch.allclose(constant_baseline(data), torch.full((1, VIEWS, 1, *GRID), 4.0))


def test_masked_cells_never_reach_the_gradient():
    """A fully masked target must leave the probe untouched rather than pull it toward zero."""
    data = _inputs(4)
    blind = TokenProbeInputs(
        features=data.features, deltas=data.deltas, deltas_mask=torch.zeros_like(data.deltas, dtype=torch.bool)
    )
    head = TokenReadout(TOKENS, HIDDEN, VIEWS, GRID)
    from starVLA.probes.geo_probe import masked_l1

    loss = masked_l1(head(blind.features), blind.deltas, blind.deltas_mask)
    loss.backward()
    assert all(not p.grad.any() for p in head.parameters())


def test_fit_selects_on_val_and_recovers_a_linear_signal():
    """A probe fitted on tokens that do determine the target must beat the feature-free constant."""
    generator = torch.Generator().manual_seed(11)
    truth = torch.randn(TOKENS * HIDDEN, VIEWS * GRID[0] * GRID[1], generator=generator)

    def make(rows: int, seed: int) -> TokenProbeInputs:
        gen = torch.Generator().manual_seed(seed)
        features = torch.randn(rows, 3, TOKENS, HIDDEN, generator=gen)
        flat = features.reshape(rows, 3, TOKENS * HIDDEN) @ truth
        deltas = flat.reshape(rows, 3, VIEWS, 1, *GRID)
        return TokenProbeInputs(features=features, deltas=deltas, deltas_mask=torch.ones_like(deltas, dtype=torch.bool))

    train, val = make(96, 1), make(32, 2)
    best = fit(train, val, tokens=TOKENS, hidden=HIDDEN, num_views=VIEWS, grid=GRID, seed=0,
               lr_grid=(3e-3,), epochs=(4,), device="cpu")
    assert best["lr"] == 3e-3 and best["epoch"] == 4

    head = TokenReadout(TOKENS, HIDDEN, VIEWS, GRID)
    head.load_state_dict(best["state_dict"])
    fitted = evaluate(head, val)

    constant = constant_baseline(train).unsqueeze(0).expand_as(val.deltas)
    from starVLA.probes.geo_probe import masked_l1

    reference = float(masked_l1(constant, val.deltas, val.deltas_mask))
    assert fitted < reference, f"probe {fitted} did not beat the feature-free {reference}"


def test_predict_is_batch_size_invariant():
    """Reporting must not depend on how the rows were chunked."""
    data = _inputs(10)
    head = TokenReadout(TOKENS, HIDDEN, VIEWS, GRID)
    assert torch.allclose(predict(head, data, batch_rows=3), predict(head, data, batch_rows=10), atol=1e-6)
