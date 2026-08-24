"""CPU contracts for I4's frozen depth-head condition interventions."""

import torch
from torch import nn

from starVLA.probes.i4_depth_causality import condition_error_sums


class MeanConditionHead(nn.Module):
    def forward(self, current: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return condition.mean(dim=(1, 2), keepdim=True).reshape(-1, 1, 1, 1).expand_as(current)


def _inputs():
    current = torch.zeros(2, 1, 1, 1, 1, 1)
    tokens = torch.tensor([[[[1.0]]], [[[3.0]]]])
    target = tokens.reshape_as(current)
    return current, tokens, target, torch.ones_like(current, dtype=torch.bool)


def test_correct_tokens_beat_shuffled_and_zero_conditions():
    current, tokens, target, mask = _inputs()
    errors = condition_error_sums(MeanConditionHead(), current, tokens, target, mask)
    assert errors["correct_absolute_error_sum"] == 0
    assert errors["shuffled_absolute_error_sum"] == 4
    assert errors["zero_absolute_error_sum"] == 4
    assert errors["valid_count"] == 2


def test_masked_targets_do_not_contribute_to_any_condition_error():
    current, tokens, target, mask = _inputs()
    mask[1] = False
    errors = condition_error_sums(MeanConditionHead(), current, tokens, target, mask)
    assert errors["correct_absolute_error_sum"] == 0
    assert errors["shuffled_absolute_error_sum"] == 2
    assert errors["zero_absolute_error_sum"] == 1
    assert errors["valid_count"] == 1


def test_cross_sample_intervention_rejects_singleton_batches():
    current, tokens, target, mask = _inputs()
    try:
        condition_error_sums(MeanConditionHead(), current[:1], tokens[:1], target[:1], mask[:1])
    except ValueError as error:
        assert "at least two" in str(error)
        return
    raise AssertionError("singleton batches must not silently skip the shuffled intervention")
