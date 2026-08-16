import torch

from starVLA.model.modules.world_model.depth_delta_head import DepthDeltaHead
from starVLA.model.modules.world_model.depth_losses import depth_delta_loss


def test_head_shape_and_condition_sensitivity():
    torch.manual_seed(7)
    head = DepthDeltaHead(hidden_size=8, channels=8)
    current = torch.randn(6, 1, 16, 16)
    condition = torch.randn(6, 4, 8)
    first = head(current, condition)
    second = head(current, condition + 1.0)
    assert first.shape == (6, 1, 16, 16)
    assert not torch.allclose(first, second)


def test_depth_loss_reaches_every_head_parameter():
    """Pin the depth loss's gradient route into the head.

    The trainer cannot answer this: under DeepSpeed ZeRO-2 the engine frees `param.grad` after
    reduction, so `module_grad_norms` reports 0 tensors and a 0.0 norm for every module including
    `depth_delta_head`. That is the helper's documented "cannot read" signal, not evidence the
    gradient arrived, so the route is asserted here instead.
    """
    torch.manual_seed(11)
    head = DepthDeltaHead(hidden_size=8, channels=8)
    current = torch.randn(6, 1, 16, 16)
    condition = torch.randn(6, 4, 8)
    target = torch.randn(6, 1, 16, 16)
    mask = torch.ones_like(target, dtype=torch.bool)

    loss, _, _ = depth_delta_loss(head(current, condition), target, mask)
    loss.backward()

    missing = [name for name, param in head.named_parameters() if param.grad is None]
    assert not missing, f"no gradient reached {missing}"
    dead = [name for name, param in head.named_parameters() if not param.grad.any()]
    assert not dead, f"all-zero gradient at {dead}"
    assert all(torch.isfinite(param.grad).all() for _, param in head.named_parameters())


def test_a_fully_masked_depth_target_contributes_no_gradient():
    """An episode with no valid depth must not push the head, rather than pushing it toward zero."""
    torch.manual_seed(13)
    head = DepthDeltaHead(hidden_size=8, channels=8)
    current = torch.randn(2, 1, 16, 16)
    condition = torch.randn(2, 4, 8)
    target = torch.randn(2, 1, 16, 16)
    mask = torch.zeros_like(target, dtype=torch.bool)

    loss, raw, gradient = depth_delta_loss(head(current, condition), target, mask)
    loss.backward()

    assert torch.isfinite(loss) and loss.item() == 0.0
    assert raw.item() == 0.0 and gradient.item() == 0.0
    assert all(not param.grad.any() for _, param in head.named_parameters())


def test_masked_loss_blocks_invalid_pixel_gradient():
    prediction = torch.zeros(1, 1, 3, 3, requires_grad=True)
    target = torch.ones_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    mask[..., 0, 0] = False
    loss, raw, gradient = depth_delta_loss(prediction, target, mask)
    loss.backward()
    assert raw.item() == 1.0
    assert gradient.item() == 0.0
    assert prediction.grad[..., 0, 0].item() == 0.0
    assert torch.isfinite(loss)
