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
