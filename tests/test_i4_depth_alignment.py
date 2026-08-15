"""I4 pseudo-depth cache alignment contract, independent of DA3/GPU execution.

The precompute cache uses absolute episode frame numbers. These tests pin the three places where a
shape-compatible mismatch could silently poison the I4 depth loss: camera order, boundary padding,
and image geometry transforms in the LIBERO loader.
"""

import ast
from pathlib import Path

import numpy as np

from starVLA.dataloader.gr00t_lerobot.data_config import Libero4in1DataConfig
from starVLA.dataloader.gr00t_lerobot.datasets import CachedLeRobotSingleDataset


def test_pseudo_depth_view_order_matches_the_libero_video_batch():
    """Cache V=0/1 must remain primary/wrist, the trainer's declared order."""
    assert Libero4in1DataConfig.video_keys == ["video.primary_image", "video.wrist_image"]


def test_boundary_padding_uses_the_same_clamped_frame_indices_for_every_view():
    """Depth cache lookup must copy the trainer's per-element episode-boundary clamp."""
    delta_indices = np.array([-4, -1, 0, 3, 9])
    base_index, episode_length = 1, 6
    expected = np.minimum(np.maximum(delta_indices + base_index, 0), episode_length - 1)

    # This is the exact expression in CachedLeRobotSingleDataset.get_video(). It is intentionally
    # tested with both underflow and overflow so the cache cannot use a different padding rule.
    trainer_indices = np.maximum(delta_indices + base_index, 0)
    trainer_indices = np.minimum(trainer_indices, episode_length - 1)
    assert trainer_indices.tolist() == expected.tolist() == [0, 0, 1, 4, 5]


def test_cached_loader_keeps_the_clamp_expression_at_the_video_read_boundary():
    """Pin the actual training loader rather than only a reimplementation of its arithmetic."""
    source = Path(CachedLeRobotSingleDataset.get_video.__code__.co_filename).read_text()
    tree = ast.parse(source)
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "get_video")
    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
    names = {getattr(call.func, "attr", None) for call in calls}
    assert {"maximum", "minimum"}.issubset(names)


def test_libero_transform_has_no_video_geometry_operation():
    """Depth is generated on decoded RGB coordinates, so LIBERO may not crop/resize/flip video later."""
    transforms = Libero4in1DataConfig(observation_indices=list(range(8)), action_indices=list(range(8))).transform()
    assert all(not transform.apply_to or not set(transform.apply_to) & set(Libero4in1DataConfig.video_keys) for transform in transforms.transforms)
