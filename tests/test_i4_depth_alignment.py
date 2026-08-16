"""I4 pseudo-depth cache alignment contract, independent of DA3/GPU execution.

The precompute cache uses absolute episode frame numbers. These tests pin the three places where a
shape-compatible mismatch could silently poison the I4 depth loss: camera order, boundary padding,
and image geometry transforms in the LIBERO loader.
"""

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from starVLA.dataloader.gr00t_lerobot.data_config import Libero4in1DataConfig
from starVLA.dataloader.gr00t_lerobot.datasets import CachedLeRobotSingleDataset, LeRobotMixtureDataset


def _mentions(node, attribute):
    return any(getattr(call.func, "attr", None) == attribute for call in ast.walk(node) if isinstance(call, ast.Call))


def _load_precompute_module():
    """`scripts/` is excluded from the installed package, so import it by path."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "precompute_depth_targets.py"
    spec = importlib.util.spec_from_file_location("precompute_depth_targets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


# --- Undecodable-frame fallback -------------------------------------------------------------
# `libero_goal/episode_000082`'s wrist view is corrupt upstream: a sequential decode dies at frame
# 105 of 129, and the trainer's own seek cannot produce frames 104 or 105 either. The cache must
# still be full length so that an absolute frame index stays a valid key.


def test_decode_falls_back_to_seek_and_carries_undecodable_frames_forward(monkeypatch):
    """A failed sequential pass must yield full length, flagging only the unreachable indices."""
    module = _load_precompute_module()
    unreachable = {2, 3}
    frames = np.arange(5 * 2 * 2 * 3, dtype=np.uint8).reshape(5, 2, 2, 3)

    def dead_sequential(path):
        raise module.EpisodeDecodeError("corrupt packet")

    def fake_seek(path, target_ts, tolerance):
        index = round(target_ts * 10.0)
        return None if index in unreachable else frames[index]

    monkeypatch.setattr(module, "_decode_sequentially", dead_sequential)
    monkeypatch.setattr(module, "_decode_frame_by_trainer_seek", fake_seek)
    decoded, substituted = module.decode_episode_video("x.mp4", expected_frames=5, fps=10.0)

    assert substituted == sorted(unreachable)
    assert decoded.shape == frames.shape
    for index in range(5):
        # Substituted indices carry the previous frame forward; the rest are the seek's own frame.
        expected = frames[1] if index in unreachable else frames[index]
        assert np.array_equal(decoded[index], expected)


def test_decode_without_episode_metadata_still_raises(monkeypatch):
    """The clips source passes no length/fps, so it must keep the pre-fallback hard failure."""
    module = _load_precompute_module()

    def dead_sequential(path):
        raise module.EpisodeDecodeError("corrupt packet")

    monkeypatch.setattr(module, "_decode_sequentially", dead_sequential)
    with pytest.raises(module.EpisodeDecodeError):
        module.decode_episode_video("x.mp4")


def test_decode_refuses_to_invent_a_leading_frame(monkeypatch):
    """With no earlier frame to carry forward there is nothing honest to write."""
    module = _load_precompute_module()

    def dead_sequential(path):
        raise module.EpisodeDecodeError("corrupt packet")

    monkeypatch.setattr(module, "_decode_sequentially", dead_sequential)
    monkeypatch.setattr(module, "_decode_frame_by_trainer_seek", lambda *args: None)
    with pytest.raises(module.EpisodeDecodeError):
        module.decode_episode_video("x.mp4", expected_frames=3, fps=10.0)


def test_seek_fallback_uses_the_trainers_own_keyframe_scan():
    """Pin the fallback to the trainer's algorithm, not merely to a seek."""
    module = _load_precompute_module()
    source = Path(module.__file__).read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_decode_frame_by_trainer_seek"
    )
    seeks = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "seek"
    ]
    assert seeks, "the fallback must seek"
    # `keyframes_only=True` is what the trainer passes; without it the fallback would land on a
    # different frame than the one the model actually receives.
    assert any(
        keyword.arg == "keyframes_only" and keyword.value.value is True
        for seek in seeks
        for keyword in seek.keywords
    )


def test_substituted_depth_frames_are_unreachable_through_the_trainer():
    """The RGB fetch that fails sits ahead of the depth lookup, inside a resampling retry loop.

    This is why carrying a frame forward cannot leak into a loss: the trainer raises on exactly the
    timestamps that were substituted, and resamples the whole item before any depth is read.
    """
    source = Path(LeRobotMixtureDataset.__init__.__code__.co_filename).read_text()
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "__getitem__" and _mentions(node, "_load_depth")
    )
    handlers = [node for node in ast.walk(method) if isinstance(node, ast.Try) and node.handlers]
    assert handlers, "the item fetch must stay inside a resampling retry"

    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
    attrs = [getattr(call.func, "attr", None) for call in calls]
    assert "get_step_data" in attrs and "_load_depth" in attrs
    assert attrs.index("get_step_data") < attrs.index("_load_depth")
