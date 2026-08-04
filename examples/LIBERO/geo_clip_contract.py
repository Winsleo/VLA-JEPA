"""Pure contract for the I3 geometry-probe clip cache (no simulator dependency).

`record_geo_clips.py` needs `libero` and `robosuite`, which only exist in the simulator environment
(`envs/libero_py310`), while the unit tests run in the pinned training environment. Everything here
is therefore framework-free so `tests/test_i3_geo_clip_recorder.py` can pin the recording contract
without a simulator - the same split of responsibilities `libero_gripper_command` already uses.
"""

import pathlib
from typing import Dict, List, Sequence, Tuple

import numpy as np

from examples.LIBERO.model2libero_interface import libero_gripper_command

# View order matches `examples/LIBERO/modality.json` ("primary_image" then "wrist_image"), which is
# the order the dataloader stacks views in and therefore the order the teacher concatenates them in.
VIEW_NAMES: Tuple[str, ...] = ("agentview", "robot0_eye_in_hand")

TARGET_TYPE = "sim_metric"
RECORDER_VERSION = 1

# Training samples 8 consecutive observations: `starVLA/dataloader/lerobot_datasets.py` passes
# `observation_indices=list(range(video_horizon))` with `video_horizon = vj2_model.num_frames = 8`.
CLIP_STRIDE = 1

# The step budget of `eval_libero.py`, so recorded episodes are exactly as long as evaluated ones.
# Duplicated (not imported) to leave that rollout loop untouched; `libero_mix` is excluded because it
# needs a `category_value` and is not part of the I1/I2 evaluation protocol. Both the values and that
# exclusion are pinned against the upstream chain by `tests/test_i3_geo_clip_recorder.py`.
MAX_STEPS_BY_SUITE: Dict[str, int] = {
    "libero_spatial": 250,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# Sensor-range validity. Kept separate from the `[d_min, d_max]` clipping in `depth_targets.py` so
# that invalid, range-clipped and usable pixels stay separately countable (I3 gate condition c).
VALID_RULE = "isfinite(d) & (d > z_near * 1.001) & (d < z_far * 0.999)"
_NEAR_MARGIN = 1.001
_FAR_MARGIN = 0.999

# Intrinsics/extrinsics describe the raw renderer frame, while stored frames are rotated 180 degrees
# to match training preprocessing. Any projection must therefore map (u, v) -> (W-1-u, H-1-v).
PIXEL_CONVENTION = "frames rotated 180 deg after render; camera matrices refer to the raw frame"


def rotate180(frame: np.ndarray) -> np.ndarray:
    """Match the training preprocessing rotation applied in `eval_libero.py:154-158`.

    Applied identically to RGB and depth: robosuite renders both with the same vertical convention
    (`robot_env.py::_create_camera_sensors` flips the colour and depth buffers together), so
    rotating both keeps RGB and depth pixel-aligned as required by AGENTS.md section 7.
    """
    return np.ascontiguousarray(frame[::-1, ::-1])


def decode_action(raw_action: dict) -> np.ndarray:
    """Turn a policy-server response into a LIBERO delta action (as `eval_libero.py:200-216`)."""
    world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
    rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
    open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
    if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
        raise ValueError(
            f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
            f"rotation_delta={rotation_delta.shape}, open_gripper={open_gripper.shape}"
        )
    gripper = libero_gripper_command(open_gripper)
    return np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)


def split_for_episode(episode_idx: int, num_train: int, num_val: int) -> str:
    """Deterministic per-episode split assignment (no RNG, so re-recording keeps the split)."""
    if episode_idx < num_train:
        return "train"
    if episode_idx < num_train + num_val:
        return "val"
    return "test"


def clip_span(clip_frames: int, stride: int = CLIP_STRIDE) -> int:
    """Episode frames one clip reaches across: `clip_frames` samples `stride` apart.

    The clip still holds `clip_frames` frames whatever the stride; what grows is the real time it
    covers, which is the interval a depth transition target spans (`scripts/run_geo_probes.py`).
    """
    return (clip_frames - 1) * stride + 1


def clip_starts(num_frames: int, clip_frames: int, max_clips: int, stride: int = CLIP_STRIDE) -> List[int]:
    """Evenly spaced, deduplicated clip start indices covering the whole episode.

    Spreading the starts (rather than taking the first N) keeps the late reach/grasp phase in the
    cache. Deterministic by construction: no sampling.

    The bound is the clip's `clip_span`, not its frame count: a strided clip ends at
    `start + span - 1`, so bounding by `clip_frames` would run off the end of the episode. An episode
    too short to hold one strided clip yields no start rather than a truncated one.
    """
    span = clip_span(clip_frames, stride)
    if num_frames < span or max_clips < 1:
        return []
    last_start = num_frames - span
    raw = np.linspace(0, last_start, num=min(max_clips, last_start + 1))
    return sorted({round(float(value)) for value in raw})


def checked_strides(requested: Sequence[int]) -> List[int]:
    """Deduplicated, ascending recording strides; a stride below 1 would reverse or freeze time."""
    if not requested:
        raise ValueError("clip_strides is empty: at least one stride must be recorded")
    if any(stride < 1 for stride in requested):
        raise ValueError(f"clip_strides must all be >= 1, got {list(requested)}")
    return sorted(set(requested))


def stride_root(out_path: str, stride: int) -> pathlib.Path:
    """The dataset root holding the clips cut at one stride.

    Nested unconditionally, including for the default stride 1: `depth_cache_dataset.load_manifest`
    discovers clips by globbing `manifest_*.jsonl` at the root it is given, so one root per stride is
    what keeps clips covering different real durations from being read as one dataset. Applying it to
    every stride rather than only to the wide ones keeps one rule and makes a cache self-describing.
    """
    return pathlib.Path(out_path) / f"s{stride}"


def depth_valid_mask(depth_m: np.ndarray, z_near: float, z_far: float) -> np.ndarray:
    """Sensor-range mask: finite and strictly inside the clipping planes (`VALID_RULE`)."""
    with np.errstate(invalid="ignore"):
        return np.isfinite(depth_m) & (depth_m > z_near * _NEAR_MARGIN) & (depth_m < z_far * _FAR_MARGIN)
