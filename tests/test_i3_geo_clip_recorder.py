"""Recording contract for the I3 geometry-probe clips (S1a).

Two things are pinned here. First, that enabling depth is strictly additive: `_get_libero_env` keeps
`camera_depths=False` by default, so the I1/I2 LIBERO evaluation protocol is unchanged. Second, the
pure parts of the recorder - clip start selection, split assignment, the 180 degree rotation applied
to RGB and depth alike, and the sensor-range mask - which together decide whether the cached clips
are reproducible (I3 gate condition b) and whether invalid pixels stay countable (condition c).

Pure logic, CPU-only, no simulator: `geo_clip_contract.py` holds everything that does not need
`libero` / `robosuite`, which are only installed in the simulator environment.
Run:  pytest tests/test_i3_geo_clip_recorder.py -v
"""

import re
from pathlib import Path

import numpy as np
import pytest

from examples.LIBERO.geo_clip_contract import (
    CLIP_STRIDE,
    MAX_STEPS_BY_SUITE,
    TARGET_TYPE,
    VIEW_NAMES,
    clip_starts,
    decode_action,
    depth_valid_mask,
    rotate180,
    split_for_episode,
)
from examples.LIBERO.model2libero_interface import LIBERO_GRIPPER_CLOSE, LIBERO_GRIPPER_OPEN

EVAL_LIBERO = Path(__file__).resolve().parents[1] / "examples" / "LIBERO" / "eval_libero.py"

# LIBERO's own clipping planes for the LIBERO scenes, in metres; only the ordering matters here.
Z_NEAR, Z_FAR = 0.1, 10.0


@pytest.fixture(scope="module")
def eval_libero_source():
    """The evaluation entry point as text: parsed, never imported (it needs the LIBERO package)."""
    return EVAL_LIBERO.read_text()


def test_depth_rendering_is_off_by_default(eval_libero_source):
    """The recorder opts in; the evaluation protocol of I1/I2 must be untouched (AGENTS 10.5)."""
    signature = re.search(r"def _get_libero_env\(([^)]*)\)", eval_libero_source)
    assert signature is not None, "could not find _get_libero_env in eval_libero.py"
    assert "camera_depths=False" in signature.group(1).replace(" ", "")


def test_depth_flag_reaches_the_environment(eval_libero_source):
    """A default-only parameter would silently record nothing; it must be forwarded to env_args."""
    assert re.search(r'"camera_depths":\s*camera_depths', eval_libero_source) is not None


def test_step_budget_matches_the_evaluation_script(eval_libero_source):
    """Recorded episodes must be exactly as long as evaluated ones (the table is duplicated)."""
    upstream = {}
    for condition, value in re.findall(
        r"(?:if|elif) ([^\n:]*task_suite_name[^\n:]*):\s*\n\s*max_steps = (\d+)", eval_libero_source
    ):
        for suite in re.findall(r'"([a-z0-9_]+)"', condition):
            upstream[suite] = int(value)

    assert upstream, "could not parse the max_steps chain from eval_libero.py"
    for suite, max_steps in MAX_STEPS_BY_SUITE.items():
        assert upstream[suite] == max_steps, f"{suite}: {upstream[suite]} upstream vs {max_steps}"
    # libero_mix is deliberately not recordable: it needs a category_value and is not part of the
    # I1/I2 evaluation protocol. Any other new suite upstream should surface here.
    assert set(upstream) - set(MAX_STEPS_BY_SUITE) == {"libero_mix"}


def test_view_order_and_target_type_are_the_documented_ones():
    """View order feeds the teacher's feature-axis concatenation; the label gates target mixing."""
    assert VIEW_NAMES == ("agentview", "robot0_eye_in_hand")
    assert TARGET_TYPE == "sim_metric"
    assert CLIP_STRIDE == 1


class TestClipStarts:
    def test_starts_are_deterministic_and_ordered(self):
        first = clip_starts(100, 8, 8)
        assert first == clip_starts(100, 8, 8)
        assert first == sorted(set(first))

    def test_starts_span_the_whole_episode(self):
        starts = clip_starts(100, 8, 8)
        assert starts[0] == 0
        assert starts[-1] == 100 - 8
        assert len(starts) == 8

    def test_clips_stay_inside_the_episode(self):
        for num_frames in (8, 9, 23, 100, 301):
            for start in clip_starts(num_frames, 8, 8):
                assert 0 <= start and start + 8 <= num_frames

    def test_short_episodes_yield_no_clip(self):
        assert clip_starts(7, 8, 8) == []
        assert clip_starts(0, 8, 8) == []

    def test_a_single_full_clip_starts_at_zero(self):
        assert clip_starts(8, 8, 8) == [0]

    def test_overlapping_requests_are_deduplicated(self):
        """Fewer possible starts than requested clips must not produce duplicate clips."""
        starts = clip_starts(10, 8, 8)
        assert starts == [0, 1, 2]


class TestSplitAssignment:
    def test_default_split_is_three_one_one(self):
        splits = [split_for_episode(idx, 3, 1) for idx in range(5)]
        assert splits == ["train", "train", "train", "val", "test"]

    def test_boundaries_move_with_the_split_sizes(self):
        assert [split_for_episode(idx, 1, 1) for idx in range(4)] == [
            "train",
            "val",
            "test",
            "test",
        ]

    def test_an_empty_validation_split_is_allowed(self):
        assert [split_for_episode(idx, 2, 0) for idx in range(3)] == ["train", "train", "test"]


class TestRotation:
    def test_rotation_matches_the_evaluation_preprocessing(self):
        frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        assert np.array_equal(rotate180(frame), frame[::-1, ::-1])

    def test_rotation_is_its_own_inverse(self):
        frame = np.random.default_rng(0).integers(0, 255, size=(4, 5, 3), dtype=np.uint8)
        assert np.array_equal(rotate180(rotate180(frame)), frame)

    def test_rgb_and_depth_stay_aligned(self):
        """AGENTS section 7: one flip convention for both modalities, or the targets are garbage."""
        rgb = np.zeros((6, 8, 3), dtype=np.uint8)
        depth = np.zeros((6, 8), dtype=np.float32)
        rgb[1, 2] = 255
        depth[1, 2] = 1.0
        moved_rgb = np.argwhere(rotate180(rgb)[..., 0] == 255)
        moved_depth = np.argwhere(rotate180(depth) == 1.0)
        assert np.array_equal(moved_rgb, moved_depth)
        assert np.array_equal(moved_rgb[0], [6 - 1 - 1, 8 - 1 - 2])

    def test_output_is_contiguous(self):
        """np.savez on a negative-stride view would work, but downstream torch.from_numpy needs it."""
        assert rotate180(np.zeros((4, 4, 3), dtype=np.uint8)).flags["C_CONTIGUOUS"]


class TestDepthValidMask:
    def test_mid_range_depth_is_valid(self):
        depth = np.array([[1.0, 5.0]], dtype=np.float32)
        assert depth_valid_mask(depth, Z_NEAR, Z_FAR).all()

    def test_clipping_planes_are_invalid(self):
        depth = np.array([[Z_NEAR, Z_FAR]], dtype=np.float32)
        assert not depth_valid_mask(depth, Z_NEAR, Z_FAR).any()

    def test_non_finite_depth_is_invalid(self):
        depth = np.array([[np.nan, np.inf, -np.inf]], dtype=np.float32)
        assert not depth_valid_mask(depth, Z_NEAR, Z_FAR).any()

    def test_mask_shape_and_dtype_follow_the_depth_map(self):
        depth = np.full((8, 2, 4, 4), 3.0, dtype=np.float32)
        mask = depth_valid_mask(depth, Z_NEAR, Z_FAR)
        assert mask.shape == depth.shape
        assert mask.dtype == np.bool_


class TestActionDecoding:
    def _raw(self, gripper):
        return {
            "world_vector": [0.1, 0.2, 0.3],
            "rotation_delta": [0.4, 0.5, 0.6],
            "open_gripper": [gripper],
        }

    def test_decoded_action_is_the_seven_channel_libero_action(self):
        action = decode_action(self._raw(1.0))
        assert action.shape == (7,)
        assert np.allclose(action[:6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_gripper_uses_the_shared_libero_convention(self):
        """Same helper as eval_libero, so a sign flip cannot appear only in the recorder (D-040).

        The unnormalised gripper channel is 1.0 (open) or 0.5 (close); LIBERO inverts the sign.
        """
        assert decode_action(self._raw(1.0))[6] == LIBERO_GRIPPER_OPEN
        assert decode_action(self._raw(0.5))[6] == LIBERO_GRIPPER_CLOSE

    def test_malformed_response_raises(self):
        raw = self._raw(1.0)
        raw["world_vector"] = [0.1, 0.2]
        with pytest.raises(ValueError, match="Invalid action sizes"):
            decode_action(raw)
