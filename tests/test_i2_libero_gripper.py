"""LIBERO gripper round trip, pinned end to end (I2 C5 / S6-b).

The gripper travels through three conventions between the policy head and the simulator: a
continuous value in [-1, 1], a binarised value that unnormalisation turns into 0.5 / 1.0, and
LIBERO's own channel where +1 closes and -1 opens. I2 replaced the magic numbers with names without
touching a value; these tests pin the chain so a later iteration cannot silently invert it.

Pure logic, CPU-only, no simulator: `libero_gripper_command` lives in model2libero_interface.py
precisely so this file does not need the LIBERO package.
"""

import numpy as np
import pytest

from examples.LIBERO.model2libero_interface import (
    GRIPPER_INDEX,
    GRIPPER_NORMALIZED_CLOSE,
    GRIPPER_NORMALIZED_OPEN,
    GRIPPER_NORMALIZED_THRESHOLD,
    LIBERO_GRIPPER_CLOSE,
    LIBERO_GRIPPER_OPEN,
    M1Inference,
    libero_gripper_command,
)

# LIBERO action stats: the gripper channel is recorded in [0, 1], the other six in [-1, 1].
ACTION_STATS = {
    "min": [-1.0] * 6 + [0.0],
    "max": [1.0] * 6 + [1.0],
}


def _unnormalize(gripper_value):
    actions = np.zeros((1, 7), dtype=np.float32)
    actions[0, GRIPPER_INDEX] = gripper_value
    return M1Inference.unnormalize_actions(actions, ACTION_STATS)


def test_the_values_are_the_upstream_ones():
    """Names only: I2 may not move a threshold or flip a sign (D-040)."""
    assert (GRIPPER_INDEX, GRIPPER_NORMALIZED_THRESHOLD) == (6, 0.5)
    assert (GRIPPER_NORMALIZED_CLOSE, GRIPPER_NORMALIZED_OPEN) == (0.0, 1.0)
    assert (LIBERO_GRIPPER_CLOSE, LIBERO_GRIPPER_OPEN) == (1.0, -1.0)


@pytest.mark.parametrize(("normalized", "expected"), [
    (-1.0, GRIPPER_NORMALIZED_CLOSE),
    (0.0, GRIPPER_NORMALIZED_CLOSE),
    (0.49, GRIPPER_NORMALIZED_CLOSE),
    (0.5, GRIPPER_NORMALIZED_OPEN),  # `<` threshold, so exactly 0.5 counts as open
    (1.0, GRIPPER_NORMALIZED_OPEN),
])
def test_the_policy_output_is_binarised_before_unnormalisation(normalized, expected):
    scaled = _unnormalize(normalized)[0, GRIPPER_INDEX]
    # min=0, max=1 => 0.5 * (x + 1) maps {0, 1} to {0.5, 1.0}
    assert scaled == 0.5 * (expected + 1.0)


@pytest.mark.parametrize(("unnormalized", "expected"), [
    (0.5, LIBERO_GRIPPER_CLOSE),  # `>` threshold, so exactly 0.5 counts as close
    (1.0, LIBERO_GRIPPER_OPEN),
])
def test_the_unnormalised_value_maps_to_the_libero_channel(unnormalized, expected):
    command = libero_gripper_command(unnormalized)
    assert command.dtype == np.float32
    assert command.shape == (1,)
    assert command[0] == expected


@pytest.mark.parametrize(("normalized", "expected"), [
    (0.0, LIBERO_GRIPPER_CLOSE),
    (1.0, LIBERO_GRIPPER_OPEN),
])
def test_the_full_round_trip_keeps_its_meaning(normalized, expected):
    """{0, 1} -> {0.5, 1.0} -> {close, open}: the chain the memo records, asserted in one place."""
    scaled = _unnormalize(normalized)[0, GRIPPER_INDEX]
    assert libero_gripper_command(scaled)[0] == expected


def test_the_other_six_channels_are_untouched_by_the_gripper_handling():
    actions = np.linspace(-1.0, 1.0, 7, dtype=np.float32)[None]
    unnormalized = M1Inference.unnormalize_actions(actions.copy(), ACTION_STATS)
    assert np.allclose(unnormalized[0, :GRIPPER_INDEX], actions[0, :GRIPPER_INDEX])


def test_a_list_or_array_input_is_accepted():
    """eval_libero.py passes a (1,) array; the SimplerEnv-style scalar path must work too."""
    assert libero_gripper_command(np.array([1.0], dtype=np.float32))[0] == LIBERO_GRIPPER_OPEN
    assert libero_gripper_command([0.5])[0] == LIBERO_GRIPPER_CLOSE
    assert libero_gripper_command(1.0)[0] == LIBERO_GRIPPER_OPEN
