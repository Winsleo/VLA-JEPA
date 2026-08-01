"""Episode-boundary padding is configurable, and its default is the upstream one (I2 C5 / S6-a).

Upstream hard-coded `padding_strategy="zero"` at the `get_data_by_modality` call site with a
"HACK for realdata" note, so the run config could not see -- let alone change -- how state/action
windows that cross an episode boundary are filled. I2 lifts it to `datasets.vla_data.padding_strategy`
with the same default: switching it changes the training data, which is a separate experiment (D-040).

Pure logic, CPU-only: `retrieve_data_and_pad` does not touch `self`, so it runs unbound and no
dataset needs to exist on disk.
"""

import inspect

import numpy as np
import pytest

from starVLA.dataloader.gr00t_lerobot.datasets import (
    DEFAULT_PADDING_STRATEGY,
    PADDING_STRATEGIES,
    LeRobotSingleDataset,
)
from starVLA.dataloader.lerobot_datasets import get_vla_dataset, make_LeRobotSingleDataset

# A 5-step episode of 2-dimensional states, distinct per step so padding is visible.
EPISODE = np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0], [50.0, 51.0]])
MAX_LENGTH = len(EPISODE)


def _pad(step_indices, strategy):
    """Call the method unbound: it reads no instance state."""
    return LeRobotSingleDataset.retrieve_data_and_pad(
        None, array=EPISODE, step_indices=np.asarray(step_indices), max_length=MAX_LENGTH,
        padding_strategy=strategy,
    )


def test_the_default_is_the_value_upstream_hard_coded():
    """"zero", not the "first_last" that retrieve_data_and_pad itself defaults to."""
    assert DEFAULT_PADDING_STRATEGY == "zero"
    assert set(PADDING_STRATEGIES) == {"first_last", "zero"}


@pytest.mark.parametrize("function", [LeRobotSingleDataset.__init__, make_LeRobotSingleDataset])
def test_every_layer_defaults_to_the_upstream_strategy(function):
    """No caller may reintroduce a hard-coded strategy on the way down."""
    assert inspect.signature(function).parameters["padding_strategy"].default == DEFAULT_PADDING_STRATEGY


def test_the_strategy_comes_from_the_data_config():
    """`get_vla_dataset` needs a dataset on disk to run, so the wiring is pinned on its source."""
    source = "".join(inspect.getsource(get_vla_dataset).split())
    assert 'padding_strategy=data_cfg.get("padding_strategy",DEFAULT_PADDING_STRATEGY)' in source


def test_a_window_inside_the_episode_is_strategy_independent():
    """The default is only reachable at a boundary, so ordinary steps are untouched by this change."""
    inside = [0, 1, 2, 3, 4]
    assert np.array_equal(_pad(inside, "zero"), _pad(inside, "first_last"))
    assert np.array_equal(_pad(inside, "zero"), EPISODE)


def test_zero_padding_fills_both_boundaries_with_zeros():
    padded = _pad([-2, -1, 0, 1], "zero")
    assert np.array_equal(padded[:2], np.zeros((2, 2)))
    assert np.array_equal(padded[2:], EPISODE[:2])
    padded = _pad([3, 4, 5, 6], "zero")
    assert np.array_equal(padded[:2], EPISODE[3:])
    assert np.array_equal(padded[2:], np.zeros((2, 2)))


def test_first_last_padding_repeats_the_boundary_step():
    """The alternative path must be reachable, not just accepted (it is what a later experiment tests)."""
    padded = _pad([-2, -1, 0, 1], "first_last")
    assert np.array_equal(padded[:2], np.tile(EPISODE[0], (2, 1)))
    padded = _pad([3, 4, 5, 6], "first_last")
    assert np.array_equal(padded[2:], np.tile(EPISODE[-1], (2, 1)))


def test_an_unknown_strategy_is_rejected_at_construction(tmp_path):
    """Fail on the config, not thousands of steps later inside the loader."""
    with pytest.raises(ValueError, match="Invalid padding strategy"):
        LeRobotSingleDataset(tmp_path, {}, "new_embodiment", padding_strategy="repeat")


def test_an_unknown_strategy_is_still_rejected_at_the_padding_call():
    with pytest.raises(ValueError, match="Invalid padding strategy"):
        _pad([-1, 0], "repeat")
