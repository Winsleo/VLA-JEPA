"""The I3 probe firewall: a probe must read the teacher and change nothing about it.

Three separate claims, because they fail independently:

1. Static -- the probe package never imports the framework, the world predictor or the action model.
   A probe that constructed them would still produce numbers, just not numbers about a frozen teacher
   (`docs/implementation-plan.md` section 9).
2. Optimiser -- only probe-head parameters are ever handed to an optimiser.
3. Runtime -- after a probe's backward pass the real teacher has no gradients, is still
   `requires_grad=False`, and is still in `eval()`.

The derived arm is checked here too: arm C is pooled from arm B's cache rather than encoded, so the
claim "the derived cache equals what the adapter would have produced" needs pinning, not assuming.

Run:  CUDA_VISIBLE_DEVICES=0 pytest tests/test_i3_probe_firewall.py -v
"""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from parity_probe import CONFIG_PATH
from starVLA.probes import arms as arm_registry
from starVLA.probes import geo_probe, probe_cache

PROBE_CACHE = Path("/vepfs/wangshilong/data/dynaweave/i3_probe_cache")
VJEPA21_WEIGHTS = Path("/vepfs/wangshilong/models/dynaweave/vjepa21/port_apiantonio")

NUM_VIEWS = 2
# Module name fragments a probe has no business importing.
FORBIDDEN_IMPORTS = ("framework", "predictor", "action_model", "qwen", "trainer", "training")


def _clip(seeds, size, frames):
    views = [
        np.random.default_rng(seed).integers(0, 255, (frames, 3, size, size), dtype=np.uint8) for seed in seeds
    ]
    return np.stack(views)[None]


# --------------------------------------------------------------------------------------
# static: what the probe package is allowed to depend on
# --------------------------------------------------------------------------------------

def _imported_modules(path: Path):
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("module", ["__init__", "arms", "probe_cache", "geo_probe", "geo_metrics"])
def test_the_probe_package_imports_no_framework_or_predictor(module):
    path = Path(geo_probe.__file__).parent / f"{module}.py"
    offenders = [
        name for name in _imported_modules(path) if any(token in name.lower() for token in FORBIDDEN_IMPORTS)
    ]
    assert offenders == [], f"{module}.py imports {offenders}"


def test_the_entrypoint_imports_no_framework_or_predictor():
    """The script is orchestration, so it is the most likely place for a stray framework import."""
    path = Path(geo_probe.__file__).parents[2] / "scripts" / "run_geo_probes.py"
    offenders = [
        name for name in _imported_modules(path) if any(token in name.lower() for token in FORBIDDEN_IMPORTS)
    ]
    assert offenders == [], f"run_geo_probes.py imports {offenders}"


# --------------------------------------------------------------------------------------
# optimiser: only the head is trainable
# --------------------------------------------------------------------------------------

def test_the_head_is_the_smallest_readout_the_task_allows():
    """1025 parameters for a 1024-channel state head: one weight per channel plus a bias."""
    head = geo_probe.LinearReadout(1024)
    assert sum(parameter.numel() for parameter in head.parameters()) == 1025


def test_the_delta_head_reads_both_endpoints_and_nothing_more():
    assert geo_probe.head_input_dim(1024, "delta") == 2048
    assert sum(p.numel() for p in geo_probe.LinearReadout(2048).parameters()) == 2049


# --------------------------------------------------------------------------------------
# delta lag: the head's endpoints and its targets must span the same interval
# --------------------------------------------------------------------------------------

def _probe_inputs(blocks: int, delta_lag: int) -> geo_probe.ProbeInputs:
    """One row on a 2x2 grid, with delta targets consistent with `delta_lag`."""
    hidden, grid = 8, (2, 2)
    states = torch.zeros(1, blocks, NUM_VIEWS, 1, *grid)
    return geo_probe.ProbeInputs(
        features=torch.zeros(1, blocks * grid[0] * grid[1], NUM_VIEWS * hidden),
        states=states,
        states_mask=torch.ones_like(states, dtype=torch.bool),
        deltas=states[:, delta_lag:],
        deltas_mask=torch.ones_like(states[:, delta_lag:], dtype=torch.bool),
        grid=grid,
        num_views=NUM_VIEWS,
        delta_lag=delta_lag,
    )


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_the_delta_head_pairs_blocks_exactly_the_lag_apart(lag):
    """`lag` widens the interval the head sees; the channel width stays 2D, so capacity is unchanged."""
    views = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1, 1).expand(1, 4, NUM_VIEWS, 2, 2, 8)
    paired = geo_probe.delta_inputs(views.contiguous(), lag=lag)
    assert paired.shape == (1, 4 - lag, NUM_VIEWS, 2, 2, 16)
    assert torch.equal(paired[..., :8], views[:, : 4 - lag])
    assert torch.equal(paired[..., 8:], views[:, lag:])


def test_the_default_lag_is_the_adjacent_pairing():
    views = torch.randn(1, 4, NUM_VIEWS, 2, 2, 8)
    assert torch.equal(geo_probe.delta_inputs(views), geo_probe.delta_inputs(views, lag=1))
    assert torch.equal(geo_probe.head_inputs(views, "delta"), geo_probe.delta_inputs(views, lag=1))


def test_a_lag_wider_than_the_clip_is_refused():
    with pytest.raises(ValueError, match="at least 5 temporal blocks"):
        geo_probe.delta_inputs(torch.zeros(1, 4, NUM_VIEWS, 2, 2, 8), lag=4)


def test_probe_inputs_refuse_targets_that_disagree_with_the_lag():
    """The invariant that keeps a lag-2 target set from being fitted with adjacent-pair inputs."""
    with pytest.raises(ValueError, match="do not match"):
        blocks, grid, hidden = 4, (2, 2), 8
        states = torch.zeros(1, blocks, NUM_VIEWS, 1, *grid)
        geo_probe.ProbeInputs(
            features=torch.zeros(1, blocks * grid[0] * grid[1], NUM_VIEWS * hidden),
            states=states,
            states_mask=torch.ones_like(states, dtype=torch.bool),
            deltas=states[:, 1:],
            deltas_mask=torch.ones_like(states[:, 1:], dtype=torch.bool),
            grid=grid,
            num_views=NUM_VIEWS,
            delta_lag=2,
        )


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_the_head_input_matches_its_own_delta_targets(lag):
    """The single entry point: head rows and target rows agree by construction, at any lag."""
    data = _probe_inputs(blocks=4, delta_lag=lag)
    targets, _ = data.target("delta")
    assert data.head_input("delta").shape[:2] == targets.shape[:2]
    assert data.head_input("state").shape[1] == data.states.shape[1]


def test_an_optimizer_tracks_exactly_the_head_parameters():
    head = geo_probe.LinearReadout(8)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    tracked = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert tracked == {id(parameter) for parameter in head.parameters()}


def test_head_capacity_is_identical_across_arms():
    """Both teachers are 1024-dimensional, so no arm gets a bigger probe than another."""
    assert geo_probe.head_input_dim(1024, "state") == 1024


# --------------------------------------------------------------------------------------
# runtime: the real teacher after a probe backward pass
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    return OmegaConf.load(CONFIG_PATH)


@pytest.fixture(scope="module")
def arm_a_adapter(config):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    weights = Path(config.framework.vj2_model.base_encoder)
    if not weights.exists():
        pytest.skip(f"missing local weights: {weights}")
    return arm_registry.build_adapter(
        arm_registry.arm_by_name("A"),
        weights={arm_registry.TEACHER_VJEPA2: weights},
        num_frames=config.framework.vj2_model.num_frames,
    )


def test_the_arm_table_pins_the_pre_registered_geometry():
    table = {arm.name: (arm.teacher, arm.input_size, arm.pool_to, arm.derives_from) for arm in arm_registry.ARMS}
    assert table == {
        "A": (arm_registry.TEACHER_VJEPA2, 256, None, None),
        "B": (arm_registry.TEACHER_VJEPA21, 384, None, None),
        "C": (arm_registry.TEACHER_VJEPA21, 384, (16, 16), "B"),
        "D": (arm_registry.TEACHER_VJEPA21, 256, None, None),
    }
    assert arm_registry.PRIMARY_PAIR == ("A", "D")
    assert arm_registry.PRIMARY_METRIC == "abs_rel"


def test_teacher_features_carry_no_gradient(arm_a_adapter, config):
    clip = _clip((1, 2), arm_a_adapter.image_size, arm_a_adapter.num_frames)
    assert not arm_a_adapter.encode_video(clip).requires_grad


def test_a_probe_backward_leaves_the_teacher_untouched(arm_a_adapter):
    """The single most important test here: gradients must stop at the probe head."""
    clip = _clip((3, 4), arm_a_adapter.image_size, arm_a_adapter.num_frames)
    features = arm_a_adapter.encode_video(clip)
    views = geo_probe.tokens_to_views(features.to(torch.float32), arm_a_adapter.grid_size, NUM_VIEWS)

    head = geo_probe.LinearReadout(arm_a_adapter.hidden_size).to(features.device)
    predictions = head(views)
    targets = torch.zeros_like(predictions)
    mask = torch.ones_like(predictions, dtype=torch.bool)
    geo_probe.masked_l1(predictions, targets, mask).backward()

    assert all(parameter.grad is not None for parameter in head.parameters()), "the head did not train"
    assert all(parameter.grad is None for parameter in arm_a_adapter.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in arm_a_adapter.encoder.parameters())
    assert not arm_a_adapter.encoder.training


def test_the_teacher_stays_in_eval_across_a_probe_step(arm_a_adapter):
    """A probe never calls `train()` on the parent, but `enforce_frozen` must survive one anyway."""
    arm_a_adapter.encoder.train()
    arm_a_adapter.enforce_frozen()
    clip = _clip((5, 6), arm_a_adapter.image_size, arm_a_adapter.num_frames)
    arm_a_adapter.encode_video(clip)
    assert not arm_a_adapter.encoder.training


# --------------------------------------------------------------------------------------
# cache paths: incompatible target sets must not be able to occupy the same directory
# --------------------------------------------------------------------------------------

def test_every_target_variant_gets_its_own_directory():
    """Grid, estimator and lag all live in the path, so no two target sets can overwrite each other."""
    grid = (16, 16)
    paths = {
        probe_cache.targets_dir("/root", grid),
        probe_cache.targets_dir("/root", grid, "DA3METRIC-LARGE"),
        probe_cache.targets_dir("/root", grid, None, 2),
        probe_cache.targets_dir("/root", grid, "DA3METRIC-LARGE", 2),
        probe_cache.targets_dir("/root", (8, 8)),
    }
    assert len(paths) == 5


def test_the_default_lag_leaves_the_target_path_unchanged():
    """The pre-sweep caches on disk must stay readable without being rebuilt."""
    assert probe_cache.targets_dir("/root", (16, 16), None, 1) == probe_cache.targets_dir("/root", (16, 16))
    assert probe_cache.targets_dir("/root", (16, 16), "est", 1) == probe_cache.targets_dir("/root", (16, 16), "est")
    assert probe_cache.targets_dir("/root", (16, 16), None, 3).name == "16x16__lag3"


# --------------------------------------------------------------------------------------
# the derived arm: pooling a cache must equal encoding with the resampler
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cached_arms():
    directories = {name: probe_cache.features_dir(PROBE_CACHE, name) for name in ("B", "C")}
    for name, directory in directories.items():
        if not (directory / probe_cache.INDEX_FILE).exists():
            pytest.skip(f"no complete cache for arm {name} at {directory}")
    return {name: probe_cache.FeatureCache.open(directory) for name, directory in directories.items()}


def test_the_derived_cache_is_its_parent_pooled_bitwise(cached_arms):
    """Recomputing arm C from the stored arm B rows must reproduce the stored bytes exactly."""
    parent, derived = cached_arms["B"], cached_arms["C"]
    assert parent.paths == derived.paths, "the two caches disagree on row order"
    resampler = arm_registry.arm_by_name("C").resampler(parent.grid)

    rows = slice(0, 4)
    recomputed = probe_cache.to_feature_dtype(
        resampler(torch.from_numpy(np.array(parent.features[rows])).to(torch.float32)).numpy()
    )
    assert np.array_equal(recomputed, np.array(derived.features[rows]))


def test_the_derived_arm_publishes_the_judging_grid(cached_arms):
    assert cached_arms["C"].grid == (16, 16)
    assert cached_arms["B"].grid == (24, 24)
    assert cached_arms["C"].index["derived_from"] == "B"
    assert cached_arms["C"].index["tokens"] == 4 * 256


def test_the_derived_cache_matches_a_live_arm_c_forward(cached_arms):
    """Closes the loop: the on-disk derived arm is what the adapter with a resampler would produce.

    Tolerance rather than equality on purpose -- the cache is float16 and was pooled from float16
    parent rows, while the adapter pools its float32 output. The bitwise claim is the test above.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if not (VJEPA21_WEIGHTS / "model.safetensors").exists():
        pytest.skip(f"missing local weights: {VJEPA21_WEIGHTS}")

    derived = cached_arms["C"]
    clip_path = Path("/vepfs/wangshilong/data/dynaweave/i3_geo_clips") / derived.paths[0]
    if not clip_path.exists():
        pytest.skip(f"missing source clip {clip_path}")

    adapter = arm_registry.build_adapter(
        arm_registry.arm_by_name("C"),
        weights={arm_registry.TEACHER_VJEPA21: VJEPA21_WEIGHTS},
        num_frames=derived.index["num_frames"],
    )
    with np.load(clip_path) as clip:
        video = np.transpose(clip["rgb"], (1, 0, 4, 2, 3))[None]  # [T,V,H,W,C] -> [1,V,T,C,H,W]

    live = adapter.encode_video(video).to(torch.float32).cpu().numpy()
    stored = np.array(derived.features[0:1]).astype(np.float32)
    assert live.shape == stored.shape
    assert np.abs(live - stored).max() < 1e-2 * max(1.0, float(np.abs(live).max()))
