# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""Entrypoint for the I3 geometry probes: extract frozen teacher features, then fit probes on them.

Orchestration only. The arm table lives in `starVLA/probes/arms.py`, the cache layout in
`probe_cache.py`, the target maths in `world_model/depth_targets.py`; this file wires them to the
filesystem and the command line.

Four stages, run in order:

    # once per grid: pool the recorded simulator metric depth onto the token grids
    python scripts/run_geo_probes.py targets --clips <clip cache> --out <probe cache>

    # once per non-derived arm: three forwards cover all four arms
    python scripts/run_geo_probes.py cache --clips <clip cache> --out <probe cache> --arms A B D

    # arm C, pooled from arm B's cache rather than forwarded again
    python scripts/run_geo_probes.py derive --arms C

    # probes, three seeds per arm, plus a closed-form ridge check and the feature-free baselines
    python scripts/run_geo_probes.py fit --arms A C D --kinds state delta

    # one curve out of many fit reports: delta skill over the floor against the target interval
    python scripts/run_geo_probes.py sweep --reports <probe cache>/reports --figure <docs figure path>

The same `targets` and `fit` stages take `--estimator <name>` to swap the simulator depth for a
precomputed pseudo-depth cache (`scripts/precompute_depth_targets.py`). That answers the S4 question
"how much probe signal survives if the target comes from an estimator rather than the simulator": the
arms, features, seeds and grid are identical, so the difference is the target source alone. Pseudo
targets live in their own cache directory, never overwriting the ground-truth ones.

The fit stage reports two floors next to the arms: the seed spread, which any claimed improvement has
to clear, and the constant predictors of `geo_probe.constant_baselines`, which say how much of an
arm's absolute number is the representation rather than a fixed camera looking at a fixed table. The
second floor is also reported as a skill score, i.e. each arm's improvement over it, since that is the
number "did the features help" actually asks for. Delta metrics come in two readings, the whole token
grid and the subset that moved, because averaged over the whole grid a delta is dominated by tokens
that did not change and on which the constant floor is exactly right.

The `sweep` stage reads *across* reports instead of within one. A delta target spans
`tubelet_size * delta_lag * clip_stride` frames, which at the LIBERO control rate is a duration, and
that duration is the x axis of the curve: how much of a transition a frozen representation can predict
depends on how much the scene moved in between. Each point keeps its own floor and its own seed
spread, and the whole curve is emitted rather than only its best point.

Nothing here constructs the Qwen backbone, the action model or the world predictor: the probe path
instantiates a `VJBackboneAdapter` and (in the fit stage) a probe head, and nothing else
(`docs/implementation-plan.md` section 9, pinned by `tests/test_i3_probe_firewall.py`).
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.depth_cache_dataset import DepthClipCacheDataset  # noqa: E402
from starVLA.model.modules.world_model.depth_targets import (  # noqa: E402
    DEFAULT_D_MAX,
    DEFAULT_D_MIN,
    TARGET_TYPE_METRIC,
    TARGET_TYPE_PSEUDO_METRIC,
    build_metric_delta_targets,
)
from starVLA.probes import arms as arm_registry  # noqa: E402
from starVLA.probes import geo_metrics, geo_probe, probe_cache  # noqa: E402

DEFAULT_CLIPS = Path("/vepfs/wangshilong/data/dynaweave/i3_geo_clips")
DEFAULT_OUT = Path("/vepfs/wangshilong/data/dynaweave/i3_probe_cache")
DEFAULT_CONFIG = REPO_ROOT / "configs" / "i1_libero_local.yaml"
DEFAULT_VJEPA21 = Path("/vepfs/wangshilong/models/dynaweave/vjepa21/port_apiantonio")
DEFAULT_PSEUDO = Path("/vepfs/wangshilong/data/dynaweave/i3_pseudo_depth")

# The one judging grid. Arm B's native 24x24 is absent on purpose: depth targets are pooled with
# exact non-overlapping windows and 24 does not divide the recorded 256x256 depth map, so a 24x24
# target grid does not exist (`starVLA/probes/arms.py`, user decision 2026-08-03).
DEFAULT_GRIDS = ((16, 16),)
TUBELET_SIZE = 2
# Clip caches recorded before the delta-interval sweep carry no `clip_stride` field; the recorder's
# only behaviour then was consecutive frames, i.e. stride 1 (`examples/LIBERO/geo_clip_contract.py`).
DEFAULT_CLIP_STRIDE = 1
# LIBERO's control rate, the only place frame counts become durations. Fixed by
# `libero/libero/envs/env_wrapper.py`'s `control_freq = 20`, which the recorder does not override, so
# one recorded frame is one control step is 0.05 s.
CONTROL_HZ = 20.0

# Metric classes as they appear in the report, in render order. The moving classes are the same metrics
# over `geo_metrics.moving_mask`, i.e. the elements that actually changed, and exist for deltas only.
METRIC_CLASSES = ("metric", "relative")
MOVING_CLASSES = ("metric_moving", "relative_moving")
# Splits the probes are scored on. Test keeps the unsuffixed report keys, so readers written before the
# interval sweep stay valid; val is added because any post-hoc choice has to be made off the test set.
REPORTED_SPLITS = ("test", "val")
VAL_SUFFIX = "_val"
# The floor a skill score is read against: the static scene layout, which is the stronger of the two
# feature-free predictors and therefore the one a probe has to beat to have used its features at all.
FLOOR_BASELINE = "per_token_constant"


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _provenance(clips: Path) -> Dict:
    return {
        "vla_jepa_commit": _git_commit(REPO_ROOT),
        "clip_cache": str(clips),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _dataset(clips: Path) -> DepthClipCacheDataset:
    """Every recorded clip in manifest order; splits are selected later, at fit time."""
    return DepthClipCacheDataset(root=clips, split=None)


def _loader(dataset: DepthClipCacheDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


def _clip_stride(dataset: DepthClipCacheDataset) -> int:
    """The frame stride every clip in the cache was cut at, refusing a cache that mixes strides.

    Mixing strides inside one root would put clips spanning different real durations into the same
    target set, which is exactly the confound the interval sweep is built to separate.
    """
    strides = {int(row.get("clip_stride", DEFAULT_CLIP_STRIDE)) for row in dataset.rows}
    if len(strides) != 1:
        raise SystemExit(f"clip cache {dataset.root} mixes recording strides {sorted(strides)}")
    return strides.pop()


def _row_index(dataset: DepthClipCacheDataset) -> Dict:
    """Row order and per-row labels, identical for every cache written from this dataset."""
    return {
        "num_rows": len(dataset.rows),
        "paths": [row["path"] for row in dataset.rows],
        "splits": [row["split"] for row in dataset.rows],
        "suites": [row["suite"] for row in dataset.rows],
    }


# --------------------------------------------------------------------------------------
# stage: depth targets
# --------------------------------------------------------------------------------------

def _estimator_provenance(pseudo_root: Path, estimator: str, num_clips: int) -> Dict:
    """The pseudo cache's own index, minus its path list, refusing an unfinished cache.

    Probe numbers are only interpretable next to which estimator produced the targets and whether it
    reported metres itself, so those fields travel into the target index rather than being looked up
    by hand later.
    """
    index = json.loads((pseudo_root / estimator / "estimator.json").read_text())
    if not index.get("complete", True) or index.get("num_present", 0) < num_clips:
        raise SystemExit(
            f"pseudo cache {pseudo_root / estimator} holds {index.get('num_present')} of {num_clips} clips"
        )
    return {key: value for key, value in index.items() if key != "paths"}


def _pseudo_depth(pseudo_root: Path, estimator: str, paths: Sequence[str]) -> torch.Tensor:
    """Precomputed estimator depth for one batch, in the dataset's `[B, V, T, 1, H, W]` layout.

    The pseudo cache mirrors the clip cache path for path, so a row is located by its own relative
    path rather than by position -- a missing file is an error, never a silently shorter batch.
    """
    clips = []
    for path in paths:
        with np.load(pseudo_root / estimator / path) as clip:
            depth = torch.from_numpy(clip["depth_m"].astype(np.float32))  # [T, V, H, W]
        clips.append(depth.permute(1, 0, 2, 3).unsqueeze(2))
    return torch.stack(clips)


def run_targets(args: argparse.Namespace) -> None:
    dataset = _dataset(args.clips)
    grids = [tuple(grid) for grid in args.grids]
    estimator = getattr(args, "estimator", None)
    source = "simulator metric depth" if estimator is None else f"pseudo depth from {estimator}"
    print(f"targets: {len(dataset)} clips -> grids {grids} ({source}, delta lag {args.delta_lag})", flush=True)

    buffers: Dict[Tuple[int, int], Dict[str, List[np.ndarray]]] = {
        grid: {"states": [], "states_mask": [], "deltas": [], "deltas_mask": []} for grid in grids
    }
    target_type = None

    for batch in _loader(dataset, args.batch_size, args.num_workers):
        # `[B, V, T, 1, H, W]` as `build_metric_delta_targets` expects.
        depth, valid = batch["depth"], batch["valid"]
        if estimator is not None:
            # The estimator's own finiteness is the only validity it has: reusing the simulator's
            # sensor mask here would hand the pseudo path information it would not have on a dataset
            # without depth, which is exactly the situation this run is measuring.
            depth, valid = _pseudo_depth(args.pseudo_root, estimator, batch["path"]), None
        for grid in grids:
            states, deltas = build_metric_delta_targets(
                cache_depth=depth,
                valid=valid,
                tubelet_size=TUBELET_SIZE,
                grid=grid,
                d_min=args.d_min,
                d_max=args.d_max,
                target_type=TARGET_TYPE_METRIC if estimator is None else TARGET_TYPE_PSEUDO_METRIC,
                delta_lag=args.delta_lag,
            )
            target_type = states.target_type
            store = buffers[grid]
            store["states"].append(states.values.numpy().astype(np.float32))
            store["states_mask"].append(states.mask.numpy())
            store["deltas"].append(deltas.values.numpy().astype(np.float32))
            store["deltas_mask"].append(deltas.mask.numpy())

    index_rows = _row_index(dataset)
    clip_stride = _clip_stride(dataset)
    for grid in grids:
        stacked = {name: np.concatenate(chunks, axis=0) for name, chunks in buffers[grid].items()}
        directory = probe_cache.targets_dir(args.out, grid, estimator, args.delta_lag)
        index = {
            **index_rows,
            "grid": list(grid),
            "target_type": target_type,
            "units": "log_meter",
            "tubelet_size": TUBELET_SIZE,
            "delta_lag": args.delta_lag,
            "clip_stride": clip_stride,
            # The interval the delta target actually spans, so a reader never has to recompute it
            # from the recording stride and the tubelet size by hand.
            "delta_frames": TUBELET_SIZE * args.delta_lag * clip_stride,
            "d_min": args.d_min,
            "d_max": args.d_max,
            "states_shape": list(stacked["states"].shape),
            "deltas_shape": list(stacked["deltas"].shape),
            "states_valid_fraction": float(stacked["states_mask"].mean()),
            "deltas_valid_fraction": float(stacked["deltas_mask"].mean()),
            "estimator": estimator,
            "estimator_index": (
                None if estimator is None else _estimator_provenance(args.pseudo_root, estimator, len(dataset))
            ),
            "provenance": _provenance(args.clips),
        }
        probe_cache.write_targets(directory, index=index, **stacked)
        print(
            f"targets {grid}: states {stacked['states'].shape} "
            f"valid {index['states_valid_fraction']:.4f} -> {directory}",
            flush=True,
        )


# --------------------------------------------------------------------------------------
# stage: frozen teacher features
# --------------------------------------------------------------------------------------

def _weights(args: argparse.Namespace) -> Dict[str, Path]:
    """Teacher weight roots. The V-JEPA 2 path comes from the pinned config, not a second copy."""
    config = OmegaConf.load(args.config)
    return {
        arm_registry.TEACHER_VJEPA2: Path(config.framework.vj2_model.base_encoder),
        arm_registry.TEACHER_VJEPA21: args.vjepa21,
    }


def run_cache(args: argparse.Namespace) -> None:
    selected = [arm_registry.arm_by_name(name) for name in args.arms]
    derived = [arm for arm in selected if arm.is_derived]
    if derived:
        raise SystemExit(
            f"arms {[arm.name for arm in derived]} are pooled from another arm's cache and need no "
            "forward pass; extract their parent instead"
        )

    dataset = _dataset(args.clips)
    index_rows = _row_index(dataset)
    weights = _weights(args)
    num_frames = int(OmegaConf.load(args.config).framework.vj2_model.num_frames)

    for arm in selected:
        directory = probe_cache.features_dir(args.out, arm.name)
        if (directory / probe_cache.INDEX_FILE).exists() and not args.overwrite:
            print(f"arm {arm.name}: complete cache already at {directory}, skipping", flush=True)
            continue

        adapter = arm_registry.build_adapter(arm, weights=weights, num_frames=num_frames, device=args.device)
        tokens = adapter.num_temporal_blocks * adapter.tokens_per_block
        dim = 2 * adapter.hidden_size  # views are fused on the feature axis
        print(
            f"arm {arm.name}: {arm.teacher} @{arm.input_size} grid {adapter.grid_size} "
            f"-> [{index_rows['num_rows']}, {tokens}, {dim}]",
            flush=True,
        )

        array = probe_cache.create_features(directory, index_rows["num_rows"], tokens, dim)
        max_abs, written, started = 0.0, 0, time.time()
        for batch in _loader(dataset, args.batch_size, args.num_workers):
            features = adapter.encode_video(batch["video"].numpy())
            block = features.to(torch.float32).cpu().numpy()
            max_abs = max(max_abs, float(np.abs(block).max()))
            array[written : written + block.shape[0]] = probe_cache.to_feature_dtype(block)
            written += block.shape[0]
            if written % (args.batch_size * 20) == 0:
                rate = written / (time.time() - started)
                print(f"  {written}/{index_rows['num_rows']} clips ({rate:.1f}/s)", flush=True)

        if written != index_rows["num_rows"]:
            raise RuntimeError(f"arm {arm.name}: wrote {written} of {index_rows['num_rows']} rows")
        array.flush()
        del array

        probe_cache.write_index(
            directory,
            {
                **index_rows,
                "arm": arm.name,
                "teacher": arm.teacher,
                "note": arm.note,
                "input_size": arm.input_size,
                "shortest_edge": arm.shortest_edge,
                "grid": list(adapter.grid_size),
                "native_grid": list(adapter.native_grid_size),
                "tokens": tokens,
                "dim": dim,
                "num_views": 2,
                "num_temporal_blocks": adapter.num_temporal_blocks,
                "num_frames": num_frames,
                "dtype": str(np.dtype(probe_cache.FEATURE_DTYPE)),
                # Both recorded because upstream's view fusion only pairs clips with their own views
                # at batch size 1: the pair states whether this cache is trustworthy at all.
                "correct_view_fusion": adapter.correct_view_fusion,
                "encode_batch_size": args.batch_size,
                "max_abs_activation": max_abs,
                "weights": str(weights[arm.teacher]),
                "provenance": _provenance(args.clips),
            },
        )
        print(f"arm {arm.name}: done in {time.time() - started:.0f}s, max |x| {max_abs:.1f}", flush=True)

        del adapter
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# stage: derive the pooled arm from its parent's cache
# --------------------------------------------------------------------------------------

def run_derive(args: argparse.Namespace) -> None:
    """Materialise every derived arm by pooling its parent's cached features.

    Written to disk rather than pooled on the fly so the arm is auditable and the fit stage treats all
    arms identically. The pooling is the same `SpatialTokenResampler` the adapter would apply, and
    `tests/test_i3_probe_firewall.py` pins the derived cache against a live adapter forward.

    Boundary: the parent cache is already float16, so the derived arm pools quantised inputs rather
    than the encoder's float32 output. Pooling averages two or three values, so the difference is at
    the float16 rounding level -- orders of magnitude below the seed spread -- but it does mean this
    cache is not bitwise a float32-pooled forward.
    """
    for name in args.arms:
        arm = arm_registry.arm_by_name(name)
        if not arm.is_derived:
            print(f"arm {arm.name}: not a derived arm, skipping", flush=True)
            continue

        directory = probe_cache.features_dir(args.out, arm.name)
        if (directory / probe_cache.INDEX_FILE).exists() and not args.overwrite:
            print(f"arm {arm.name}: complete cache already at {directory}, skipping", flush=True)
            continue

        parent = probe_cache.FeatureCache.open(probe_cache.features_dir(args.out, arm.derives_from))
        resampler = arm.resampler(parent.grid)
        if resampler is None:
            raise SystemExit(f"arm {arm.name} pools onto {arm.pool_to}, which is parent grid {parent.grid}")

        blocks = parent.index["num_temporal_blocks"]
        tokens = blocks * resampler.tokens_out
        array = probe_cache.create_features(directory, parent.index["num_rows"], tokens, parent.index["dim"])
        print(f"arm {arm.name}: {arm.derives_from} {parent.grid} -> {arm.pool_to} [{tokens} tokens]", flush=True)

        for start in range(0, parent.index["num_rows"], args.batch_size):
            # `np.array` copies: the memmap slice is read-only, and torch refuses to wrap that quietly.
            block = torch.from_numpy(np.array(parent.features[start : start + args.batch_size]))
            pooled = resampler(block.to(torch.float32))
            array[start : start + block.shape[0]] = probe_cache.to_feature_dtype(pooled.numpy())
        array.flush()
        del array

        probe_cache.write_index(
            directory,
            {
                **{key: parent.index[key] for key in ("num_rows", "paths", "splits", "suites")},
                "arm": arm.name,
                "teacher": arm.teacher,
                "note": arm.note,
                "input_size": arm.input_size,
                "shortest_edge": arm.shortest_edge,
                "grid": list(arm.pool_to),
                "native_grid": list(parent.grid),
                "tokens": tokens,
                "dim": parent.index["dim"],
                "num_views": parent.num_views,
                "num_temporal_blocks": blocks,
                "num_frames": parent.index["num_frames"],
                "dtype": str(np.dtype(probe_cache.FEATURE_DTYPE)),
                "derived_from": arm.derives_from,
                "derived_from_dtype": parent.index["dtype"],
                "correct_view_fusion": parent.index["correct_view_fusion"],
                "encode_batch_size": parent.index["encode_batch_size"],
                "weights": parent.index["weights"],
                "provenance": _provenance(args.clips),
            },
        )
        print(f"arm {arm.name}: written to {directory}", flush=True)


# --------------------------------------------------------------------------------------
# stage: fit the probes
# --------------------------------------------------------------------------------------

def _probe_inputs(
    cache: probe_cache.FeatureCache,
    targets: probe_cache.TargetCache,
    rows: np.ndarray,
    order: np.ndarray,
    device: str,
    delta_lag: int = 1,
) -> geo_probe.ProbeInputs:
    """Assemble one split, with the target rows reordered onto the feature cache's row order."""
    target_rows = order[rows]

    def load(array: np.ndarray, dtype) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=dtype)

    return geo_probe.ProbeInputs(
        features=load(cache.features[rows], torch.float16),
        states=load(targets.states[target_rows], torch.float32),
        states_mask=load(targets.states_mask[target_rows], torch.bool),
        deltas=load(targets.deltas[target_rows], torch.float32),
        deltas_mask=load(targets.deltas_mask[target_rows], torch.bool),
        grid=cache.grid,
        num_views=cache.num_views,
        delta_lag=delta_lag,
    )


def _metrics_from(
    predictions: torch.Tensor, data: geo_probe.ProbeInputs, kind: str, target_type: str
) -> Dict[str, Dict[str, float]]:
    """Both metric classes for one set of predictions, flattened into one dict per class prefix.

    A delta head predicts transitions only, so state-level metrics have no meaning for it; the delta
    arrays are passed as the states of the evaluation, which `ProbeInputs.target` already selects.

    Deltas are scored a second time over the moving subset. Over every token a delta metric is
    dominated by tokens that did not change, where a constant predictor is already right, so the two
    readings answer different questions: "how well is the whole grid predicted" and "how well is the
    part that moved predicted". Both are reported, neither replaces the other.
    """
    targets, mask = data.target(kind)
    predictions = predictions.expand_as(targets).contiguous()
    scored = geo_metrics.evaluate(
        pred_states=predictions,
        target_states=targets,
        states_mask=mask,
        target_type=target_type,
    )
    rows = {"metric": scored.metric, "relative": scored.relative, "counts": scored.counts}
    if kind == "delta":
        moving = geo_metrics.evaluate(
            pred_states=predictions,
            target_states=targets,
            states_mask=geo_metrics.moving_mask(targets, mask),
            target_type=target_type,
        )
        rows.update(metric_moving=moving.metric, relative_moving=moving.relative, counts_moving=moving.counts)
    return rows


def _metrics_for(head, data: geo_probe.ProbeInputs, kind: str, target_type: str) -> Dict[str, Dict[str, float]]:
    return _metrics_from(geo_probe.predict(head, data, kind), data, kind, target_type)


def _baselines_for(
    train: geo_probe.ProbeInputs, scored: geo_probe.ProbeInputs, kind: str, target_type: str
) -> Dict[str, Dict[str, float]]:
    """The feature-free floor, fitted on train and scored on `scored` exactly like the probes.

    Computed once rather than per arm: these predictors never touch the features, so every arm would
    produce the same numbers on the same splits.
    """
    return {
        name: _metrics_from(prediction, scored, kind, target_type)
        for name, prediction in geo_probe.constant_baselines(train, kind).items()
    }


def _summarise(seed_runs: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Mean and seed spread per metric. The spread is the noise floor any claim must clear."""
    summary: Dict[str, Dict[str, float]] = {}
    for metric_class in METRIC_CLASSES + MOVING_CLASSES:
        if metric_class not in seed_runs[0]:
            continue  # the moving classes are emitted for the delta kind only
        summary[metric_class] = {}
        for name in seed_runs[0][metric_class]:
            values = np.array([run[metric_class][name] for run in seed_runs], dtype=np.float64)
            summary[metric_class][name] = float(values.mean())
            summary[metric_class][f"{name}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return summary


def _floor_skill(summary: Dict[str, Dict[str, float]], floor: Dict[str, Dict[str, float]]) -> Dict:
    """Each metric's improvement over the feature-free floor, class by class.

    The direct reading of "did the representation contribute anything at all": an arm's absolute delta
    error means nothing until it is put next to a predictor that never saw a feature. Reuses
    `geo_metrics.relative_improvement`, so this introduces no second notion of improvement and a metric
    whose better direction is undeclared raises instead of being scored the wrong way round.

    Metric names are taken from the floor because it carries no `_std` keys; both dicts come from the
    same `_metrics_from` call shape, so the names line up by construction.
    """
    skill: Dict[str, Dict[str, float]] = {}
    for metric_class in METRIC_CLASSES + MOVING_CLASSES:
        names = floor.get(metric_class) or {}
        if not names or metric_class not in summary:
            continue
        skill[metric_class] = {
            name: geo_metrics.relative_improvement(value, summary[metric_class][name], name)
            for name, value in names.items()
        }
    return skill


def _delta_interval(targets: probe_cache.TargetCache) -> Dict:
    """How far apart in real time the two endpoints of one delta target are.

    `frames = tubelet_size * delta_lag * clip_stride`, converted with the LIBERO control rate. Target
    caches written before the sweep carry neither the stride nor the lag, so both default to 1 and the
    derived interval reproduces the original 2-frame contract.
    """
    lag = targets.delta_lag
    stride = int(targets.index.get("clip_stride", DEFAULT_CLIP_STRIDE))
    tubelet = int(targets.index.get("tubelet_size", TUBELET_SIZE))
    frames = tubelet * lag * stride
    return {
        "delta_lag": lag,
        "clip_stride": stride,
        "tubelet_size": tubelet,
        "frames": frames,
        "seconds": frames / CONTROL_HZ,
    }


def run_fit(args: argparse.Namespace) -> None:
    grid = tuple(args.grid)
    targets = probe_cache.TargetCache.open(probe_cache.targets_dir(args.out, grid, args.estimator, args.delta_lag))
    # The lag is read back off the cache rather than trusted from the command line: the directory name
    # encodes it, so a mismatch means the two disagree about what the targets are.
    if targets.delta_lag != args.delta_lag:
        raise SystemExit(f"target cache at lag {targets.delta_lag} was requested as lag {args.delta_lag}")
    config = geo_probe.FitConfig(
        lr_grid=args.lr_grid,
        epochs=args.epochs,
        batch_rows=args.batch_rows,
        ridge_grid=args.ridge_grid,
    )

    results: Dict[str, Dict] = {}
    # Split -> kind -> predictor name. The floor is the same on every arm, so it is computed once.
    baselines: Dict[str, Dict[str, Dict]] = {split: {} for split in REPORTED_SPLITS}
    for name in args.arms:
        cache = probe_cache.FeatureCache.open(probe_cache.features_dir(args.out, name))
        if cache.grid != grid:
            raise SystemExit(f"arm {name} publishes grid {cache.grid}, targets are on {grid}")
        order = probe_cache.align_rows(cache.paths, targets.paths)
        splits = {
            split: _probe_inputs(
                cache,
                targets,
                probe_cache.split_rows(cache.index, split),
                order,
                args.device,
                targets.delta_lag,
            )
            for split in ("train", "val", "test")
        }
        print(
            f"arm {name}: grid {cache.grid} rows "
            f"{ {split: data.num_rows for split, data in splits.items()} }",
            flush=True,
        )

        arm_results: Dict[str, Dict] = {}
        for kind in args.kinds:
            for split in REPORTED_SPLITS:
                if kind not in baselines[split]:
                    baselines[split][kind] = _baselines_for(
                        splits["train"], splits[split], kind, targets.target_type
                    )
            in_dim = geo_probe.head_input_dim(splits["train"].hidden_size, kind)
            # The firewall, stated in the run log: the teacher contributes no trainable parameter and
            # is never handed to an optimiser (`tests/test_i3_probe_firewall.py`).
            print(f"  {kind} probe: {in_dim + 1} trainable parameters, teacher 0", flush=True)
            seed_runs: Dict[str, List[Dict]] = {split: [] for split in REPORTED_SPLITS}
            seed_details = []
            for seed in args.seeds:
                started = time.time()
                fitted = geo_probe.fit_sgd(splits["train"], splits["val"], kind, seed, config, args.device)
                head = geo_probe.load_head(fitted["state_dict"], in_dim, args.device)
                for split in REPORTED_SPLITS:
                    seed_runs[split].append(_metrics_for(head, splits[split], kind, targets.target_type))
                seed_details.append(
                    {"seed": seed, "lr": fitted["lr"], "epoch": fitted["epoch"], "val_loss": fitted["val_loss"]}
                )
                print(
                    f"  {kind} seed {seed}: lr {fitted['lr']:g} epoch {fitted['epoch']} "
                    f"val {fitted['val_loss']:.4f} ({time.time() - started:.0f}s)",
                    flush=True,
                )

            ridge = geo_probe.fit_ridge(splits["train"], splits["val"], kind, config, args.device)
            ridge_head = geo_probe.load_head(ridge["state_dict"], in_dim, args.device)
            summaries = {split: _summarise(runs) for split, runs in seed_runs.items()}
            skills = {
                split: _floor_skill(summaries[split], baselines[split][kind][FLOOR_BASELINE])
                for split in REPORTED_SPLITS
            }
            arm_results[kind] = {
                # Test under the plain keys, val under `*_val`: see `REPORTED_SPLITS`.
                "summary": summaries["test"],
                f"summary{VAL_SUFFIX}": summaries["val"],
                "seeds": seed_details,
                "per_seed": seed_runs["test"],
                f"per_seed{VAL_SUFFIX}": seed_runs["val"],
                "floor_skill": skills["test"],
                f"floor_skill{VAL_SUFFIX}": skills["val"],
                "head_parameters": in_dim + 1,
                "ridge_secondary": {
                    "ridge": ridge["ridge"],
                    "val_loss": ridge["val_loss"],
                    **_metrics_for(ridge_head, splits["test"], kind, targets.target_type),
                },
            }
            for metric_class, scores in skills["test"].items():
                if geo_probe.SELECTION_METRIC in scores:
                    print(
                        f"  {kind} {metric_class} {geo_probe.SELECTION_METRIC} vs {FLOOR_BASELINE}: "
                        f"{scores[geo_probe.SELECTION_METRIC] * 100:+.2f}%",
                        flush=True,
                    )

        # `correct_view_fusion` and `encode_batch_size` travel with the numbers: a cache extracted
        # through upstream's batch>1 fusion scrambles clips against views, and the report is the only
        # place a reader would notice (`docs/provenance/upstream-conflicts.md`).
        index_keys = ("arm", "teacher", "input_size", "grid", "note", "correct_view_fusion", "encode_batch_size")
        results[name] = {"index": {key: cache.index.get(key) for key in index_keys}}
        results[name]["probes"] = arm_results
        del splits
        torch.cuda.empty_cache()

    report = {
        "grid": list(grid),
        "target_type": targets.target_type,
        "estimator": args.estimator,
        "estimator_index": targets.index.get("estimator_index"),
        "seeds": list(args.seeds),
        "selection": {
            "split": "val",
            "criterion": "masked L1 on log depth",
            "primary_metric": geo_probe.SELECTION_METRIC,
            # Where each split's numbers live. Any comparison made after seeing results -- the delta
            # interval of the sweep, for one -- has to be made on val, so val is reported too.
            "reported_splits": {"unsuffixed": "test", VAL_SUFFIX: "val"},
        },
        "fit_config": {
            "lr_grid": list(config.lr_grid),
            "epochs": list(config.epochs),
            "batch_rows": config.batch_rows,
            "ridge_grid": list(config.ridge_grid),
            "delta_lag": targets.delta_lag,
        },
        # What one delta step spans in this report, carried so a curve can be assembled from a
        # directory of reports without re-deriving it from each cache.
        "delta_interval": _delta_interval(targets),
        "arms": results,
        "baselines": baselines["test"],
        f"baselines{VAL_SUFFIX}": baselines["val"],
        "floor_baseline": FLOOR_BASELINE,
        "primary": _primary_comparison(results, args.kinds),
        "provenance": _provenance(args.clips),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"report -> {args.report}", flush=True)
    print(_markdown(report), flush=True)


def _primary_comparison(results: Dict[str, Dict], kinds: Sequence[str]) -> Dict:
    """The pre-registered A vs D comparison on AbsRel, stated against the seed noise floor.

    The gate verdict itself is the user's call: this reports the improvement, the noise floor and
    whether one exceeds the other, and stops there.
    """
    baseline_name, candidate_name = arm_registry.PRIMARY_PAIR
    metric = arm_registry.PRIMARY_METRIC
    if "state" not in kinds or not {baseline_name, candidate_name} <= results.keys():
        return {"available": False, "reason": f"needs arms {arm_registry.PRIMARY_PAIR} and the state probe"}

    summaries = {
        name: results[name]["probes"]["state"]["summary"]["metric"] for name in (baseline_name, candidate_name)
    }
    baseline, candidate = summaries[baseline_name][metric], summaries[candidate_name][metric]
    # Conservative floor: both arms' seed spreads, expressed relative to the baseline, added together.
    floor = (summaries[baseline_name][f"{metric}_std"] + summaries[candidate_name][f"{metric}_std"]) / abs(baseline)
    improvement = geo_metrics.relative_improvement(baseline, candidate, metric)
    return {
        "available": True,
        "pair": [baseline_name, candidate_name],
        "metric": metric,
        "kind": "state",
        "baseline": baseline,
        "candidate": candidate,
        "relative_improvement": improvement,
        "seed_noise_floor": floor,
        "exceeds_noise_floor": bool(improvement > floor),
        "reaches_five_percent": bool(improvement >= 0.05),
        "verdict": "user decision (AGENTS.md section 10.5): this script reports evidence only",
    }


def _moving_share(report: Dict, kind: str) -> str:
    """`valid moving / valid total` elements for one kind, or an empty string when there is no subset.

    Reported next to every moving-subset table so a strong subset number is never read without the
    count it was averaged over: at the 2-frame interval most tokens do not move at all.
    """
    counts = next((arm["probes"][kind]["per_seed"][0] for arm in report["arms"].values() if kind in arm["probes"]), {})
    if "counts_moving" not in counts:
        return ""
    moving, total = counts["counts_moving"]["states_valid"], counts["counts"]["states_valid"]
    return f", {moving} of {total} valid elements ({moving / total:.1%})" if total else ""


def _markdown(report: Dict) -> str:
    """Results table with the two metric classes kept in separate blocks."""
    source = report.get("estimator") or "simulator"
    interval = report.get("delta_interval", {})
    span = f", delta {interval['frames']} frames / {interval['seconds']:.2f}s" if interval else ""
    lines = [
        f"### I3 probe results (grid {report['grid']}, target {report['target_type']}, source {source}{span})",
        "",
    ]
    for kind in sorted({kind for arm in report["arms"].values() for kind in arm["probes"]}):
        rows = {name: arm["probes"][kind]["summary"] for name, arm in report["arms"].items() if kind in arm["probes"]}
        if not rows:
            continue
        for metric_class in METRIC_CLASSES + MOVING_CLASSES:
            names = sorted(
                {key for row in rows.values() for key in row.get(metric_class, {}) if not key.endswith("_std")}
            )
            if not names:
                continue
            label = metric_class.removesuffix("_moving") + " class"
            if metric_class in MOVING_CLASSES:
                label += (
                    f", moving subset only (|target| > {geo_metrics.TEMPORAL_DEADBAND}"
                    f"{_moving_share(report, kind)})"
                )
            lines += [f"**{kind} probe, {label}** (mean +- seed std over {len(report['seeds'])} seeds)", ""]
            lines.append("| arm | " + " | ".join(names) + " |")
            lines.append("|---|" + "---|" * len(names))
            for name in sorted(rows):
                cells = [
                    f"{rows[name][metric_class][metric]:.4f} +- {rows[name][metric_class][f'{metric}_std']:.4f}"
                    for metric in names
                ]
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
            # Feature-free floors last, in the same table: they have no seed spread, being deterministic.
            for name, scores in sorted(report.get("baselines", {}).get(kind, {}).items()):
                cells = [f"{scores[metric_class][metric]:.4f}" for metric in names]
                lines.append(f"| _{name}_ | " + " | ".join(cells) + " |")
            lines.append("")

    primary = report["primary"]
    if primary.get("available"):
        lines += [
            f"**Primary (pre-registered): {primary['pair'][0]} vs {primary['pair'][1]} on "
            f"{primary['metric']}**",
            "",
            f"- baseline {primary['baseline']:.4f}, candidate {primary['candidate']:.4f}",
            f"- relative improvement {primary['relative_improvement'] * 100:+.2f}%",
            f"- seed noise floor {primary['seed_noise_floor'] * 100:.2f}%",
            f"- exceeds noise floor: {primary['exceeds_noise_floor']}; reaches 5%: "
            f"{primary['reaches_five_percent']}",
            "- gate verdict: user decision, not asserted here",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# stage: the interval curve, assembled from a set of fit reports
# --------------------------------------------------------------------------------------

# What a sweep report is called, so pointing `--reports` at a directory cannot pick up a report from
# before the sweep: those carry no `delta_interval` and would abort the whole aggregation.
SWEEP_REPORT_GLOB = "geo_probes_s*_lag*.json"
# The two element sets a delta is read over, in render order. See `geo_metrics.moving_mask`.
READINGS = ("full", "moving")


def _split_suffix(split: str) -> str:
    """Report key suffix for a split; test is unsuffixed, for the reason `REPORTED_SPLITS` states."""
    if split not in REPORTED_SPLITS:
        raise SystemExit(f"split {split!r} is not reported; pick one of {REPORTED_SPLITS}")
    return "" if split == "test" else VAL_SUFFIX


def _report_paths(paths: Sequence[Path]) -> List[Path]:
    """The given report files, with any directory expanded to the sweep reports inside it."""
    expanded: List[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob(SWEEP_REPORT_GLOB)))
        elif path.exists():
            expanded.append(path)
        else:
            # A missing point must not silently shorten the curve, which is why this is not a skip.
            raise SystemExit(f"no report at {path}")
    if not expanded:
        raise SystemExit(f"no reports matching {SWEEP_REPORT_GLOB} under {[str(path) for path in paths]}")
    return expanded


def _load_reports(paths: Sequence[Path]) -> List[Tuple[Path, Dict]]:
    """Reports ordered along the curve's x axis, i.e. by the interval their delta targets span.

    Two reports may share an interval on purpose: the same duration reached with a wide recording
    stride and with a wide lag is the H5 contrast, so the stride breaks the tie rather than being
    treated as a duplicate.
    """
    reports = []
    for path in _report_paths(paths):
        report = json.loads(path.read_text())
        if "delta_interval" not in report:
            raise SystemExit(f"{path} predates the interval sweep and carries no delta_interval")
        reports.append((path, report))
    reports.sort(key=lambda item: (item[1]["delta_interval"]["seconds"], item[1]["delta_interval"]["clip_stride"]))
    return reports


def _curve_point(report: Dict, arm: str, kind: str, metric_class: str, metric: str, split: str) -> Dict:
    """One (interval, arm) point: the metric, the floor it is read against, the skill and the noise.

    Skill and seed spread are both divided by the floor's own value, so "did the arm beat a predictor
    that never saw a feature" and "is that difference larger than reseeding the head" are two numbers
    on one axis instead of two incomparable ones.
    """
    suffix = _split_suffix(split)
    probes = report["arms"][arm]["probes"][kind]
    summary, skills = probes[f"summary{suffix}"], probes[f"floor_skill{suffix}"]
    floors = report[f"baselines{suffix}"][kind][report.get("floor_baseline", FLOOR_BASELINE)]
    counts = probes[f"per_seed{suffix}"][0]

    point = {"arm": arm, "kind": kind, "split": split, **report["delta_interval"], "readings": {}}
    for reading in READINGS:
        scored_class = metric_class if reading == "full" else f"{metric_class}_moving"
        if metric not in summary.get(scored_class, {}):
            continue
        floor, value = floors[scored_class][metric], summary[scored_class][metric]
        spread = summary[scored_class][f"{metric}_std"]
        point["readings"][reading] = {
            "value": value,
            "spread": spread,
            "floor": floor,
            "skill": skills[scored_class][metric],
            # The seed spread on the skill axis: a skill smaller than this is not distinguishable
            # from having reseeded the head.
            "noise": spread / abs(floor) if floor else float("nan"),
            "elements": counts["counts" if reading == "full" else "counts_moving"]["states_valid"],
        }
    return point


def _ascii_curve(series: Dict[str, List[Tuple[float, float]]], height: int = 11, cell: int = 7) -> List[str]:
    """Skill in percent (y) against target interval in seconds (x), one marker character per arm.

    Plain text on purpose: it lands in the results document next to the numbers, survives review as a
    diff and needs no image viewer. The zero line is always drawn because zero is the feature-free
    floor, which is the only reference that decides whether the features contributed at all.
    """
    xs = sorted({x for points in series.values() for x, _ in points})
    ys = [y for points in series.values() for _, y in points if not math.isnan(y)]
    if not xs or not ys:
        return ["(nothing to plot)"]

    top, bottom = max(ys + [0.0]), min(ys + [0.0])
    span = (top - bottom) or 1.0
    width = len(xs) * cell
    canvas = [[" "] * width for _ in range(height)]

    def row_of(value: float) -> int:
        return int(round((top - value) / span * (height - 1)))

    zero = row_of(0.0)
    canvas[zero] = ["-"] * width
    for label, points in sorted(series.items()):
        for x, y in points:
            if not math.isnan(y):
                canvas[row_of(y)][xs.index(x) * cell] = label[0]

    lines = []
    for index, row in enumerate(canvas):
        value = top - index / (height - 1) * span
        prefix = f"{value:+7.1f} |" if index in (0, zero, height - 1) else f"{'':7} |"
        lines.append(prefix + "".join(row))
    return lines + [f"{'':8}+" + "-" * width, f"{'':9}" + "".join(f"{x:<{cell}g}" for x in xs)]


def _sweep_markdown(points: List[Dict], metric: str, metric_class: str, split: str, sources: List[Path]) -> str:
    """The curve as a table per reading, an ASCII chart of the skill, and the reports it came from."""
    lines = [
        f"### I3 delta interval sweep ({metric_class} class {metric}, {split} split, "
        f"floor {FLOOR_BASELINE})",
        "",
    ]
    for reading in READINGS:
        rows = [point for point in points if reading in point["readings"]]
        if not rows:
            continue
        subset = " (moving subset)" if reading == "moving" else " (all tokens)"
        lines += [
            f"**{metric}{subset}**",
            "",
            f"| interval s | frames | stride x lag | arm | {metric} | floor | skill % | seed noise % "
            "| clears floor | elements |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for point in rows:
            scored = point["readings"][reading]
            lines.append(
                f"| {point['seconds']:.2f} | {point['frames']} | "
                f"{point['clip_stride']} x {point['delta_lag']} | {point['arm']} | "
                f"{scored['value']:.4f} +- {scored['spread']:.4f} | {scored['floor']:.4f} | "
                f"{scored['skill'] * 100:+.2f} | {scored['noise'] * 100:.2f} | "
                f"{scored['skill'] > scored['noise']} | {scored['elements']} |"
            )
        series: Dict[str, List[Tuple[float, float]]] = {}
        for point in rows:
            series.setdefault(point["arm"], []).append((point["seconds"], point["readings"][reading]["skill"] * 100))
        lines += ["", f"skill % over the floor, {reading} reading (marker = arm):", "```", *_ascii_curve(series), "```"]
    return "\n".join(lines + ["", "Sources:", *[f"- `{path}`" for path in sources]])


def _write_figure(path: Path, points: List[Dict], metric: str, split: str) -> None:
    """The same curve as a figure for the paper: skill over the floor, with the seed spread as error.

    matplotlib is imported here rather than at module scope so the extract and fit stages, which run
    on a GPU box and never plot, do not pay for the plotting stack.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    readings = [reading for reading in READINGS if any(reading in point["readings"] for point in points)]
    figure, axes = plt.subplots(1, len(readings), figsize=(5.5 * len(readings), 4.0), squeeze=False)
    for axis, reading in zip(axes[0], readings, strict=True):
        for arm in sorted({point["arm"] for point in points}):
            rows = [point for point in points if point["arm"] == arm and reading in point["readings"]]
            axis.errorbar(
                [point["seconds"] for point in rows],
                [point["readings"][reading]["skill"] * 100 for point in rows],
                yerr=[point["readings"][reading]["noise"] * 100 for point in rows],
                marker="o",
                capsize=3,
                label=f"arm {arm}",
            )
        axis.axhline(0.0, color="black", linewidth=0.8)  # the feature-free floor
        axis.set_xlabel("delta target interval (s)")
        axis.set_ylabel(f"skill over {FLOOR_BASELINE} (%)")
        axis.set_title(f"{metric}, {'all tokens' if reading == 'full' else 'moving subset'} ({split})")
        axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(path.with_suffix(suffix), dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"figure -> {path.with_suffix('.png')} and {path.with_suffix('.pdf')}", flush=True)


def run_sweep(args: argparse.Namespace) -> None:
    reports = _load_reports(args.reports)
    points = [
        _curve_point(report, arm, args.kind, args.metric_class, args.metric, args.split)
        for _, report in reports
        for arm in args.arms
        if arm in report["arms"] and args.kind in report["arms"][arm]["probes"]
    ]
    if not points:
        raise SystemExit(f"no {args.kind} probe for arms {args.arms} in {len(reports)} reports")

    text = _sweep_markdown(points, args.metric, args.metric_class, args.split, [path for path, _ in reports])
    print(text, flush=True)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text + "\n")
        print(f"table -> {args.markdown}", flush=True)
    if args.figure:
        _write_figure(args.figure, points, args.metric, args.split)


# --------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------

def _grid_pair(text: str) -> Tuple[int, int]:
    side = int(text)
    return side, side


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="stage", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--clips", type=Path, default=DEFAULT_CLIPS, help="recorded RGB+depth clip cache")
        subparser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="probe cache root")
        subparser.add_argument("--batch-size", type=int, default=8)
        subparser.add_argument("--num-workers", type=int, default=4)

    targets = subparsers.add_parser("targets", help="pool recorded metric depth onto the token grids")
    add_common(targets)
    targets.add_argument("--grids", type=_grid_pair, nargs="+", default=list(DEFAULT_GRIDS))
    targets.add_argument("--d-min", type=float, default=DEFAULT_D_MIN)
    targets.add_argument("--d-max", type=float, default=DEFAULT_D_MAX)
    targets.add_argument(
        "--estimator",
        default=None,
        help="build targets from this estimator's pseudo-depth cache instead of the simulator depth",
    )
    targets.add_argument("--pseudo-root", type=Path, default=DEFAULT_PSEUDO, help="root of the pseudo-depth caches")
    targets.add_argument("--delta-lag", type=int, default=1, help="tubelet blocks one delta target spans")
    targets.set_defaults(handler=run_targets)

    cache = subparsers.add_parser("cache", help="extract frozen teacher features for one or more arms")
    add_common(cache)
    cache.add_argument(
        "--arms",
        nargs="+",
        default=[arm.name for arm in arm_registry.ARMS if not arm.is_derived],
        help="arms to extract; derived arms are pooled from their parent at fit time",
    )
    cache.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="pinned config holding the V-JEPA 2 path")
    cache.add_argument("--vjepa21", type=Path, default=DEFAULT_VJEPA21)
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--overwrite", action="store_true", help="re-extract arms that already have a complete cache")
    cache.set_defaults(handler=run_cache)

    derive = subparsers.add_parser("derive", help="pool a derived arm out of its parent's cached features")
    add_common(derive)
    derive.add_argument("--arms", nargs="+", default=[arm.name for arm in arm_registry.ARMS if arm.is_derived])
    derive.add_argument("--overwrite", action="store_true")
    derive.set_defaults(handler=run_derive)

    fit = subparsers.add_parser("fit", help="fit probes on the cached features and report the arm comparison")
    add_common(fit)
    fit.add_argument("--arms", nargs="+", default=["A", "C", "D"], help="arms judged on the target grid")
    fit.add_argument("--kinds", nargs="+", default=["state", "delta"], choices=["state", "delta"])
    fit.add_argument("--grid", type=int, nargs=2, default=[16, 16])
    fit.add_argument(
        "--estimator",
        default=None,
        help="fit against this estimator's pseudo targets instead of the simulator ones",
    )
    fit.add_argument("--delta-lag", type=int, default=1, help="lag of the target set to fit against")
    fit.add_argument("--seeds", type=int, nargs="+", default=list(geo_probe.DEFAULT_SEEDS))
    fit.add_argument("--lr-grid", type=float, nargs="+", default=list(geo_probe.DEFAULT_LR_GRID))
    fit.add_argument("--epochs", type=int, nargs="+", default=list(geo_probe.DEFAULT_EPOCHS))
    fit.add_argument("--ridge-grid", type=float, nargs="+", default=list(geo_probe.DEFAULT_RIDGE_GRID))
    fit.add_argument("--batch-rows", type=int, default=32)
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--report", type=Path, default=DEFAULT_OUT / "reports" / "geo_probes.json")
    fit.set_defaults(handler=run_fit)

    sweep = subparsers.add_parser("sweep", help="assemble the delta interval curve from a set of fit reports")
    sweep.add_argument(
        "--reports",
        type=Path,
        nargs="+",
        default=[DEFAULT_OUT / "reports"],
        help=f"report files, or directories searched for {SWEEP_REPORT_GLOB}",
    )
    sweep.add_argument("--arms", nargs="+", default=list(arm_registry.PRIMARY_PAIR))
    sweep.add_argument("--kind", default="delta", choices=["state", "delta"])
    sweep.add_argument("--metric-class", default="metric", choices=list(METRIC_CLASSES))
    sweep.add_argument("--metric", default=geo_probe.SELECTION_METRIC)
    sweep.add_argument(
        "--split",
        default="val",
        choices=list(REPORTED_SPLITS),
        help="val by default: the interval is chosen after seeing results, so it may not be chosen on test",
    )
    sweep.add_argument("--markdown", type=Path, default=None, help="also write the table to this file")
    sweep.add_argument("--figure", type=Path, default=None, help="write <path>.png and <path>.pdf")
    sweep.set_defaults(handler=run_sweep)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps({key: str(value) for key, value in vars(args).items() if key != "handler"}), flush=True)
    args.handler(args)


if __name__ == "__main__":
    main()
