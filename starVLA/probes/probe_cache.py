# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""On-disk layout of the I3 probe cache: frozen teacher features and pooled depth targets.

The probes run in two stages (`docs/implementation-plan.md` section 5): extract frozen features once
per arm, then fit every probe and seed on top of them. Splitting it this way is what makes the study
affordable -- three forwards cover four arms and all three seeds -- and it also removes a whole class
of confound, because every seed of every probe reads the *same bytes* rather than re-running a
nominally deterministic encoder.

Layout, all of it outside the repository:

    <root>/features/<arm>/features.npy   float16 [N, blocks * tokens_per_block, V * hidden]
    <root>/features/<arm>/index.json     geometry, row order, provenance
    <root>/targets/<h>x<w>/targets.npz   states/deltas and their masks, float32 + bool
    <root>/targets/<h>x<w>/index.json    row order and target contract

Two invariants:

* `index.json` is written only after the array is flushed, so its presence means "cache complete".
  A crashed extraction leaves a `features.npy` with no index and is refused rather than silently
  probed on zeros.
* Row order is stored explicitly, as the clip paths in manifest order. Features and targets are
  matched by comparing those lists, never by assuming two independent runs enumerated the cache the
  same way.

Features are stored as float16. That is a deliberate precision loss: it halves ~28 GB of cache and is
far below the seed-to-seed spread a probe is compared against, but it does mean probe numbers are not
bitwise comparable to a float32 forward. The observed maximum absolute activation is recorded in the
index so the headroom against the float16 range stays auditable.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FEATURES_DIR = "features"
TARGETS_DIR = "targets"
FEATURES_FILE = "features.npy"
TARGETS_FILE = "targets.npz"
INDEX_FILE = "index.json"

FEATURE_DTYPE = np.float16
# float16 saturates at 65504. Anything within a factor of ~8 of that is close enough to the edge that
# the cast stops being a rounding detail, so extraction fails instead of writing infinities.
FEATURE_ABS_MAX = 8000.0


def features_dir(root, arm_name: str) -> Path:
    return Path(root) / FEATURES_DIR / arm_name


def targets_dir(root, grid: Tuple[int, int]) -> Path:
    return Path(root) / TARGETS_DIR / f"{grid[0]}x{grid[1]}"


def create_features(directory, num_rows: int, tokens: int, dim: int) -> np.memmap:
    """Allocate the writable feature array. The index is written afterwards by `write_index`."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        directory / FEATURES_FILE,
        mode="w+",
        dtype=FEATURE_DTYPE,
        shape=(num_rows, tokens, dim),
    )


def to_feature_dtype(features: np.ndarray) -> np.ndarray:
    """Cast to the cache dtype, refusing values that the cast could not represent."""
    max_abs = float(np.abs(features).max()) if features.size else 0.0
    if not np.isfinite(max_abs):
        raise ValueError("teacher features contain NaN or Inf")
    if max_abs > FEATURE_ABS_MAX:
        raise ValueError(f"activation magnitude {max_abs:.1f} is too close to the float16 range")
    return features.astype(FEATURE_DTYPE)


def write_index(directory, index: Dict) -> None:
    """Write the index last, so its presence marks the cache as complete."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / INDEX_FILE).write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


def read_index(directory) -> Dict:
    path = Path(directory) / INDEX_FILE
    if not path.exists():
        raise FileNotFoundError(f"no complete cache at {directory} (missing {INDEX_FILE})")
    return json.loads(path.read_text())


@dataclass(frozen=True)
class FeatureCache:
    """One arm's frozen features, memory-mapped, plus the row order they were written in."""

    root: Path
    index: Dict
    features: np.ndarray

    @classmethod
    def open(cls, directory) -> "FeatureCache":
        directory = Path(directory)
        index = read_index(directory)
        features = np.load(directory / FEATURES_FILE, mmap_mode="r")
        expected = (index["num_rows"], index["tokens"], index["dim"])
        if features.shape != expected:
            raise ValueError(f"{directory}: features {features.shape} do not match index {expected}")
        return cls(root=directory, index=index, features=features)

    @property
    def arm(self) -> str:
        return self.index["arm"]

    @property
    def grid(self) -> Tuple[int, int]:
        return tuple(self.index["grid"])

    @property
    def paths(self) -> List[str]:
        return self.index["paths"]

    @property
    def num_views(self) -> int:
        return self.index["num_views"]

    @property
    def hidden_size(self) -> int:
        return self.index["dim"] // self.num_views


@dataclass(frozen=True)
class TargetCache:
    """Pooled depth targets on one token grid, shared by every arm judged on that grid."""

    root: Path
    index: Dict
    states: np.ndarray
    states_mask: np.ndarray
    deltas: np.ndarray
    deltas_mask: np.ndarray

    @classmethod
    def open(cls, directory) -> "TargetCache":
        directory = Path(directory)
        index = read_index(directory)
        with np.load(directory / TARGETS_FILE) as arrays:
            fields = {name: arrays[name] for name in ("states", "states_mask", "deltas", "deltas_mask")}
        return cls(root=directory, index=index, **fields)

    @property
    def grid(self) -> Tuple[int, int]:
        return tuple(self.index["grid"])

    @property
    def paths(self) -> List[str]:
        return self.index["paths"]

    @property
    def target_type(self) -> str:
        return self.index["target_type"]


def write_targets(
    directory,
    states: np.ndarray,
    states_mask: np.ndarray,
    deltas: np.ndarray,
    deltas_mask: np.ndarray,
    index: Dict,
) -> None:
    """Store one grid's targets. Small enough (tens of MB) to keep as a single loadable archive."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        directory / TARGETS_FILE,
        states=states,
        states_mask=states_mask,
        deltas=deltas,
        deltas_mask=deltas_mask,
    )
    write_index(directory, index)


def align_rows(feature_paths: Sequence[str], target_paths: Sequence[str]) -> np.ndarray:
    """Row indices that reorder the target cache onto the feature cache's row order.

    Both caches are written from the same manifest, so this is normally the identity. It is computed
    rather than assumed because a probe silently fitting arm A's features against arm B's clip order
    would still converge, just to a meaningless number.
    """
    position = {path: row for row, path in enumerate(target_paths)}
    missing = [path for path in feature_paths if path not in position]
    if missing:
        raise KeyError(f"{len(missing)} feature rows absent from the target cache, e.g. {missing[0]}")
    return np.array([position[path] for path in feature_paths], dtype=np.int64)


def split_rows(index: Dict, split: Optional[str]) -> np.ndarray:
    """Row indices belonging to `split`, or all rows when `split` is None."""
    splits = index["splits"]
    if split is None:
        return np.arange(len(splits), dtype=np.int64)
    rows = np.array([row for row, value in enumerate(splits) if value == split], dtype=np.int64)
    if rows.size == 0:
        raise ValueError(f"no rows for split {split!r}; cache holds {sorted(set(splits))}")
    return rows
