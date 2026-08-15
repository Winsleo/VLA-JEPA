# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""Offline pseudo-depth precomputation for the I3 estimator bake-off (implementation-plan section 5).

Runs in the separate `envs/da3` environment and **must not import starVLA**: `depth-anything-3` pulls
`numpy<2`, `open3d`, `pycolmap` and `moviepy==1.0.3`, which cannot coexist with the pinned
`envs/dynaweave` that carries the I1/I2 parity result (`docs/provenance/environment.md`). The
consequence is that this file reimplements the two things it needs from the clip cache -- reading the
per-suite manifests and loading a clip's RGB -- rather than importing `DepthClipCacheDataset`. Both are
a handful of lines and the cache layout is a written contract, so the duplication is cheaper than a
shared dependency that would drag the estimator stack into the training environment.

Everything downstream of the raw depth map (log-clipping, tubelet alignment, pooling, deltas) is
*not* done here. That maths lives in `starVLA/model/modules/world_model/depth_targets.py` and is
applied later, in `envs/dynaweave`, to the metres this script writes -- so the pseudo path and the
simulator path go through the identical target builder and differ only in where the metres came from.

One inference call per (clip, view): implementation-plan section 6 asks for whole-clip inference, and
both estimator families exploit temporal context, but the two cameras are physically different
sensors, so mixing them into one call would ask a static-scene multi-view model to reconcile a
third-person and a wrist view of a moving scene. Views stay separate; frames do not.

Nothing is passed *into* the networks except pixels: no extrinsics and no intrinsics. The bake-off asks
what an estimator delivers from RGB alone, which is the situation I4 would be in on a dataset without
depth; handing DA3 the true poses would also let it align its scale to the ground truth
(`api._align_to_input_extrinsics_intrinsics`) and quietly turn the metric comparison into a relative
one.

The recorded focal length *is* used afterwards, for one estimator, as a unit conversion. `DA3METRIC-LARGE`
emits canonical depth, not metres -- its model card says "Canonical metric depth; multiplying by focal
length gives metric depth", and upstream only performs that multiplication inside
`NestedDepthAnything3._apply_metric_scaling` (`depth * focal / 300`), never on the standalone metric
model, which does not even predict intrinsics. So `canonical_to_metric` below applies it here, taking
the focal from the episode's recorded camera matrix. This is a deterministic per-camera constant, not a
fit: unlike `geo_metrics.align_scale_shift` it never looks at the ground-truth depth, and a calibrated
camera is something a real deployment has. Measured over the full cache, the converted median lands
within 0.88-1.67 of the ground-truth median per suite and view, which is enough to confirm the constant
and that the focal belongs to the *processed* resolution -- and not enough to call the absolute scale
trustworthy (the wrist camera's AbsRel exceeds 0.7). `docs/provenance/upstream-conflicts.md` UC-002 has
the table, the decision and that boundary; the audit's unaligned `metric_raw` block keeps the number
visible rather than assuming it.

Two input sources feed the same estimator loop, selected by `--source`:

    clips    the I3 recorded clip cache (`--clips`), 8-frame clips with recorded intrinsics. This is
             the path the I3 bake-off ran on; adding the `lerobot` source did not change it.
    lerobot  the I4 training set (`--lerobot-root`), one inference per (episode, view) over the whole
             episode. Added for I4: the training set carries no depth, so the transition target has to
             be estimated over the very frames the trainer decodes.

Only the *input adapter* differs. Backend, canonical-to-metric conversion, resolution reduction,
provenance and resume behaviour are shared, so a pseudo depth map means the same thing on both.

"Did not change it" is deliberately not "bit-for-bit identical": the DA3 forward is not bitwise
deterministic on this GPU. Recomputing three I3 clips twice with *unmodified* code reproduced two of
them exactly and differed on the third by one pixel out of 1,048,576 (4.883e-04 m, one float16 step
at that magnitude). That non-determinism predates this file's source refactor and is a property of
the estimator, not of the cache contract -- I3 gate (b) asserts the *target* maths is reproducible
from a fixed depth cache, which is a separate claim and still holds.

Usage (see `docs/provenance/environment.md` for the env):

    /vepfs/wangshilong/envs/da3/bin/python scripts/precompute_depth_targets.py \
        --estimator DA3METRIC-LARGE \
        --weights /vepfs/wangshilong/models/dynaweave/depth_estimators/DA3METRIC-LARGE \
        --clips /vepfs/wangshilong/data/dynaweave/i3_geo_clips \
        --out /vepfs/wangshilong/data/dynaweave/i3_pseudo_depth

    /vepfs/wangshilong/envs/da3/bin/python scripts/precompute_depth_targets.py \
        --source lerobot \
        --estimator DA3METRIC-LARGE \
        --weights /vepfs/wangshilong/models/dynaweave/depth_estimators/DA3METRIC-LARGE \
        --lerobot-root /vepfs/wangshilong/data/dynaweave/LEROBOT_LIBERO_DATA \
        --out /vepfs/wangshilong/data/dynaweave/i4_pseudo_depth
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_GLOB = "manifest_*.jsonl"
EPISODE_META = "meta.json"
ESTIMATOR_INDEX = "estimator.json"
# Pseudo depth is stored at half precision: the largest recorded LIBERO depth is ~3.1 m, where float16
# quantises to ~2e-3 m (a 6e-4 relative step), three orders of magnitude below any estimator's error,
# and it halves a 6.7 GB per-estimator cache. The simulator ground truth stays float32.
DEPTH_DTYPE = np.float16

# DA3's own default processing resolution. Worth stating rather than inheriting silently: the clips are
# 256x256, `upper_bound_resize` scales the longest side to 504, and 504 = 36 * 14 is already a whole
# number of DA3 patches, so the divisibility rounding step is a no-op and the only geometric operation
# is a clean 256 -> 504 upscale followed by the 504 -> 256 reduction below.
DA3_PROCESS_RES = 504
DA3_PROCESS_RES_METHOD = "upper_bound_resize"

# The canonical-depth constant of `depth_anything_3.utils.alignment.apply_metric_scaling`. Copied from
# upstream rather than tuned: `metres = canonical * focal / 300` with the focal in pixels of the
# network's own input grid. See the module docstring for the measurement that confirms both.
DA3_CANONICAL_FOCAL = 300.0


# --------------------------------------------------------------------------------------
# input sources
# --------------------------------------------------------------------------------------
#
# A source answers three questions and nothing else: which units of work exist, where each one's
# output file goes, and how to load one unit's `([T, V, H, W, 3] uint8 RGB, per-view focal)`. Keeping
# it that narrow is what lets the estimator loop below stay identical for both, which is the property
# that makes an I3 pseudo depth map and an I4 pseudo depth map the same kind of number.

def read_index(out: Path) -> Dict[str, object]:
    """The index of a partially built cache, or `{}` when there is none yet."""
    path = out / ESTIMATOR_INDEX
    return json.loads(path.read_text()) if path.exists() else {}


class ClipCacheSource:
    """The I3 recorded clip cache: 8-frame clips with the recorder's own intrinsics.

    The manifest reading and RGB loading are the minimum duplicated from
    `starVLA/dataloader/depth_cache_dataset.py` -- see the module docstring for why this file cannot
    import starVLA.
    """

    name = "clips"
    unit = "clip"

    def __init__(self, clips: Path) -> None:
        self.clips = clips
        rows: List[dict] = []
        for manifest in sorted(clips.glob(MANIFEST_GLOB)):
            for line in manifest.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        if not rows:
            raise SystemExit(f"no {MANIFEST_GLOB} rows under {clips}")
        # Ordered by path so this cache's rows match the probe cache's rows.
        self.keys: List[str] = [row["path"] for row in sorted(rows, key=lambda row: row["path"])]

    def destination(self, key: str, out: Path) -> Path:
        return out / key

    def load(self, key: str) -> Tuple[np.ndarray, List[float]]:
        clip_path = self.clips / key
        with np.load(clip_path) as clip:
            # Exactly as recorded -- no flip, crop or resize is applied here.
            rgb = np.ascontiguousarray(clip["rgb"])  # [T, V, H, W, 3]
        # The average of `f_x` and `f_y` matches what `apply_metric_scaling` uses upstream; for LIBERO
        # the two are equal.
        meta = json.loads((clip_path.parent / EPISODE_META).read_text())
        intrinsics = meta["intrinsics"]
        focals = [0.5 * (intrinsics[view][0][0] + intrinsics[view][1][1]) for view in meta["views"]]
        return rgb, focals

    def provenance(self) -> Dict[str, object]:
        return {"source": self.name, "clip_cache": str(self.clips)}


# LIBERO's cameras are fixed and its renders are all 256 wide, so the focal is a per-camera constant
# rather than per-episode metadata -- which is just as well, because the LeRobot conversion kept no
# intrinsics at all. Measured, not assumed: all 200 I3 recording episodes report byte-identical
# matrices (`f_x == f_y`, principal point 128), and both datasets render the same two LIBERO cameras
# at the same 256x256 (`agentview` -> `observation.images.image`, `robot0_eye_in_hand` ->
# `observation.images.wrist_image`).
#
# What rides on these numbers is smaller than it looks. They only enter through
# `canonical_to_metric`, i.e. as one constant factor per camera, and I4's target is a *delta of log
# depth* -- `log(a*d2) - log(a*d1)` drops `a` exactly. So the constants affect only which pixels fall
# inside the `[d_min, d_max]` clip, never the supervised signal. D-048's standing warning that this
# estimator's absolute scale is not trustworthy therefore does not propagate into the I4 target.
LEROBOT_FOCALS = {
    "observation.images.image": 309.01933598375615,
    "observation.images.wrist_image": 166.81284772367434,
}
LEROBOT_FOCAL_WIDTH = 256
# View order, which must match `Libero4in1DataConfig.video_keys` (`primary_image`, `wrist_image`) so
# that axis `V` of the cache means the same camera as axis `V` of the trainer's video batch.
LEROBOT_VIEW_KEYS: Tuple[str, ...] = ("observation.images.image", "observation.images.wrist_image")


class EpisodeDecodeError(RuntimeError):
    """A recoverable decode failure with the episode/view provenance needed for a retry."""


def decode_episode_video(path: str, retries: int = 0) -> np.ndarray:
    """One episode's frames as `[T, H, W, 3]` uint8, decoded the way the trainer decodes them.

    `starVLA/dataloader/lerobot_datasets.py` pins `video_backend="torchvision_av"`, so the frames the
    model sees come from `torchvision.io.VideoReader` on the pyav backend. This reads the same reader
    straight through instead of re-seeking per timestamp: measured over 4 suites x 3 episodes x 2
    views, a sequential pass is **byte-identical** to the trainer's per-timestamp seek and yields
    exactly `episodes.jsonl`'s `length` frames, so frame `i` here is frame `i` there. That equality is
    what makes an absolute frame index a valid cache key; `tests/test_i4_depth_alignment.py` keeps it
    honest. Decord is *not* used -- neither environment's build can open these AV1 streams.
    """
    import torchvision

    torchvision.set_video_backend("pyav")
    last_error = None
    for attempt in range(retries + 1):
        try:
            reader = torchvision.io.VideoReader(path, "video")
            frames = []
            try:
                frames = [frame["data"].cpu().numpy() for frame in reader]
            finally:
                # torchvision's pyav reader otherwise keeps its C++ decoder alive until GC. This
                # script opens two streams per episode for thousands of episodes, so release it at
                # the episode boundary rather than allowing native handles to accumulate.
                if hasattr(reader, "_c"):
                    reader._c = None
                container = getattr(reader, "container", None)
                if container is not None:
                    container.close()
            if not frames:
                raise EpisodeDecodeError(f"decoded no frames from {path}")
            return np.stack(frames).transpose(0, 2, 3, 1)  # [T, C, H, W] -> [T, H, W, 3]
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(0.25 * (attempt + 1))
    raise EpisodeDecodeError(f"failed to decode {path} after {retries + 1} attempt(s): {last_error}") from last_error


class LeRobotSource:
    """The I4 training set: one unit of work per episode, all views together.

    Unlike the clip cache there is no manifest, so the work list is `meta/episodes.jsonl` per dataset
    and the video path template comes from `meta/info.json` rather than being hardcoded. Episode
    length is cross-checked against the decoded frame count instead of trusted, because a silent
    truncation here would shift every depth frame relative to its RGB frame.
    """

    name = "lerobot"
    unit = "episode"

    def __init__(self, root: Path, datasets: Optional[Sequence[str]] = None, decode_retries: int = 0) -> None:
        self.root = root
        self.decode_retries = decode_retries
        names = sorted(path.name for path in root.iterdir() if (path / "meta" / "info.json").is_file())
        if datasets is not None:
            missing = sorted(set(datasets) - set(names))
            if missing:
                raise SystemExit(f"no such dataset(s) under {root}: {missing}")
            names = [name for name in names if name in datasets]
        if not names:
            raise SystemExit(f"no LeRobot datasets under {root}")
        self.datasets = names
        self._info = {name: json.loads((root / name / "meta" / "info.json").read_text()) for name in names}
        self._lengths: Dict[str, int] = {}
        self.keys: List[str] = []
        for name in names:
            lines = (root / name / "meta" / "episodes.jsonl").read_text().splitlines()
            for line in lines:
                if not line.strip():
                    continue
                episode = json.loads(line)
                key = f"{name}/episode_{episode['episode_index']:06d}.npz"
                self._lengths[key] = int(episode["length"])
                self.keys.append(key)

    def destination(self, key: str, out: Path) -> Path:
        return out / key

    def _video_path(self, dataset: str, episode_index: int, view_key: str) -> str:
        info = self._info[dataset]
        return (
            self.root
            / dataset
            / info["video_path"].format(
                episode_chunk=episode_index // info["chunks_size"],
                video_key=view_key,
                episode_index=episode_index,
            )
        ).as_posix()

    def load(self, key: str) -> Tuple[np.ndarray, List[float]]:
        dataset, stem = key.split("/", 1)
        episode_index = int(stem[len("episode_") : -len(".npz")])
        views = []
        for view in LEROBOT_VIEW_KEYS:
            path = self._video_path(dataset, episode_index, view)
            try:
                views.append(decode_episode_video(path, retries=self.decode_retries))
            except EpisodeDecodeError as error:
                raise EpisodeDecodeError(f"{key} {view}: {error}") from error
        expected = self._lengths[key]
        for view_key, frames in zip(LEROBOT_VIEW_KEYS, views):
            if frames.shape[0] != expected:
                raise SystemExit(
                    f"{key} {view_key}: decoded {frames.shape[0]} frames, episodes.jsonl says {expected}"
                )
        rgb = np.stack(views, axis=1)  # [T, V, H, W, 3]
        return rgb, [LEROBOT_FOCALS[view] for view in LEROBOT_VIEW_KEYS]

    def provenance(self) -> Dict[str, object]:
        return {
            "source": self.name,
            "lerobot_root": str(self.root),
            "datasets": self.datasets,
            "view_keys": list(LEROBOT_VIEW_KEYS),
            "video_backend": "torchvision.io.VideoReader (pyav), read sequentially",
            "focals": dict(LEROBOT_FOCALS),
            "focal_width": LEROBOT_FOCAL_WIDTH,
            "focal_provenance": (
                "per-camera constants; LIBERO's cameras are fixed and all 200 I3 recording episodes"
                " report byte-identical intrinsics at 256x256. They enter only as one constant factor"
                " per camera in canonical_to_metric and therefore cancel in a log-depth delta."
            ),
        }


def build_source(args: argparse.Namespace) -> object:
    if args.source == "clips":
        return ClipCacheSource(args.clips)
    if args.source == "lerobot":
        return LeRobotSource(args.lerobot_root, args.datasets, args.decode_retries)
    raise SystemExit(f"unknown source {args.source!r}")


def canonical_to_metric(depth: np.ndarray, focal_recorded: float, recorded_width: int) -> np.ndarray:
    """Turn DA3 canonical depth into metres: `depth * focal_processed / 300`.

    The focal is rescaled from the recorded grid to the grid `depth` is actually on, because a canonical
    depth is defined against the focal in pixels of the network's *input*. The ratio is taken from the
    returned map's own width rather than assuming `DA3_PROCESS_RES`, so a different `process_res` stays
    correct instead of silently mis-scaling.
    """
    focal_processed = focal_recorded * depth.shape[-1] / recorded_width
    return depth * (focal_processed / DA3_CANONICAL_FOCAL)


# --------------------------------------------------------------------------------------
# estimator backends
# --------------------------------------------------------------------------------------

def _is_metric(prediction: object) -> bool:
    """Normalise `Prediction.is_metric`, which upstream cannot report as `False`.

    `output_processor.py` fills the field with `getattr(model_output, "is_metric", 0)` on an
    `addict.Dict`, whose attribute access autovivifies -- so a model that never sets the flag yields an
    empty `Dict` rather than the intended `0`. Only `NestedDepthAnything3` sets it, to the int `1`.
    Requiring an int equal to 1 therefore recovers the meaning the field was supposed to carry.
    """
    flag = getattr(prediction, "is_metric", 0)
    return isinstance(flag, int) and not isinstance(flag, bool) and flag == 1


class DepthAnything3Backend:
    """`DA3METRIC-LARGE` / `DA3NESTED-GIANT-LARGE` through the upstream `DepthAnything3` API.

    The model is a multi-view predictor, so a clip's frames go in as one "scene". Whether its depth is
    already metric is read off the prediction (via `_is_metric`) rather than assumed from the model
    name: the nested model rescales its own output internally, the standalone metric model returns
    canonical depth and leaves the multiplication to the caller.
    """

    name = "da3"

    def __init__(self, weights: Path, device: str) -> None:
        from depth_anything_3.api import DepthAnything3  # local import: only this backend needs it

        self.model = DepthAnything3.from_pretrained(str(weights)).to(device).eval()
        self.device = device
        self.settings = {
            "process_res": DA3_PROCESS_RES,
            "process_res_method": DA3_PROCESS_RES_METHOD,
            "extrinsics_passed": False,
            "intrinsics_passed": False,
        }

    def __call__(
        self,
        frames: np.ndarray,
        focal_recorded: float,
        recorded_width: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
        """Metric depth for one camera's clip.

        Args:
            frames: `[T, H, W, 3]` uint8, one camera.
            focal_recorded: that camera's focal length in pixels of the recorded grid.
            recorded_width: width of the recorded grid the focal belongs to.

        Returns:
            `(depth [T, h, w] in metres, conf or None, model_reported_metric)`. The depth is in metres
            either way; the flag says whether the model got there by itself or the canonical conversion
            was applied.
        """
        prediction = self.model.inference(
            image=[frames[index] for index in range(frames.shape[0])],
            process_res=DA3_PROCESS_RES,
            process_res_method=DA3_PROCESS_RES_METHOD,
            export_dir=None,
        )
        self_metric = _is_metric(prediction)
        depth = prediction.depth
        if not self_metric:
            depth = canonical_to_metric(depth, focal_recorded, recorded_width)
        return depth, prediction.conf, self_metric


class MetricVideoDepthAnythingBackend:
    """`Metric-Video-Depth-Anything-Large` through the upstream `Video-Depth-Anything` repository.

    Not a pip package: the code is a GitHub checkout added to `sys.path`, and the checkout's commit is
    recorded in the cache index. Its `infer_video_depth` consumes a whole clip and returns metric
    depth per frame with no confidence channel -- so `conf` is absent for this estimator rather than
    faked, and the audit reports that asymmetry instead of hiding it.

    `metric=True` is not cosmetic: it is the constructor flag `run.py --metric` sets, and inside
    `infer_video_depth` it replaces the cross-window scale/shift fit with the identity, which is what
    keeps the output in metres. A clip loaded into a `metric=False` model would still run and would
    still look like depth.
    """

    name = "video_depth_anything"
    ENCODER = "vitl"
    # Upstream's own default for this checkpoint. `INFER_LEN = 32` is marked "do not change" upstream,
    # and an 8-frame clip is padded up to it by repeating its last frame, so the clip is a single
    # window -- one temporal context, no chunk stitching, and the 24 padding frames are static.
    INPUT_SIZE = 518

    def __init__(self, weights: Path, repo: Path, device: str) -> None:
        # Only the checkout root goes on the path: the metric head is a constructor flag on the same
        # module tree, not a separate `metric_depth/` package, and `video_depth.py` imports
        # `utils.util` relative to this root.
        if not (repo / "video_depth_anything" / "video_depth.py").is_file():
            raise SystemExit(f"{repo} does not look like a Video-Depth-Anything checkout")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from video_depth_anything.video_depth import VideoDepthAnything

        # Upstream's `run.py` hard-codes these per-encoder dimensions; copied rather than guessed.
        configs = {
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        checkpoint = weights if weights.is_file() else weights / f"metric_video_depth_anything_{self.ENCODER}.pth"
        self.model = VideoDepthAnything(**configs[self.ENCODER], metric=True)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
        self.model = self.model.to(device).eval()
        self.device = device
        self.checkpoint = checkpoint
        self.repo_commit = _git_commit(repo)
        self.settings = {
            "input_size": self.INPUT_SIZE,
            "encoder": self.ENCODER,
            "repo": str(repo),
            "repo_commit": self.repo_commit,
            "fp32": True,
            "metric_head": True,
        }

    def __call__(
        self,
        frames: np.ndarray,
        focal_recorded: float,
        recorded_width: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
        """Same contract as `DepthAnything3Backend.__call__`; the focal is unused, see the return."""
        del focal_recorded, recorded_width  # this head is metric on its own, nothing to convert
        depth, _fps = self.model.infer_video_depth(
            frames,
            target_fps=-1,
            input_size=self.INPUT_SIZE,
            device=self.device,
            fp32=True,
        )
        # This checkpoint's own head emits metres directly; there is no canonical-depth step to undo.
        return np.asarray(depth), None, True


def build_backend(args: argparse.Namespace) -> object:
    if args.backend == "da3":
        return DepthAnything3Backend(args.weights, args.device)
    if args.backend == "video_depth_anything":
        return MetricVideoDepthAnythingBackend(args.weights, args.vda_repo, args.device)
    raise SystemExit(f"unknown backend {args.backend!r}")


# --------------------------------------------------------------------------------------
# resolution reduction
# --------------------------------------------------------------------------------------

def to_cache_resolution(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    """Reduce an estimator's `[T, h, w]` depth back onto the recorded `[T, height, width]` grid.

    Area averaging, matching `depth_targets.pool_to_grid`: both reductions average the metres falling
    inside a cell, so the pseudo path and the simulator path are reduced the same way and a difference
    between them cannot be an artefact of one using nearest-neighbour and the other a mean.
    """
    if depth.shape[-2:] == (height, width):
        return depth.astype(np.float32, copy=False)
    tensor = torch.from_numpy(np.ascontiguousarray(depth)).to(torch.float32).unsqueeze(1)
    resized = F.interpolate(tensor, size=(height, width), mode="area")
    return resized.squeeze(1).numpy()


# --------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------

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


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def weight_provenance(weights: Path, hf_revision: Optional[str] = None) -> Dict[str, object]:
    """Per-file sha256, so a cache can be re-identified without trusting a directory name.

    `snapshot_download(local_dir=...)` leaves no revision marker behind, so the Hub commit is only
    recorded when the caller passes it; the hashes are the identity that does not depend on that.
    """
    files = sorted(path for path in weights.rglob("*") if path.is_file() and ".cache" not in path.parts)
    return {
        "weights_root": str(weights),
        "hf_revision": hf_revision,
        "files": {
            str(path.relative_to(weights)): {"bytes": path.stat().st_size, "sha256": sha256_of(path)}
            for path in files
            if path.suffix in (".safetensors", ".pth", ".pt", ".json")
        },
    }


# --------------------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------------------

def pending_jobs(
    source: object,
    keys: Sequence[str],
    out: Path,
    overwrite: bool,
    limit: Optional[int] = None,
) -> Iterator[Tuple[str, Path]]:
    """Units still to compute. Already-written ones are skipped, which makes a run resumable."""
    yielded = 0
    for key in keys:
        if limit is not None and yielded >= limit:
            return
        destination = source.destination(key, out)
        if destination.exists() and not overwrite:
            continue
        yielded += 1
        yield key, destination


def read_decode_failures(path: Path) -> Dict[str, Dict[str, str]]:
    """Latest failure provenance by cache key, preserving unresolved failures across resumes."""
    if not path.exists():
        return {}
    failures = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            failures[record["key"]] = record
    return failures


def run(args: argparse.Namespace) -> None:
    # `--stride` takes every n-th clip in manifest order, which spans all four suites instead of the
    # first suite `--limit` alone would give. Deterministic from the manifest, so two runs of the same
    # stride in different environments compute the *same* clips and their outputs are comparable.
    # It narrows the *work list* only: the index below still counts coverage against the whole
    # manifest, so a strided cache reports itself incomplete instead of redefining "complete".
    # `--offset` shifts the start, so `--stride n` with offsets 0..n-1 shards one cache across n GPUs
    # without two workers picking the same clip.
    source = build_source(args)
    all_keys = source.keys
    keys = all_keys[args.offset :: args.stride]
    out = args.out / args.estimator
    out.mkdir(parents=True, exist_ok=True)

    backend = build_backend(args)
    print(f"{args.estimator}: {len(keys)} of {len(all_keys)} {source.unit}s -> {out}", flush=True)

    # Two separate facts, never merged: what the model itself claimed, and whether this script had to
    # perform the canonical-to-metric multiplication to reach metres. Both describe the *cache*, not
    # this run, so a resumed run that writes nothing seeds them from the existing index instead of
    # replacing recorded flags with empty lists.
    previous = read_index(out)
    self_metric_seen = set(previous.get("model_reported_metric", []))
    converted_seen = set(previous.get("canonical_conversion_applied", []))
    written, started = 0, time.time()
    failure_path = out / "decode_failures.jsonl"
    failures = read_decode_failures(failure_path)
    for key, destination in pending_jobs(source, keys, out, args.overwrite, args.limit):
        try:
            rgb, focals = source.load(key)  # [T, V, H, W, 3], one focal per view
        except EpisodeDecodeError as error:
            if not args.record_decode_failures:
                raise
            failure = {"key": key, "error": str(error), "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            if key not in failures:
                with failure_path.open("a") as handle:
                    handle.write(json.dumps(failure, sort_keys=True) + "\n")
            failures[key] = failure
            print(f"WARNING: skipped undecodable {key}: {error}", flush=True)
            continue
        # A resumed run may repair a previously recorded decode failure.  Remove the stale
        # provenance entry so `num_decode_failures` describes unresolved failures, not history.
        failures.pop(key, None)
        num_frames, num_views, height, width = rgb.shape[:4]

        depths, confs = [], []
        for view in range(num_views):
            depth, conf, self_metric = backend(rgb[:, view], focal_recorded=focals[view], recorded_width=width)
            self_metric_seen.add(self_metric)
            converted_seen.add(not self_metric)
            depths.append(to_cache_resolution(depth, height, width))
            confs.append(None if conf is None else to_cache_resolution(conf, height, width))

        # Back to the cache's `[T, V, H, W]` layout so the pseudo cache is a drop-in for `depth_m`.
        stacked = np.stack(depths, axis=1).astype(DEPTH_DTYPE)
        payload = {"depth_m": stacked}
        if all(conf is not None for conf in confs):
            payload["conf"] = np.stack(confs, axis=1).astype(DEPTH_DTYPE)

        # Write beside the destination and rename: a crashed or killed run then leaves no half-written
        # `.npz` that a later resume would skip as "already done", and two workers sharding the same
        # cache cannot interleave into one file -- the loser's rename is simply overwritten.
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f"{destination.stem}.partial-{os.getpid()}.npz")
        np.savez_compressed(partial, **payload)
        os.replace(partial, destination)
        written += 1
        if written % 50 == 0:
            rate = written / (time.time() - started)
            print(f"  {written} {source.unit}s ({rate:.2f}/s, last {num_frames}x{num_views} frames)", flush=True)

    # A resumed or `--limit`ed run must not overwrite a finished index with an empty-looking one, so
    # `is_metric` / `has_conf` fall back to a unit already on disk and `complete` says which it is.
    written_keys = [key for key in all_keys if source.destination(key, out).exists()]
    present = len(written_keys)
    # A prior failed run can leave a failure record after an external repair or a successful
    # resume that skipped the already-present destination.  The destination is authoritative:
    # retain only unresolved keys so the index and sidecar describe the same cache state.
    failures = {key: record for key, record in failures.items() if not source.destination(key, out).exists()}
    if written:
        has_conf = "conf" in payload
    elif present:
        with np.load(source.destination(written_keys[0], out)) as clip:
            has_conf = "conf" in clip.files
    else:
        has_conf = None

    if args.record_decode_failures:
        if failures:
            failure_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in failures.values()))
        elif failure_path.exists():
            failure_path.unlink()

    index = {
        "estimator": args.estimator,
        "backend": backend.name,
        "settings": backend.settings,
        "model_reported_metric": sorted(self_metric_seen),
        "canonical_conversion_applied": sorted(converted_seen),
        "canonical_conversion": (
            f"metres = canonical * focal_processed / {DA3_CANONICAL_FOCAL:g}, focal from the episode's"
            " recorded intrinsics rescaled to the processed grid; applied only when the model does not"
            " report metric itself. See docs/provenance/upstream-conflicts.md."
        ),
        "units": "meter",
        "has_conf": has_conf,
        "depth_dtype": np.dtype(DEPTH_DTYPE).name,
        "resize_to_cache": "torch area interpolation back to the recorded 256x256 grid",
        "per_inference_unit": f"one call per ({source.unit}, view): all frames of one camera in one forward",
        "num_clips": len(all_keys),
        "num_written": written,
        "num_present": present,
        "num_decode_failures": len(failures),
        "stride": args.stride,
        "offset": args.offset,
        "complete": present == len(all_keys),
        **source.provenance(),
        "paths": list(all_keys),
        "weights": (
            weight_provenance(args.weights, args.hf_revision)
            if args.weights.is_dir()
            else {"file": str(args.weights), "sha256": sha256_of(args.weights), "hf_revision": args.hf_revision}
        ),
        "torch": torch.__version__,
        "vla_jepa_commit": _git_commit(REPO_ROOT),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / ESTIMATOR_INDEX).write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"{args.estimator}: wrote {written} {source.unit}s, index -> {out / ESTIMATOR_INDEX}", flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--estimator", required=True, help="cache subdirectory name, e.g. DA3METRIC-LARGE")
    parser.add_argument("--backend", choices=("da3", "video_depth_anything"), default="da3")
    parser.add_argument("--weights", type=Path, required=True, help="local weight directory or checkpoint file")
    parser.add_argument(
        "--vda-repo",
        type=Path,
        default=Path("/vepfs/wangshilong/third_party/Video-Depth-Anything"),
        help="Video-Depth-Anything checkout (video_depth_anything backend only)",
    )
    parser.add_argument(
        "--source",
        choices=("clips", "lerobot"),
        default="clips",
        help="input adapter: the I3 clip cache (default, unchanged) or the I4 LeRobot training set",
    )
    parser.add_argument("--clips", type=Path, default=Path("/vepfs/wangshilong/data/dynaweave/i3_geo_clips"))
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=Path("/vepfs/wangshilong/data/dynaweave/LEROBOT_LIBERO_DATA"),
        help="root holding one directory per LeRobot dataset (lerobot source only)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="restrict the lerobot source to these dataset directories; default is all of them",
    )
    parser.add_argument(
        "--decode-retries",
        type=int,
        default=0,
        help="retry each LeRobot AV1 decode this many times before surfacing its episode/view provenance",
    )
    parser.add_argument(
        "--record-decode-failures",
        action="store_true",
        help="continue after exhausted LeRobot decode retries and append provenance to decode_failures.jsonl",
    )
    parser.add_argument("--out", type=Path, default=Path("/vepfs/wangshilong/data/dynaweave/i3_pseudo_depth"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true", help="recompute clips that already have a file")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many clips (smoke tests)")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="take every n-th clip in manifest order; a strided cache is marked incomplete",
    )
    parser.add_argument("--offset", type=int, default=0, help="first clip index of the strided work list")
    parser.add_argument("--hf-revision", default=None, help="Hub commit the weights were downloaded at")
    args = parser.parse_args(argv)
    if args.decode_retries < 0:
        parser.error("--decode-retries must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())
