"""Dataset over the I3 RGB + depth clip cache written by `examples/LIBERO/record_geo_clips.py`.

Owns cache IO, sample lookup and metadata only; the target maths lives in
`starVLA/model/modules/world_model/depth_targets.py` (`docs/implementation-plan.md` section 5). The
one invariant it must enforce is the augmentation alignment of AGENTS.md section 7: crop, flip and
view reorder are applied identically to RGB, depth and the validity mask, while photometric jitter
touches RGB only. Any misalignment silently corrupts every depth target downstream, so the geometry
parameters are drawn once per sample and shared across modalities.

Resizing is deliberately not offered: depth must not be interpolated across object boundaries, and
the recorder already renders at the teacher's resolution. Temporal subsampling and frame dropping
(also listed in section 6) are out of scope while clips are a fixed 8 frames; adding them later must
reuse the same shared-parameter mechanism.

Sample layout matches the adapter's `encode_video` convention and the depth path of section 4.1:

    video  uint8    [V, T, 3, H, W]
    depth  float32  [V, T, 1, H, W]   metres
    valid  bool     [V, T, 1, H, W]   sensor-range mask recorded with the clip
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

MANIFEST_GLOB = "manifest_*.jsonl"


@dataclass(frozen=True)
class ClipAugmentation:
    """Geometric transforms shared by every modality, plus an RGB-only photometric one.

    Defaults are all off: the I3 probes read the cache unaugmented so that arms differ only by
    teacher. The alignment contract is still exercised by the unit tests.

    Attributes:
        crop_size: square random crop side; None keeps the full frame.
        horizontal_flip: flip with probability 0.5 (shared across modalities).
        swap_views: exchange the two views with probability 0.5. Only meaningful for models that do
            not rely on view identity, since the teacher concatenates views on the feature axis.
        brightness: RGB-only multiplicative jitter, sampled in `[1 - brightness, 1 + brightness]`.
    """

    crop_size: Optional[int] = None
    horizontal_flip: bool = False
    swap_views: bool = False
    brightness: float = 0.0

    @property
    def is_identity(self) -> bool:
        return self.crop_size is None and not self.horizontal_flip and not self.swap_views and self.brightness == 0.0

    def __call__(
        self,
        video: torch.Tensor,
        depth: torch.Tensor,
        valid: torch.Tensor,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply one sampled parameter set to all three tensors.

        Args:
            video: `[V, T, 3, H, W]` uint8.
            depth: `[V, T, 1, H, W]` float32.
            valid: `[V, T, 1, H, W]` bool.
            generator: the sample's RNG; the same seed must reproduce the same transform.
        """
        if self.crop_size is not None:
            top, left = self._sample_crop(video.shape[-2], video.shape[-1], generator)
            box = (slice(top, top + self.crop_size), slice(left, left + self.crop_size))
            video, depth, valid = (tensor[..., box[0], box[1]] for tensor in (video, depth, valid))

        if self.horizontal_flip and self._coin(generator):
            video, depth, valid = (tensor.flip(-1) for tensor in (video, depth, valid))

        if self.swap_views and self._coin(generator):
            video, depth, valid = (tensor.flip(0) for tensor in (video, depth, valid))

        if self.brightness > 0.0:
            scale = 1.0 + self.brightness * (2.0 * torch.rand((), generator=generator, dtype=torch.float32) - 1.0)
            video = (video.to(torch.float32) * scale).clamp_(0, 255).to(torch.uint8)

        return video, depth, valid

    def _sample_crop(self, height: int, width: int, generator: torch.Generator) -> Tuple[int, int]:
        if self.crop_size > min(height, width):
            raise ValueError(f"crop_size {self.crop_size} exceeds the {height}x{width} frame")
        top = int(torch.randint(0, height - self.crop_size + 1, (), generator=generator))
        left = int(torch.randint(0, width - self.crop_size + 1, (), generator=generator))
        return top, left

    @staticmethod
    def _coin(generator: torch.Generator) -> bool:
        return bool(torch.rand((), generator=generator) < 0.5)


def load_manifest(
    root: Path,
    split: Optional[str] = None,
    suites: Optional[Sequence[str]] = None,
) -> List[dict]:
    """Read every per-suite manifest under `root` and filter it.

    One manifest per suite exists because the recorder runs one process per suite. Rows are sorted by
    clip path so the dataset order is independent of the order the recorders happened to finish in.
    """
    rows: List[dict] = []
    for manifest in sorted(root.glob(MANIFEST_GLOB)):
        for line in manifest.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))

    if split is not None:
        rows = [row for row in rows if row["split"] == split]
    if suites is not None:
        allowed = set(suites)
        rows = [row for row in rows if row["suite"] in allowed]
    return sorted(rows, key=lambda row: row["path"])


class DepthClipCacheDataset(Dataset):
    """Clips from the recorded cache, with aligned augmentation and per-episode metadata.

    Args:
        root: cache root containing `manifest_<suite>.jsonl` and the per-suite clip tree.
        split: `"train"`, `"val"`, `"test"`, or None for everything.
        suites: restrict to these LIBERO suites, or None for all recorded ones.
        augmentation: shared-parameter augmentation; the default is a no-op.
        seed: base seed. Sample `i` uses `seed + i`, so augmentation is reproducible per sample and
            independent of loader worker count or iteration order.
    """

    def __init__(
        self,
        root,
        split: Optional[str] = None,
        suites: Optional[Sequence[str]] = None,
        augmentation: Optional[ClipAugmentation] = None,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.rows = load_manifest(self.root, split=split, suites=suites)
        if not self.rows:
            raise ValueError(f"no clips under {self.root} for split={split!r} suites={suites!r}")
        self.augmentation = augmentation or ClipAugmentation()
        self.seed = seed
        self._episode_meta: Dict[Path, dict] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def episode_meta(self, clip_path: Path) -> dict:
        """Recording contract of the episode a clip belongs to (cached per episode directory)."""
        episode_dir = clip_path.parent
        if episode_dir not in self._episode_meta:
            self._episode_meta[episode_dir] = json.loads((episode_dir / "meta.json").read_text())
        return self._episode_meta[episode_dir]

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        clip_path = self.root / row["path"]
        with np.load(clip_path) as clip:
            # Cache layout is [T, V, ...]; models consume [V, T, ...].
            video = torch.from_numpy(clip["rgb"]).permute(1, 0, 4, 2, 3).contiguous()
            depth = torch.from_numpy(clip["depth_m"]).permute(1, 0, 2, 3).unsqueeze(2).contiguous()
            valid = torch.from_numpy(clip["valid"]).permute(1, 0, 2, 3).unsqueeze(2).contiguous()

        if not self.augmentation.is_identity:
            generator = torch.Generator().manual_seed(self.seed + index)
            video, depth, valid = self.augmentation(video, depth, valid, generator)

        meta = self.episode_meta(clip_path)
        return {
            "video": video,
            "depth": depth,
            "valid": valid,
            "target_type": row["target_type"],
            "depth_units": meta["depth_units"],
            "z_near": meta["z_near"],
            "z_far": meta["z_far"],
            "suite": row["suite"],
            "task_id": row["task_id"],
            "episode_index": row["episode_index"],
            "clip_index": row["clip_index"],
            "start": row["start"],
            "split": row["split"],
            "path": row["path"],
        }
