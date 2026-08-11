# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""The pre-registered I3 teacher arms and how each one is built.

Four arms, fixed before any probe ran (`docs/implementation-plan.md`, "I3 probe arms", 2026-08-03).
Pre-registration is the point: with four arms and several metrics, picking the comparison after
seeing the numbers would inflate any apparent win (AGENTS.md section 12).

    A  V-JEPA 2   @256  ->      16x16     the pinned production teacher, the baseline
    B  V-JEPA 2.1 @384  ->      24x24     2.1 as it was distilled; extracted only as C's parent
    C  V-JEPA 2.1 @384  -> pooled 16x16   B on A's output grid, so token count is not the variable
    D  V-JEPA 2.1 @256  ->      16x16     2.1 at A's resolution: the single-variable control

Arm B carries no probe number of its own, which is a change from the plan's "supplementary 24x24
report" and a recorded limitation (user decision, 2026-08-03). Depth targets are pooled onto the
token grid with exact, non-overlapping valid-weighted windows, and 24 does not divide the recorded
256x256 depth map, so a 24x24 target grid does not exist without resampling the ground truth.
Nothing was lost that could have been compared: arm A cannot emit a 24x24 grid either, so a native-B
number would have had no baseline, and the token-count axis is unanswerable on any single grid --
which is exactly the axis arm C exists to hold fixed.

Arm D is legitimate rather than a wrong-resolution forward: `pretrained_grid_size` is 16, the RoPE
rescale factor is `(pretrained_grid_size - 1) / (patches - 1)`, and the patch grid comes from the
input tensor, so 256 is the interpolation *basis* (factor 1.0) and 384 is the interpolated case
(`tests/test_i3_adapter_vjepa21.py::test_rope_is_the_identity_at_256_and_interpolated_at_384`).

Arm C spends no forward pass of its own. The adapter applies its resampler last, after view fusion,
so pooling arm B's cached features reproduces arm C exactly -- pinned both by
`test_arm_c_is_exactly_arm_b_pooled` and by `tests/test_i3_probe_firewall.py`. Three forwards cover
four arms.

Matched field of view: 292 -> 256 and 438 -> 384 share the crop ratio 0.8767, so every arm sees the
same scene content through the same pinned processor class, and arm D's processor is byte-identical
in configuration to arm A's pinned `video_preprocessor_config.json`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

from starVLA.model.modules.world_model.spatial_token_resampler import SpatialTokenResampler
from starVLA.model.modules.world_model.teacher_loader import (
    SHORTEST_EDGE,
    TEACHER_VJEPA2,
    TEACHER_VJEPA21,
    load_teacher,
)
from starVLA.model.modules.world_model.vj_backbone_adapter import VJBackboneAdapter

# Teacher ids and the crop ratios live in `world_model.teacher_loader` because I4 loads the same two
# teachers from the training framework. Re-exported here so every probe call site keeps reading them
# off the arm registry, which is the table that pre-registers them.
__all__ = [
    "ARMS",
    "PRIMARY_METRIC",
    "PRIMARY_PAIR",
    "ProbeArm",
    "SHORTEST_EDGE",
    "TEACHER_VJEPA2",
    "TEACHER_VJEPA21",
    "arm_by_name",
    "build_adapter",
]


@dataclass(frozen=True)
class ProbeArm:
    """One pre-registered teacher configuration.

    Attributes:
        name: single-letter arm id used in every cache path, table and log line.
        teacher: `TEACHER_VJEPA2` or `TEACHER_VJEPA21`; selects the loader, not the geometry.
        input_size: square edge fed to the encoder. May differ from what the config states, which is
            why `VJBackboneAdapter` takes an explicit `input_size`.
        pool_to: token grid to average-pool onto, or None to publish the native grid.
        derives_from: name of the arm whose cached features this one is pooled from. Set exactly when
            the arm needs no forward pass of its own.
        note: why the arm is in the table, carried into the results table.
    """

    name: str
    teacher: str
    input_size: int
    pool_to: Optional[Tuple[int, int]] = None
    derives_from: Optional[str] = None
    note: str = ""

    @property
    def shortest_edge(self) -> int:
        return SHORTEST_EDGE[self.input_size]

    @property
    def is_derived(self) -> bool:
        return self.derives_from is not None

    def resampler(self, native_grid: Tuple[int, int]) -> Optional[SpatialTokenResampler]:
        """Pooling this arm applies to `native_grid`, or None when it publishes the native grid."""
        if self.pool_to is None or tuple(self.pool_to) == tuple(native_grid):
            return None
        return SpatialTokenResampler(grid_in=tuple(native_grid), grid_out=tuple(self.pool_to))


ARMS: Tuple[ProbeArm, ...] = (
    ProbeArm("A", TEACHER_VJEPA2, 256, note="pinned production teacher (baseline)"),
    ProbeArm("B", TEACHER_VJEPA21, 384, note="2.1 native 24x24; extracted as C's parent, not judged"),
    ProbeArm("C", TEACHER_VJEPA21, 384, pool_to=(16, 16), derives_from="B", note="2.1 on A's grid"),
    ProbeArm("D", TEACHER_VJEPA21, 256, note="2.1 at A's resolution (primary control)"),
)

# gate (a) is judged on this pair and this metric alone; everything else is secondary.
PRIMARY_PAIR = ("A", "D")
PRIMARY_METRIC = "abs_rel"


def arm_by_name(name: str) -> ProbeArm:
    for arm in ARMS:
        if arm.name == name:
            return arm
    raise KeyError(f"unknown arm {name!r}; registered arms are {[arm.name for arm in ARMS]}")


def build_adapter(
    arm: ProbeArm,
    weights: Mapping[str, Path],
    num_frames: int,
    device: str = "cuda",
) -> VJBackboneAdapter:
    """Load `arm`'s frozen teacher and wrap it in the adapter with `arm`'s geometry.

    Args:
        weights: teacher id -> local weight directory. Passed in rather than hardcoded so the caller
            keeps the pinned config as the single source of truth for the V-JEPA 2 path.
        num_frames: clip length; must match the recorded cache.

    Returns:
        An adapter whose `grid_size` is the arm's published grid. The encoder is already frozen and
        in `eval()` by `VJBackboneAdapter.__init__`.
    """
    try:
        encoder, processor = load_teacher(
            teacher=arm.teacher,
            root=Path(weights[arm.teacher]),
            input_size=arm.input_size,
            device=device,
        )
    except (FileNotFoundError, ValueError) as error:
        raise type(error)(f"arm {arm.name}: {error}") from error

    native = arm.input_size // encoder.config.patch_size
    return VJBackboneAdapter(
        encoder=encoder,
        processor=processor,
        num_frames=num_frames,
        resampler=arm.resampler((native, native)),
        input_size=arm.input_size,
        # Probes read one clip's features against that clip's depth, so the fusion must pair views
        # with their own clip at any batch size. Upstream's default only does that at batch 1.
        correct_view_fusion=True,
    )
