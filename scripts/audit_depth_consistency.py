# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""Metric-versus-pseudo depth consistency audit (iteration-plan section 7, I3 gate condition c).

Scores every precomputed pseudo-depth cache against the recorded simulator metric depth, so the
estimator bake-off is decided on measurements rather than on model cards. Runs in the pinned
`envs/dynaweave` -- unlike `scripts/precompute_depth_targets.py`, which needs `envs/da3` -- precisely
so that the pseudo depth goes through the *same* `depth_targets` builder and the *same* `geo_metrics`
implementation the S3 probes were judged with. Nothing about the comparison is reimplemented here.

Three metric blocks per estimator, deliberately not collapsed into one:

* `metric_raw` -- the metric class on the metres as cached, unaligned. This is the only block that tests
  the absolute scale, whether the estimator produced metres itself or `precompute_depth_targets.py`
  converted canonical depth with the recorded focal length; an estimator that is right only up to an
  unknown factor scores badly here and well below.
* `metric_aligned` -- the same metrics after a least-squares scale and shift in log space, solved per
  (clip, view). Per view rather than per clip because a monocular estimator picks its gauge per image
  and the two cameras see wildly different depth ranges (agentview 0.70-3.07 m, wrist 0.04-0.38 m);
  scoring one shared gauge across both would charge the estimator for a mismatch it never claimed.
* `relative` -- `geo_metrics.relative_depth_metrics` unchanged, i.e. the per-clip gauge the S3 probe
  arms were scored under, so the two tables can be read against each other.

Two grids, because the estimator matters at two different resolutions: the dense 256x256 grid says how
good the depth map is, and the 16x16 token grid says how much of that survives into the quantity a
probe or an I4 loss would actually see.

Aggregation is per clip and then averaged (NaN-skipping) rather than pooled: every clip carries equal
weight, and a metric that is undefined on some clips -- `boundary_f1` on a boundary-free crop,
`temporal_sign_agreement` on a still clip -- reports how many clips it was defined on instead of
silently changing its denominator.

Usage:

    python scripts/audit_depth_consistency.py \
        --clips /vepfs/wangshilong/data/dynaweave/i3_geo_clips \
        --pseudo-root /vepfs/wangshilong/data/dynaweave/i3_pseudo_depth \
        --estimators DA3METRIC-LARGE Metric-Video-Depth-Anything-Large DA3NESTED-GIANT-LARGE
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

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
    range_clip_mask,
    sensor_valid_mask,
)
from starVLA.probes import geo_metrics  # noqa: E402

DEFAULT_CLIPS = Path("/vepfs/wangshilong/data/dynaweave/i3_geo_clips")
DEFAULT_PSEUDO = Path("/vepfs/wangshilong/data/dynaweave/i3_pseudo_depth")
DEFAULT_REPORT = Path("/vepfs/wangshilong/data/dynaweave/i3_probe_cache/reports/depth_consistency.json")
ESTIMATOR_INDEX = "estimator.json"

TUBELET_SIZE = 2
# Dense pixels first, then the token grid the S3 probes and the I4 loss are defined on.
DEFAULT_GRIDS: Tuple[Optional[Tuple[int, int]], ...] = (None, (16, 16))
# The headline columns of the markdown table, in reading order. The JSON keeps every metric.
TABLE_METRICS = ("abs_rel", "rmse", "delta1", "log_mae", "gradient", "boundary_f1", "temporal_sign_agreement")
RELATIVE_TABLE_METRICS = ("aligned_state_mae", "gradient", "temporal_rank_consistency")


# --------------------------------------------------------------------------------------
# pseudo cache reading
# --------------------------------------------------------------------------------------

def load_pseudo_depth(root: Path, relative_path: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """One clip of a pseudo cache as `[V, T, 1, H, W]` metres, matching the dataset's layout."""
    with np.load(root / relative_path) as clip:
        depth = torch.from_numpy(clip["depth_m"].astype(np.float32))
        conf = torch.from_numpy(clip["conf"].astype(np.float32)) if "conf" in clip.files else None
    # Cache layout is [T, V, H, W]; consumers want [V, T, 1, H, W].
    reorder = (1, 0, 2, 3)
    depth = depth.permute(*reorder).unsqueeze(2).contiguous()
    if conf is not None:
        conf = conf.permute(*reorder).unsqueeze(2).contiguous()
    return depth, conf


def read_estimator_index(root: Path, allow_partial: bool) -> Dict:
    """The cache's own index, refusing a partial cache unless the caller asked for a subset.

    A partial cache would still produce a full-looking table, just over fewer clips, so the check is
    here rather than left to whoever reads the report.
    """
    path = root / ESTIMATOR_INDEX
    if not path.exists():
        raise SystemExit(f"{root} has no {ESTIMATOR_INDEX}: the precompute run did not finish")
    index = json.loads(path.read_text())
    if not index.get("complete", True) and not allow_partial:
        raise SystemExit(
            f"{root} holds {index.get('num_present')} of {index.get('num_clips')} clips; "
            "finish the precompute run or pass --limit to audit a subset on purpose"
        )
    return index


# --------------------------------------------------------------------------------------
# per-clip scoring
# --------------------------------------------------------------------------------------

def align_per_view(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """`geo_metrics.align_scale_shift` with the view axis folded into the sample axis.

    A pure reshape around the existing solver: `align_scale_shift(per_sample=True)` solves one gauge
    per leading-axis element, so moving `V` there gives one gauge per (clip, view) with no new maths.
    """
    # `[N, Tp, V, 1, h, w] -> [N * V, Tp, 1, h, w]`
    def fold(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.permute(0, 2, 1, 3, 4, 5).reshape(-1, *tensor.shape[1:2], *tensor.shape[3:])

    folded = geo_metrics.align_scale_shift(fold(pred), fold(target), fold(mask))
    num_rows, num_views = pred.shape[0], pred.shape[2]
    unfolded = folded.reshape(num_rows, num_views, *folded.shape[1:2], *folded.shape[2:])
    return unfolded.permute(0, 2, 1, 3, 4, 5).contiguous()


def score_clip(
    gt_depth: torch.Tensor,
    gt_valid: torch.Tensor,
    pseudo_depth: torch.Tensor,
    grid: Optional[Tuple[int, int]],
    d_min: float,
    d_max: float,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
    """Score one clip's pseudo depth against its simulator depth on one grid.

    Args:
        gt_depth / gt_valid: `[1, V, T, 1, H, W]` recorded metric depth and its sensor mask.
        pseudo_depth: `[1, V, T, 1, H, W]` estimator metres, same layout.
        grid: token grid to pool onto, or None for the dense pixel grid.

    Returns:
        `(blocks, counts)` where `blocks` holds `metric_raw` / `metric_aligned` / `relative`.
    """
    gt_states, gt_deltas = build_metric_delta_targets(
        cache_depth=gt_depth,
        valid=gt_valid,
        tubelet_size=TUBELET_SIZE,
        grid=grid,
        d_min=d_min,
        d_max=d_max,
        target_type=TARGET_TYPE_METRIC,
    )
    # No `valid=`: an estimator publishes no sensor mask, so its own validity is only finiteness and
    # positivity, which `log_metric_depth` derives. Keeping the two masks apart is what makes the
    # invalid-pixel columns below separable (gate condition c).
    pseudo_states, pseudo_deltas = build_metric_delta_targets(
        cache_depth=pseudo_depth,
        valid=None,
        tubelet_size=TUBELET_SIZE,
        grid=grid,
        d_min=d_min,
        d_max=d_max,
        target_type=TARGET_TYPE_PSEUDO_METRIC,
    )

    states_mask = gt_states.mask & pseudo_states.mask
    deltas_mask = gt_deltas.mask & pseudo_deltas.mask
    shared = {
        "target_states": gt_states.values,
        "states_mask": states_mask,
        "pred_deltas": pseudo_deltas.values,
        "target_deltas": gt_deltas.values,
        "deltas_mask": deltas_mask,
    }

    aligned = align_per_view(pseudo_states.values, gt_states.values, states_mask)
    blocks = {
        "metric_raw": geo_metrics.metric_depth_metrics(pseudo_states.values, **shared),
        "metric_aligned": geo_metrics.metric_depth_metrics(aligned, **shared),
        "relative": geo_metrics.relative_depth_metrics(pseudo_states.values, **shared),
    }
    counts = {
        "states_total": int(states_mask.numel()),
        "states_shared_valid": int(states_mask.sum()),
        "gt_invalid": int((~gt_states.mask).sum()),
        "pseudo_invalid": int((~pseudo_states.mask).sum()),
        "deltas_total": int(deltas_mask.numel()),
        "deltas_shared_valid": int(deltas_mask.sum()),
    }
    return blocks, counts


def pixel_counts(gt_depth: torch.Tensor, gt_valid: torch.Tensor, pseudo_depth: torch.Tensor, d_min: float, d_max: float):
    """Dense pixel bookkeeping, kept apart from the pooled scores so gate (c) can be read directly."""
    gt_mask = sensor_valid_mask(gt_depth, gt_valid)
    pseudo_mask = sensor_valid_mask(pseudo_depth, None)
    return {
        "pixels_total": int(gt_mask.numel()),
        "pixels_gt_valid": int(gt_mask.sum()),
        "pixels_gt_invalid": int((~gt_mask).sum()),
        "pixels_pseudo_invalid": int((~pseudo_mask).sum()),
        "pixels_shared_valid": int((gt_mask & pseudo_mask).sum()),
        "pixels_gt_range_clipped": int(range_clip_mask(gt_depth, d_min, d_max).sum()),
        "pixels_pseudo_range_clipped": int(range_clip_mask(pseudo_depth, d_min, d_max).sum()),
    }


# --------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------

def nan_summary(values: Sequence[float]) -> Dict[str, float]:
    """Mean and spread over the clips where the metric was defined, plus how many those were."""
    finite = [value for value in values if not math.isnan(value)]
    if not finite:
        return {"mean": float("nan"), "std": float("nan"), "clips": 0}
    array = np.asarray(finite, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "clips": int(array.size),
    }


def aggregate(per_clip: List[Dict]) -> Dict:
    """Group per-clip blocks by grid and suite, then by grid alone ("all")."""
    grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for record in per_clip:
        for scope in (record["suite"], "all"):
            for block, metrics in record["blocks"].items():
                for name, value in metrics.items():
                    grouped[record["grid"]][f"{scope}|{block}"][name].append(value)
            for name, value in record["counts"].items():
                counts[f"{record['grid']}|{scope}"][name] += value

    summarised = {
        grid: {scope: {name: nan_summary(values) for name, values in block.items()} for scope, block in scopes.items()}
        for grid, scopes in grouped.items()
    }
    return {"metrics": summarised, "counts": {key: dict(value) for key, value in counts.items()}}


# --------------------------------------------------------------------------------------
# markdown rendering
# --------------------------------------------------------------------------------------

def _cell(summary: Dict[str, float]) -> str:
    if summary["clips"] == 0:
        return "n/a"
    return f"{summary['mean']:.4f}"


def _table(
    report: Dict,
    grid: str,
    block: str,
    metrics: Sequence[str],
    estimators: Sequence[str],
    scope: str = "all",
) -> List[str]:
    header = f"| estimator | {' | '.join(metrics)} |"
    lines = [header, "|" + "---|" * (len(metrics) + 1)]
    for estimator in estimators:
        summaries = report[estimator]["metrics"].get(grid, {}).get(f"{scope}|{block}", {})
        cells = [_cell(summaries.get(name, {"clips": 0})) for name in metrics]
        lines.append(f"| `{estimator}` | {' | '.join(cells)} |")
    return lines


def render_markdown(report: Dict, estimators: Sequence[str], suites: Sequence[str], grids: Sequence[str]) -> str:
    lines = [
        "# I3 metric-versus-pseudo depth consistency",
        "",
        "Generated by `scripts/audit_depth_consistency.py`. Reference is the simulator metric depth",
        "recorded in the I3 clip cache; every estimator is scored against it through the same",
        "`depth_targets` builder and the same `geo_metrics` implementation as the S3 probe arms.",
        "",
        "`metric_raw` is unaligned and therefore the only block that tests the absolute metric scale.",
        "`metric_aligned` solves a least-squares scale and shift in log space per (clip, view).",
        "`relative` is the per-clip relative class, unchanged from S3 so the two tables are comparable.",
        "Cells are means over the clips where the metric is defined; per-metric clip counts and standard",
        "deviations are in the JSON next to this file.",
        "",
        f"Clips scored: {report[estimators[0]]['num_clips_scored']} of "
        f"{report[estimators[0]]['num_clips_available']} in the clip cache.",
        "",
    ]
    for grid in grids:
        lines += [f"## Grid {grid}", ""]
        for block, metrics in (
            ("metric_raw", TABLE_METRICS),
            ("metric_aligned", TABLE_METRICS),
            ("relative", RELATIVE_TABLE_METRICS),
        ):
            lines += [f"### {block}", "", *_table(report, grid, block, metrics, estimators), ""]

        lines += ["### metric_aligned by suite", ""]
        header = f"| estimator | {' | '.join(suites)} |"
        lines += [header, "|" + "---|" * (len(suites) + 1)]
        for estimator in estimators:
            cells = []
            for suite in suites:
                summaries = report[estimator]["metrics"].get(grid, {}).get(f"{suite}|metric_aligned", {})
                cells.append(_cell(summaries.get("abs_rel", {"clips": 0})))
            lines.append(f"| `{estimator}` | {' | '.join(cells)} |")
        lines += ["", "AbsRel, aligned, per LIBERO suite.", ""]

    lines += ["## Pixel bookkeeping (gate condition c)", ""]
    columns = (
        "pixels_total",
        "pixels_gt_invalid",
        "pixels_pseudo_invalid",
        "pixels_shared_valid",
        "pixels_gt_range_clipped",
        "pixels_pseudo_range_clipped",
    )
    lines += [f"| estimator | {' | '.join(columns)} |", "|" + "---|" * (len(columns) + 1)]
    for estimator in estimators:
        pixels = report[estimator]["pixels"]
        lines.append(f"| `{estimator}` | {' | '.join(str(pixels.get(name, 0)) for name in columns)} |")
    lines += [
        "",
        "Simulator invalid pixels come from the recorded sensor mask; estimator invalid pixels are",
        "non-finite or non-positive predictions. The two are counted separately and never unioned into",
        "a single \"bad pixel\" number.",
        "",
        "## Estimator provenance",
        "",
    ]
    for estimator in estimators:
        index = report[estimator]["index"]
        lines += [
            f"- `{estimator}`: backend `{index['backend']}`, "
            f"model reported metric {index.get('model_reported_metric')}, "
            f"canonical conversion applied {index.get('canonical_conversion_applied')}, "
            f"conf {'yes' if index.get('has_conf') else 'no'}, settings `{json.dumps(index['settings'])}`",
        ]
    lines += [
        "",
        "`model reported metric` is what the estimator claimed for itself; `canonical conversion"
        " applied` is whether `precompute_depth_targets.py` had to multiply canonical depth by the"
        " recorded focal length to reach metres. The two are recorded separately because only the first"
        " is the estimator's own claim.",
        "",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def audit_estimator(args: argparse.Namespace, estimator: str, dataset: DepthClipCacheDataset) -> Dict:
    root = args.pseudo_root / estimator
    index = read_estimator_index(root, allow_partial=args.limit is not None)
    grids = [None if grid == "dense" else tuple(int(part) for part in grid.split("x")) for grid in args.grids]

    per_clip: List[Dict] = []
    pixels: Dict[str, int] = defaultdict(int)
    started = time.time()
    num_clips = len(dataset) if args.limit is None else min(args.limit, len(dataset))

    for position in range(num_clips):
        sample = dataset[position]
        gt_depth = sample["depth"].unsqueeze(0).to(args.device)
        gt_valid = sample["valid"].unsqueeze(0).to(args.device)
        pseudo_depth, _conf = load_pseudo_depth(root, sample["path"])
        pseudo_depth = pseudo_depth.unsqueeze(0).to(args.device)
        if pseudo_depth.shape != gt_depth.shape:
            raise SystemExit(f"{estimator} {sample['path']}: {tuple(pseudo_depth.shape)} != {tuple(gt_depth.shape)}")

        for name, count in pixel_counts(gt_depth, gt_valid, pseudo_depth, args.d_min, args.d_max).items():
            pixels[name] += count

        for grid in grids:
            blocks, counts = score_clip(gt_depth, gt_valid, pseudo_depth, grid, args.d_min, args.d_max)
            per_clip.append(
                {
                    "grid": "dense" if grid is None else f"{grid[0]}x{grid[1]}",
                    "suite": sample["suite"],
                    "split": sample["split"],
                    "blocks": blocks,
                    "counts": counts,
                }
            )

        if (position + 1) % 100 == 0:
            rate = (position + 1) / (time.time() - started)
            print(f"  {estimator}: {position + 1}/{num_clips} clips ({rate:.1f}/s)", flush=True)

    result = aggregate(per_clip)
    result["index"] = index
    result["pixels"] = dict(pixels)
    result["num_clips_scored"] = num_clips
    result["num_clips_available"] = len(dataset)
    return result


def run(args: argparse.Namespace) -> None:
    dataset = DepthClipCacheDataset(root=args.clips, split=None)
    suites = sorted({row["suite"] for row in dataset.rows})
    print(f"audit: {len(dataset)} clips, suites {suites}, estimators {args.estimators}", flush=True)

    report = {estimator: audit_estimator(args, estimator, dataset) for estimator in args.estimators}
    grids = [grid for grid in args.grids]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "clips": str(args.clips),
        "pseudo_root": str(args.pseudo_root),
        "d_min": args.d_min,
        "d_max": args.d_max,
        "tubelet_size": TUBELET_SIZE,
        "grids": grids,
        "suites": suites,
        "estimators": dict(report),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown = args.report.with_suffix(".md")
    markdown.write_text(render_markdown(report, args.estimators, suites, grids))
    print(f"audit: {args.report}\naudit: {markdown}", flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=Path, default=DEFAULT_CLIPS)
    parser.add_argument("--pseudo-root", type=Path, default=DEFAULT_PSEUDO)
    parser.add_argument("--estimators", nargs="+", required=True)
    parser.add_argument(
        "--grids",
        nargs="+",
        default=["dense", "16x16"],
        help='"dense" for the recorded pixel grid, or "<h>x<w>" for a pooled token grid',
    )
    parser.add_argument("--d-min", type=float, default=DEFAULT_D_MIN)
    parser.add_argument("--d-max", type=float, default=DEFAULT_D_MAX)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N clips (smoke tests)")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
