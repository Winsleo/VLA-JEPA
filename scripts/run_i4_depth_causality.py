"""Evaluate a frozen I4 depth head under correct, shuffled, and zero action-token conditions."""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from scripts.run_i4_token_probe import _clip_examples, _instructions, capture_action_tokens
from starVLA.dataloader.depth_cache_dataset import DepthClipCacheDataset
from starVLA.model.modules.world_model.depth_targets import build_metric_delta_targets
from starVLA.probes.i4_depth_causality import condition_error_sums


def _load_frozen_model(checkpoint: Path, device: str):
    from starVLA.model.framework.base_framework import baseframework

    model = baseframework.from_pretrained(checkpoint.as_posix())
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("I4 causal evaluation requires a fully frozen eval-mode checkpoint")
    if not getattr(model, "depth_enabled", False):
        raise RuntimeError("checkpoint does not enable the I4 depth head")
    return model


def _report(totals: Dict[str, float], checkpoint: Path, args: argparse.Namespace) -> Dict[str, object]:
    count = totals.pop("valid_count")
    if count <= 0:
        raise RuntimeError("depth cache contains no valid target cells")
    errors = {
        name.removesuffix("_absolute_error_sum"): total / count
        for name, total in totals.items()
        if name.endswith("_absolute_error_sum")
    }
    correct, shuffled, zero = errors["correct"], errors["shuffled"], errors["zero"]
    return {
        "checkpoint": checkpoint.as_posix(),
        "clips": args.clips.as_posix(),
        "target_type": args.target_type,
        "tubelet_size": args.tubelet_size,
        "delta_lag": args.delta_lag,
        "grid": list(args.grid),
        "condition_masked_l1": errors,
        "zero_condition_improvement": (zero - correct) / zero,
        "shuffled_condition_degradation": (shuffled - correct) / correct,
        "valid_count": int(count),
        "frozen": "eval(), requires_grad=False, no_grad()",
        "shuffle": "deterministic cross-sample roll by one batch row",
    }


def run(args: argparse.Namespace) -> None:
    if not args.device.startswith("cuda"):
        raise SystemExit("this evaluator requires CUDA because token extraction uses the Qwen CUDA autocast path")
    model = _load_frozen_model(args.checkpoint, args.device)
    dataset = DepthClipCacheDataset(root=args.clips, split=None)
    languages = _instructions(args.clips)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    totals: Dict[str, float] = {}

    for batch in loader:
        tokens = capture_action_tokens(model, _clip_examples(batch, languages), args.device)
        with torch.no_grad():
            states, deltas = build_metric_delta_targets(
                batch["depth"].to(device=args.device, dtype=torch.float32),
                tubelet_size=args.tubelet_size,
                delta_lag=args.delta_lag,
                grid=tuple(args.grid),
                target_type=args.target_type,
            )
        result = condition_error_sums(model.depth_delta_head, states.values[:, :-1], tokens, deltas.values, deltas.mask)
        for name, value in result.items():
            totals[name] = totals.get(name, 0.0) + float(value.item())

    report = _report(totals, args.checkpoint, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid", type=int, nargs=2, default=[16, 16])
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--delta-lag", type=int, default=1)
    parser.add_argument("--target-type", default="sim_metric")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
