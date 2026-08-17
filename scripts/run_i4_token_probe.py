"""Probe a trained I4 checkpoint's action tokens for action-conditioned depth change.

Why this exists: the I4 policy contrasts all sit inside a 2.59 pp three-seed floor, so rollout
success cannot say whether depth supervision changed the model. This asks the first link of the
causal chain instead -- did the representation the policy reads become more geometric -- on the one
tensor that both the depth loss and the policy touch (`VLA_JEPA.forward` hands it to
`depth_delta_head` as `condition`; `predict_action` reads the same Qwen last-hidden layer).

Stages, each resumable and each writing its own provenance:

    extract  one frozen forward per clip per checkpoint -> [rows, transitions, tokens, hidden]
    fit      one linear readout per (checkpoint, seed), selected on val, reported on test
    report   per-arm means, the feature-free reference, and the seed noise floor

`action_tokens` is a local inside `VLA_JEPA.forward`, so `extract` captures the Qwen last-hidden
layer with a hook and re-derives the same slice from `action_token_ids`. That shortcut is only
trustworthy if it agrees with `forward`; `tests/test_i4_token_probe_extraction.py` pins the
agreement rather than assuming it.

The clip cache carries no language, so the instruction is recovered from each row's
`suite`/`task_id` through the mapping exported from LIBERO into `task_language.json`. A wrong
instruction would change what the action tokens encode, so this is read, never guessed.
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from starVLA.dataloader.depth_cache_dataset import DepthClipCacheDataset
from starVLA.model.modules.world_model.depth_targets import build_metric_delta_targets
from starVLA.probes import token_probe

FEATURE_DTYPE = np.float16
INDEX_FILE = "index.json"


# --------------------------------------------------------------------------------------
# stage: extract
# --------------------------------------------------------------------------------------

def _instructions(root: Path) -> Dict[str, Dict[str, str]]:
    path = root / "task_language.json"
    if not path.is_file():
        raise SystemExit(
            f"missing {path}. Export it from the LIBERO env, which owns the task table:\n"
            "  LIBERO_CONFIG_PATH=<libero>/libero <libero-python> -c \"...get_task(i).language...\""
        )
    return json.loads(path.read_text())


def _load_frozen_model(checkpoint: Path, device: str):
    """The trained framework, frozen: no grad, eval mode, and asserted so before any forward."""
    from starVLA.model.framework.base_framework import baseframework

    model = baseframework.from_pretrained(checkpoint.as_posix())
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("the probed checkpoint must be fully frozen")
    if model.training:
        raise RuntimeError("the probed checkpoint must be in eval mode")
    return model


def capture_action_tokens(model, examples: List[dict], device: str) -> torch.Tensor:
    """`[B, transitions, tokens, hidden]` Qwen states at the `<|action_i|>` positions.

    Grouped by transition with the same expression `forward` uses before handing the tensor to
    `depth_delta_head`, so a row of this cache is the condition of one predictor transition rather
    than an undifferentiated token run.

    Mirrors `VLA_JEPA.forward`: build the same prompt, take the last hidden layer, and gather the
    rows whose input id is an action token. Nothing else in the framework runs, so no depth target
    or future frame can reach the captured tensor.

    `forward` passes `prompt_template=CoT_prompt` and `predict_action` does not, so the two build
    different prompts upstream. This matches `forward`, because the depth gradient flowed into the
    tokens that `forward` produced; probing the `predict_action` prompt would ask a different
    question about a tensor the depth loss never shaped.
    """
    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=[example["image"] for example in examples],
        instructions=[example["instruction"] for example in examples],
        prompt_replace_dict={"{actions}": model.replace_prompt, "{e_actions}": model.embodied_replace_prompt},
        prompt_template=model.config.datasets.vla_data.get("CoT_prompt", ""),
    )
    ids = qwen_inputs["input_ids"]
    action_indices = torch.isin(ids, torch.tensor(model.action_token_ids, device=ids.device)).nonzero(as_tuple=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.qwen_vl_interface(
            **qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True
        )
        last_hidden = outputs.hidden_states[-1]
        batch, _, hidden = last_hidden.shape
        tokens = last_hidden[action_indices[0], action_indices[1], :].view(batch, -1, hidden)
    groups = model.vj_backbone.num_temporal_blocks - 1
    if tokens.shape[1] % groups:
        raise RuntimeError(f"action token count {tokens.shape[1]} is not divisible by {groups} transitions")
    return tokens.view(batch, groups, -1, hidden).to(torch.float32)


def _clip_examples(batch: dict, languages: Dict[str, Dict[str, str]]) -> List[dict]:
    """One framework example per clip: the first frame of each view plus the task instruction."""
    from PIL import Image

    video = batch["video"]  # [B, V, T, 3, H, W] uint8
    examples = []
    for row in range(video.shape[0]):
        suite = batch["suite"][row]
        task_id = str(int(batch["task_id"][row]))
        if suite not in languages or task_id not in languages[suite]:
            raise SystemExit(f"no instruction recorded for {suite}/task {task_id}")
        frames = [Image.fromarray(video[row, view, 0].permute(1, 2, 0).numpy()) for view in range(video.shape[1])]
        examples.append({"image": frames, "instruction": languages[suite][task_id]})
    return examples


def run_extract(args: argparse.Namespace) -> None:
    dataset = DepthClipCacheDataset(root=args.clips, split=None)
    languages = _instructions(args.clips)
    rows = len(dataset.rows)
    directory = args.out / args.name
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / INDEX_FILE).exists() and not args.overwrite:
        print(f"{args.name}: complete cache at {directory}, skipping")
        return

    model = _load_frozen_model(args.checkpoint, args.device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    array: Optional[np.memmap] = None
    written, started = 0, time.time()
    for batch in loader:
        tokens = capture_action_tokens(model, _clip_examples(batch, languages), args.device)
        block = tokens.cpu().numpy()
        if array is None:
            array = np.lib.format.open_memmap(
                directory / "features.npy", mode="w+", dtype=FEATURE_DTYPE, shape=(rows, *block.shape[1:])
            )
        array[written : written + block.shape[0]] = block.astype(FEATURE_DTYPE)
        written += block.shape[0]
        if written % (args.batch_size * 20) == 0:
            print(f"  {written}/{rows} clips ({written / (time.time() - started):.1f}/s)", flush=True)
    if array is None:
        raise SystemExit("the clip cache yielded no rows")
    array.flush()

    (directory / INDEX_FILE).write_text(json.dumps({
        "name": args.name,
        "checkpoint": args.checkpoint.as_posix(),
        "clips": args.clips.as_posix(),
        "num_rows": rows,
        "shape": list(array.shape),
        "dtype": np.dtype(FEATURE_DTYPE).name,
        "tensor": "Qwen last-hidden states at <|action_i|> positions, the depth head's condition",
        "frozen": "no_grad, eval(), requires_grad False on every parameter",
        "paths": [row["path"] for row in dataset.rows],
        "splits": [row["split"] for row in dataset.rows],
        "suites": [row["suite"] for row in dataset.rows],
        "complete": written == rows,
    }, indent=1, sort_keys=True) + "\n")
    print(f"{args.name}: wrote {written}/{rows} rows -> {directory}")


# --------------------------------------------------------------------------------------
# stage: targets
# --------------------------------------------------------------------------------------

def run_targets(args: argparse.Namespace) -> None:
    """Depth-delta targets on the probe grid, identical for every arm so only features differ."""
    dataset = DepthClipCacheDataset(root=args.clips, split=None)
    directory = args.out / f"targets_{args.grid[0]}x{args.grid[1]}_lag{args.delta_lag}"
    directory.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    values: Optional[np.memmap] = None
    masks: Optional[np.memmap] = None
    written = 0
    for batch in loader:
        with torch.no_grad():
            _, deltas = build_metric_delta_targets(
                batch["depth"].to(torch.float32),
                tubelet_size=args.tubelet_size,
                delta_lag=args.delta_lag,
                grid=tuple(args.grid),
                target_type=args.target_type,
            )
        block, mask = deltas.values.numpy(), deltas.mask.numpy()
        if values is None:
            rows = len(dataset.rows)
            values = np.lib.format.open_memmap(
                directory / "deltas.npy", mode="w+", dtype=np.float16, shape=(rows, *block.shape[1:])
            )
            masks = np.lib.format.open_memmap(
                directory / "delta_mask.npy", mode="w+", dtype=bool, shape=(rows, *mask.shape[1:])
            )
        values[written : written + block.shape[0]] = block.astype(np.float16)
        masks[written : written + block.shape[0]] = mask
        written += block.shape[0]
    if values is None:
        raise SystemExit("the clip cache yielded no rows")
    values.flush()
    masks.flush()
    (directory / INDEX_FILE).write_text(json.dumps({
        "num_rows": written,
        "grid": list(args.grid),
        "tubelet_size": args.tubelet_size,
        "delta_lag": args.delta_lag,
        "target_type": args.target_type,
        "shape": list(values.shape),
        "complete": written == len(dataset.rows),
    }, indent=1, sort_keys=True) + "\n")
    print(f"targets: wrote {written} rows -> {directory}")


# --------------------------------------------------------------------------------------
# stage: fit
# --------------------------------------------------------------------------------------

def _split_rows(index: Dict, split: str) -> np.ndarray:
    return np.array([i for i, value in enumerate(index["splits"]) if value == split], dtype=np.int64)


def _inputs(features: np.ndarray, deltas: np.ndarray, mask: np.ndarray, rows: np.ndarray, device: str):
    return token_probe.TokenProbeInputs(
        features=torch.from_numpy(np.ascontiguousarray(features[rows])).to(device),
        deltas=torch.from_numpy(np.ascontiguousarray(deltas[rows])).to(device=device, dtype=torch.float32),
        deltas_mask=torch.from_numpy(np.ascontiguousarray(mask[rows])).to(device),
    )


def run_fit(args: argparse.Namespace) -> None:
    targets = args.out / f"targets_{args.grid[0]}x{args.grid[1]}_lag{args.delta_lag}"
    deltas = np.load(targets / "deltas.npy", mmap_mode="r")
    mask = np.load(targets / "delta_mask.npy", mmap_mode="r")

    report: Dict[str, object] = {"arms": {}, "grid": list(args.grid), "delta_lag": args.delta_lag}
    for name in args.arms:
        directory = args.out / name
        index = json.loads((directory / INDEX_FILE).read_text())
        features = np.load(directory / "features.npy", mmap_mode="r")
        splits = {s: _split_rows(index, s) for s in ("train", "val", "test")}
        data = {s: _inputs(features, deltas, mask, rows, args.device) for s, rows in splits.items()}

        transitions, tokens, hidden = features.shape[1:]
        if transitions != deltas.shape[1]:
            raise SystemExit(f"{name}: {transitions} token transitions vs {deltas.shape[1]} target transitions")

        constant = token_probe.constant_baseline(data["train"]).unsqueeze(0)
        from starVLA.probes.geo_probe import masked_l1

        reference = float(masked_l1(constant.expand_as(data["test"].deltas), data["test"].deltas, data["test"].deltas_mask))

        scores = []
        for seed in args.seeds:
            best = token_probe.fit(
                data["train"], data["val"], tokens=tokens, hidden=hidden, num_views=deltas.shape[2],
                grid=tuple(args.grid), seed=seed, device=args.device,
            )
            head = token_probe.TokenReadout(tokens, hidden, deltas.shape[2], tuple(args.grid)).to(args.device)
            head.load_state_dict(best["state_dict"])
            test = token_probe.evaluate(head, data["test"])
            scores.append(test)
            print(f"  {name} seed{seed}: val {best['val']:.5f} test {test:.5f} (lr {best['lr']}, epoch {best['epoch']})")
        report["arms"][name] = {
            "test_masked_l1": scores,
            "mean": statistics.mean(scores),
            "sd": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "constant_baseline_test": reference,
            "checkpoint": index["checkpoint"],
        }

    (args.out / "probe_report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report["arms"], indent=1, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="probe cache root")
    parser.add_argument("--clips", type=Path, default=Path("/vepfs/wangshilong/data/dynaweave/i3_geo_clips"))
    parser.add_argument("--grid", type=int, nargs=2, default=[16, 16])
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--delta-lag", type=int, default=1)
    parser.add_argument("--target-type", default="pseudo_metric")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    sub = parser.add_subparsers(dest="stage", required=True)

    extract = sub.add_parser("extract", help="one frozen forward per clip for one checkpoint")
    extract.add_argument("--name", required=True, help="arm name, e.g. A_seed42")
    extract.add_argument("--checkpoint", type=Path, required=True)
    extract.add_argument("--overwrite", action="store_true")
    extract.set_defaults(func=run_extract)

    targets = sub.add_parser("targets", help="depth-delta targets shared by every arm")
    targets.set_defaults(func=run_targets)

    fit = sub.add_parser("fit", help="linear readouts and the seed spread")
    fit.add_argument("--arms", nargs="+", required=True)
    fit.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    fit.set_defaults(func=run_fit)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.func(arguments)
