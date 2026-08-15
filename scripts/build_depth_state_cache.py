"""Build predictor-aligned log-depth states/deltas from an I4 pseudo-depth cache."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.model.modules.world_model.depth_targets import build_metric_delta_targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="estimator directory containing episode NPZ files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=int, nargs=2, default=(16, 16))
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--delta-lag", type=int, default=1)
    parser.add_argument("--target-type", default="pseudo_metric")
    args = parser.parse_args()
    if (args.tubelet_size, args.delta_lag) != (2, 1):
        raise SystemExit("I4 training cache is fixed to --tubelet-size 2 --delta-lag 1")

    files = sorted(args.input.rglob("episode_*.npz"))
    if not files:
        raise SystemExit(f"no episode NPZ files under {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    written = 0
    state_shape = delta_shape = None
    for source in files:
        relative = source.relative_to(args.input)
        destination = args.output / relative
        if destination.exists():
            written += 1
            continue
        with np.load(source) as payload:
            raw = np.asarray(payload["depth_m"], dtype=np.float32)
            # Episode lengths are not required to be even.  The trainer only consumes 8-frame
            # windows, so the final unpaired frame cannot form a tubelet state and is discarded.
            raw = raw[: raw.shape[0] - (raw.shape[0] % args.tubelet_size)]
            depth = torch.from_numpy(raw).permute(1, 0, 2, 3)[None, :, :, None]
        with torch.no_grad():
            states, deltas = build_metric_delta_targets(
                depth,
                tubelet_size=args.tubelet_size,
                delta_lag=args.delta_lag,
                grid=tuple(args.grid),
                target_type=args.target_type,
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(f".partial-{written}.npz")
        np.savez_compressed(
            partial,
            states=states.values.squeeze(0).numpy().astype(np.float16),
            state_mask=states.mask.squeeze(0).numpy(),
            deltas=deltas.values.squeeze(0).numpy().astype(np.float16),
            delta_mask=deltas.mask.squeeze(0).numpy(),
        )
        partial.replace(destination)
        written += 1
        state_shape = list(states.values.shape[1:])
        delta_shape = list(deltas.values.shape[1:])
        if written % 100 == 0:
            print(f"{written}/{len(files)}", flush=True)

    (args.output / "index.json").write_text(json.dumps({
        "complete": written == len(files),
        "num_files": len(files),
        "num_written": written,
        "input": str(args.input),
        "target_type": args.target_type,
        "tubelet_size": args.tubelet_size,
        "delta_lag": args.delta_lag,
        "grid": list(args.grid),
        "state_shape": state_shape,
        "delta_shape": delta_shape,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
