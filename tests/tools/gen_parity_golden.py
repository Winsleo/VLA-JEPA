"""Generate the I2 parity golden file.

Run this on the revision that is being frozen as the reference (I2 commit 1, before any
interface change), then again on any later revision to compare by hand if a test fails:

    CUDA_VISIBLE_DEVICES=0 PATH=/vepfs/wangshilong/envs/dynaweave/bin:$PATH \
        python tests/tools/gen_parity_golden.py -o tests/data/i2_parity_baseline.json

Needs one visible GPU (~50 GB free for the short optimisation run) and the published LIBERO
checkpoint. tests/test_i2_parity.py consumes the output.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from parity_probe import PUBLISHED_LIBERO_CKPT, collect_probe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "tests" / "data" / "i2_parity_baseline.json",
        help="destination JSON path",
    )
    args = parser.parse_args()

    if not PUBLISHED_LIBERO_CKPT.exists():
        print(f"missing published checkpoint: {PUBLISHED_LIBERO_CKPT}", file=sys.stderr)
        return 1

    payload = collect_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    print(json.dumps(payload["forward"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
