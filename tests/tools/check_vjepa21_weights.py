"""Mapping-free authenticity check for the vendored V-JEPA 2.1 weights.

Meta publishes V-JEPA 2.1 only as a native training checkpoint
(`vjepa2_1_vitl_dist_vitG_384.pt`), while the weights we actually load come from a community
HuggingFace port. Nothing rules out a port having silently altered a tensor, so before trusting one
we check the ported `model.safetensors` against Meta's own file **without a name mapping**: every
tensor is reduced to `(shape, float64 sum, float64 sum of squares)` and the two multisets are
matched. Parameter names, key prefixes and tensor order are irrelevant to the verdict.

Two passes, because the ports legitimately restructure attention:

1. direct - exact multiset match on the statistic triple. Order-free and name-free.
2. fused - the ports split Meta's fused `qkv` projection into separate q/k/v tensors. Both the sum
   and the sum of squares are additive under concatenation, so a group of `g` port tensors can be
   verified against one official tensor without knowing which member is q, k or v. Only the grouping
   uses file order (per shape, see `group_for_fusion`); the match itself is still order-free. Pass 2
   needs a relative tolerance because summing three partial sums is not bitwise equal to summing the
   whole tensor at once.

Honest boundary: this proves the port's *weight tensors* were not modified. It is not a forward
equivalence proof - Meta's own code is never executed here, so a port could still differ in how it
wires those tensors. That caveat belongs in `docs/provenance/teachers.md` alongside the output.

Usage:
    python tests/tools/check_vjepa21_weights.py \
        --official /vepfs/wangshilong/models/dynaweave/vjepa21/vjepa2_1_vitl_dist_vitG_384.pt \
        --port /vepfs/wangshilong/models/dynaweave/vjepa21/port_dev_jahn \
        --port /vepfs/wangshilong/models/dynaweave/vjepa21/port_apiantonio \
        --markdown

Exits non-zero if any port section fails to match an official section.
"""

import argparse
import hashlib
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# The ports split one fused qkv projection into three tensors.
QKV_GROUP_SIZE = 3
# Pass 2 only: three partial float64 sums do not re-add bitwise to the whole-tensor sum.
FUSED_REL_TOL = 1e-9
# Official sections a port's `encoder.*` block could plausibly come from. `ema_encoder` is the
# EMA/target encoder, `encoder` the context encoder; which one a port shipped is a provenance fact
# the match itself establishes.
ENCODER_SECTIONS = ("ema_encoder", "encoder")
PREDICTOR_SECTIONS = ("predictor",)


@dataclass(frozen=True)
class TensorStat:
    """Name-free fingerprint of one tensor. Hashable, so multisets are plain Counters."""

    shape: Tuple[int, ...]
    total: float
    sq_total: float

    def merged_with(self, others: Iterable["TensorStat"]) -> "TensorStat":
        """Fingerprint of this tensor concatenated with `others` along axis 0."""
        group = [self, *others]
        rows = sum(stat.shape[0] for stat in group)
        return TensorStat(
            shape=(rows, *self.shape[1:]),
            total=math.fsum(stat.total for stat in group),
            sq_total=math.fsum(stat.sq_total for stat in group),
        )

    def close_to(self, other: "TensorStat", rel_tol: float) -> bool:
        return (
            self.shape == other.shape
            and math.isclose(self.total, other.total, rel_tol=rel_tol, abs_tol=rel_tol)
            and math.isclose(self.sq_total, other.sq_total, rel_tol=rel_tol, abs_tol=rel_tol)
        )


@dataclass
class MatchReport:
    """Outcome of comparing two named tensor collections."""

    left_label: str
    right_label: str
    left_count: int
    right_count: int
    direct: int
    fused: int
    unmatched_left: List[str]
    unmatched_right: List[str]

    @property
    def ok(self) -> bool:
        return not self.unmatched_left and not self.unmatched_right

    def summary(self) -> str:
        verdict = "MATCH" if self.ok else "MISMATCH"
        return (
            f"[{verdict}] {self.left_label} ({self.left_count}) vs {self.right_label} "
            f"({self.right_count}): {self.direct} direct, {self.fused} fused, "
            f"{len(self.unmatched_left)} + {len(self.unmatched_right)} unmatched"
        )


def tensor_stats(named_tensors) -> Dict[str, TensorStat]:
    """Reduce a state dict to fingerprints. Accumulates in float64 to stay dtype-independent."""
    import torch

    stats: Dict[str, TensorStat] = {}
    for name, tensor in named_tensors.items():
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            continue
        values = tensor.detach().to(torch.float64)
        stats[name] = TensorStat(
            shape=tuple(tensor.shape),
            total=float(values.sum()),
            sq_total=float((values * values).sum()),
        )
    return stats


def match_direct(left: Dict[str, TensorStat], right: Dict[str, TensorStat]) -> Tuple[int, List[str], List[str]]:
    """Exact multiset match. Returns `(pairs, unmatched_left, unmatched_right)`.

    Exactness is deliberate: identical float32 bytes reduced the same way give identical float64
    statistics, so pass 1 needs no tolerance and cannot accept a near-miss.
    """
    pool: Dict[TensorStat, List[str]] = {}
    for name, stat in right.items():
        pool.setdefault(stat, []).append(name)

    pairs = 0
    unmatched_left: List[str] = []
    for name, stat in left.items():
        candidates = pool.get(stat)
        if candidates:
            candidates.pop()
            pairs += 1
        else:
            unmatched_left.append(name)
    unmatched_right = [name for names in pool.values() for name in names]
    return pairs, unmatched_left, sorted(unmatched_right)


def group_for_fusion(stats: Sequence[Tuple[str, TensorStat]], group_size: int) -> List[Tuple[List[str], TensorStat]]:
    """Fuse each shape's residuals, `group_size` at a time in file order, into one fingerprint.

    Grouping is per shape rather than over strictly adjacent entries because HuggingFace ports
    interleave the weight and the bias of each projection (`query.weight`, `query.bias`,
    `key.weight`, ...), so layer `i`'s three q/k/v weights are adjacent only *within* their shape.
    File order inside a shape is what makes the grouping well defined; the subsequent match is still
    name-free and order-free. A trailing incomplete group is dropped, and any mis-grouping shows up
    as a failed sum rather than as a silent pass.
    """
    buckets: Dict[Tuple[int, ...], List[Tuple[str, TensorStat]]] = {}
    for name, stat in stats:
        buckets.setdefault(stat.shape, []).append((name, stat))

    groups: List[Tuple[List[str], TensorStat]] = []
    for entries in buckets.values():
        for start in range(0, len(entries) - group_size + 1, group_size):
            run = entries[start : start + group_size]
            groups.append(([name for name, _ in run], run[0][1].merged_with(stat for _, stat in run[1:])))
    return groups


def match_fused(
    left: List[Tuple[str, TensorStat]],
    right_groups: List[Tuple[List[str], TensorStat]],
    rel_tol: float = FUSED_REL_TOL,
) -> Tuple[int, List[str], List[str]]:
    """Match official fused tensors against grouped port tensors, within `rel_tol`.

    Returns `(pairs, unmatched_left, matched_right)`. The third element lists the port tensors that a
    matched group consumed, so the caller can derive the unmatched port side from its own residual
    list - port tensors that never formed a group must not disappear from the report.
    """
    available = list(right_groups)
    pairs = 0
    unmatched_left: List[str] = []
    matched_right: List[str] = []
    for name, stat in left:
        hit = next(
            (index for index, (_, group) in enumerate(available) if group.close_to(stat, rel_tol)),
            None,
        )
        if hit is None:
            unmatched_left.append(name)
        else:
            matched_right.extend(available.pop(hit)[0])
            pairs += 1
    return pairs, unmatched_left, matched_right


def compare(
    left: Dict[str, TensorStat],
    right: Dict[str, TensorStat],
    left_label: str,
    right_label: str,
    group_size: int = QKV_GROUP_SIZE,
    rel_tol: float = FUSED_REL_TOL,
) -> MatchReport:
    """Direct pass, then a fused pass over whatever the direct pass left over."""
    direct, left_rest, right_rest = match_direct(left, right)

    # Preserve file order in the residuals; grouping depends on it.
    left_names, right_names = set(left_rest), set(right_rest)
    left_residual = [(name, left[name]) for name in left if name in left_names]
    right_residual = [(name, right[name]) for name in right if name in right_names]

    fused, unmatched_left, matched_right = match_fused(
        left_residual, group_for_fusion(right_residual, group_size), rel_tol
    )
    consumed = set(matched_right)
    unmatched_right = [name for name, _ in right_residual if name not in consumed]
    return MatchReport(
        left_label=left_label,
        right_label=right_label,
        left_count=len(left),
        right_count=len(right),
        direct=direct,
        fused=fused,
        unmatched_left=unmatched_left,
        unmatched_right=unmatched_right,
    )


def sha256_of(path: Path, chunk_size: int = 8 << 20) -> str:
    """Mirrors `tools/verify_assets.py::sha256_of`; duplicated because the submodule must not
    import from the superproject."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def load_official(path: Path, section: str) -> Dict[str, TensorStat]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if section not in checkpoint:
        raise KeyError(f"{path} has no section {section!r}; sections are {sorted(checkpoint)}")
    return tensor_stats(checkpoint[section])


def load_port(path: Path, prefix: str) -> Dict[str, TensorStat]:
    from safetensors.torch import load_file

    weights = load_file(path)
    return tensor_stats({name: tensor for name, tensor in weights.items() if name.startswith(prefix)})


def best_section(
    port_stats: Dict[str, TensorStat],
    official_sections: Dict[str, Dict[str, TensorStat]],
    port_label: str,
) -> Tuple[MatchReport, List[MatchReport]]:
    """Compare a port block against every candidate official section, best match first."""
    reports = [
        compare(official, port_stats, f"official/{name}", port_label) for name, official in official_sections.items()
    ]
    reports.sort(key=lambda report: (not report.ok, len(report.unmatched_left)))
    return reports[0], reports[1:]


def render_markdown(rows: List[Tuple[str, str, str, str]]) -> str:
    header = "| File | Bytes | sha256 | Verdict |\n|---|---:|---|---|\n"
    return header + "".join(f"| `{name}` | {size} | `{digest}` | {verdict} |\n" for name, size, digest, verdict in rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--official", type=Path, required=True, help="Meta's native .pt checkpoint")
    parser.add_argument("--port", type=Path, action="append", required=True, help="port directory (repeatable)")
    parser.add_argument("--group-size", type=int, default=QKV_GROUP_SIZE)
    parser.add_argument("--rel-tol", type=float, default=FUSED_REL_TOL)
    parser.add_argument("--sha256", action="store_true", help="also hash the files (slow)")
    parser.add_argument("--markdown", action="store_true", help="print a provenance table")
    args = parser.parse_args(argv)

    official_encoders = {name: load_official(args.official, name) for name in ENCODER_SECTIONS}
    official_predictors = {name: load_official(args.official, name) for name in PREDICTOR_SECTIONS}

    failed = False
    rows: List[Tuple[str, str, str, str]] = []
    port_blocks: Dict[str, Dict[str, Dict[str, TensorStat]]] = {}

    for port_dir in args.port:
        weights_path = port_dir / "model.safetensors"
        label = port_dir.name
        blocks = {
            "encoder": load_port(weights_path, "encoder."),
            "predictor": load_port(weights_path, "predictor."),
        }
        port_blocks[label] = blocks

        verdicts = []
        for block, sections in (("encoder", official_encoders), ("predictor", official_predictors)):
            best, others = best_section(blocks[block], sections, f"{label}/{block}")
            print(best.summary())
            for report in others:
                print(
                    f"    also tried {report.right_label} vs {report.left_label}: "
                    f"{report.direct} direct, {report.fused} fused"
                )
            if best.unmatched_left or best.unmatched_right:
                failed = True
                print(f"    unmatched official: {best.unmatched_left[:5]}")
                print(f"    unmatched port: {best.unmatched_right[:5]}")
            verdicts.append(f"{block} = {best.left_label.split('/')[-1]}" if best.ok else f"{block} MISMATCH")

        rows.append(
            (
                str(weights_path),
                f"{weights_path.stat().st_size:,}",
                sha256_of(weights_path) if args.sha256 else "-",
                ", ".join(verdicts),
            )
        )

    for first, second in itertools.pairwise(sorted(port_blocks)):
        for block in ("encoder", "predictor"):
            # Independent ports of the same upstream weights must agree with no fusion involved.
            report = compare(
                port_blocks[first][block],
                port_blocks[second][block],
                f"{first}/{block}",
                f"{second}/{block}",
                group_size=args.group_size,
                rel_tol=args.rel_tol,
            )
            print(report.summary())
            failed = failed or not report.ok

    rows.insert(
        0,
        (
            str(args.official),
            f"{args.official.stat().st_size:,}",
            sha256_of(args.official) if args.sha256 else "-",
            "Meta native checkpoint (reference)",
        ),
    )
    if args.markdown:
        print()
        print(render_markdown(rows))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
