"""Contract for the mapping-free V-JEPA 2.1 weight checker (I3 / S2a).

The checker is what lets us trust a community HuggingFace port of weights Meta only publishes as a
native `.pt`, so its own logic has to be tested rather than trusted: a check that silently accepts a
tampered tensor is worse than no check. The tests below pin what it must catch, what it must not
false-alarm on, and - explicitly - the one blind spot the method has, so that boundary is recorded in
code and not only in prose.

Synthetic state dicts only; no checkpoint files and no network.
Run:  pytest tests/test_i3_vjepa21_weight_check.py -v
"""

import hashlib
import sys
from pathlib import Path

import pytest
import torch

# Same path handling as `tests/tools/gen_parity_golden.py`: `tests/tools/` holds standalone scripts
# and is not a package, so it is put on the path explicitly rather than imported through `tests.`.
sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))

from check_vjepa21_weights import (
    FUSED_REL_TOL,
    QKV_GROUP_SIZE,
    compare,
    group_for_fusion,
    match_direct,
    match_fused,
    render_markdown,
    sha256_of,
    tensor_stats,
)

DIM, LAYERS = 8, 3


def official_like(seed=0, layers=LAYERS):
    """Meta's layout: one fused `qkv` weight and bias per block, plus a plain projection."""
    generator = torch.Generator().manual_seed(seed)
    weights = {}
    for layer in range(layers):
        weights[f"blocks.{layer}.attn.qkv.weight"] = torch.randn(3 * DIM, DIM, generator=generator, dtype=torch.float32)
        weights[f"blocks.{layer}.attn.qkv.bias"] = torch.randn(3 * DIM, generator=generator, dtype=torch.float32)
        weights[f"blocks.{layer}.attn.proj.weight"] = torch.randn(DIM, DIM, generator=generator, dtype=torch.float32)
    return weights


def port_like(official, layers=LAYERS):
    """A faithful port: the fused projections split into q/k/v, everything renamed."""
    ported = {}
    for layer in range(layers):
        fused_weight = official[f"blocks.{layer}.attn.qkv.weight"]
        fused_bias = official[f"blocks.{layer}.attn.qkv.bias"]
        for index, name in enumerate(("query", "key", "value")):
            rows = slice(index * DIM, (index + 1) * DIM)
            ported[f"layer.{layer}.attention.{name}.weight"] = fused_weight[rows].clone()
            ported[f"layer.{layer}.attention.{name}.bias"] = fused_bias[rows].clone()
        ported[f"layer.{layer}.attention.output.weight"] = official[f"blocks.{layer}.attn.proj.weight"].clone()
    return ported


@pytest.fixture
def official():
    return official_like()


@pytest.fixture
def port(official):
    return port_like(official)


class TestTensorStat:
    def test_statistics_are_additive_under_concatenation(self):
        """The property pass 2 rests on: split tensors re-add to the fused tensor's fingerprint."""
        whole = torch.randn(9, 4, generator=torch.Generator().manual_seed(1))
        stats = tensor_stats({"whole": whole})["whole"]
        pieces = [tensor_stats({"p": whole[i * 3 : (i + 1) * 3]})["p"] for i in range(3)]
        merged = pieces[0].merged_with(pieces[1:])
        assert merged.shape == stats.shape
        assert merged.close_to(stats, FUSED_REL_TOL)

    def test_reduction_is_dtype_independent(self):
        """A port may store bf16; equal *bytes* is not the claim, equal float64 statistics is."""
        values = torch.tensor([0.5, -0.25, 2.0], dtype=torch.float32)
        assert tensor_stats({"a": values})["a"] == tensor_stats({"a": values.double().float()})["a"]

    def test_non_float_entries_are_skipped(self):
        stats = tensor_stats({"w": torch.ones(2), "step": torch.tensor([3]), "note": "not a tensor"})
        assert set(stats) == {"w"}


class TestDirectMatch:
    def test_renaming_and_reordering_do_not_affect_the_verdict(self):
        left = tensor_stats(official_like())
        shuffled = {f"renamed/{name}": tensor for name, tensor in reversed(list(official_like().items()))}
        pairs, unmatched_left, unmatched_right = match_direct(left, tensor_stats(shuffled))
        assert (pairs, unmatched_left, unmatched_right) == (len(left), [], [])

    def test_a_perturbed_tensor_is_reported(self):
        tampered = official_like()
        tampered["blocks.1.attn.proj.weight"][0, 0] += 1e-3
        pairs, unmatched_left, unmatched_right = match_direct(tensor_stats(official_like()), tensor_stats(tampered))
        assert pairs == len(tampered) - 1
        assert unmatched_left == ["blocks.1.attn.proj.weight"]
        assert unmatched_right == ["blocks.1.attn.proj.weight"]

    def test_a_transposed_non_square_tensor_is_reported(self):
        """Both moments survive a transpose; the shape in the fingerprint is what catches it."""
        base = {"w": torch.randn(4, 6, generator=torch.Generator().manual_seed(2))}
        transposed = {"w": base["w"].t().contiguous()}
        pairs, unmatched_left, _ = match_direct(tensor_stats(base), tensor_stats(transposed))
        assert (pairs, unmatched_left) == (0, ["w"])

    def test_a_sign_flip_is_reported(self):
        """The sum alone would miss a symmetric change; the sum of squares would miss a sign flip."""
        base = {"w": torch.tensor([1.0, 2.0, 3.0])}
        flipped = {"w": torch.tensor([-1.0, -2.0, -3.0])}
        pairs, unmatched_left, _ = match_direct(tensor_stats(base), tensor_stats(flipped))
        assert (pairs, unmatched_left) == (0, ["w"])

    def test_a_permutation_of_equal_statistic_tensors_is_the_known_blind_spot(self):
        """Documented limitation: tensors with identical fingerprints are interchangeable.

        Multiset matching cannot see *which* name carries which tensor, so a rearrangement among
        tensors sharing a shape and both moments passes - as does any within-tensor permutation
        (here a transposed square matrix). That is exactly why the honest boundary in
        `docs/provenance/teachers.md` says this is evidence of unmodified weight *tensors*, not of
        correct wiring; only running Meta's own forward pass could rule the latter out.
        """
        swapped = {"a": torch.tensor([-1.0, 1.0]), "b": torch.tensor([1.0, -1.0])}
        pairs, _, _ = match_direct(tensor_stats({"a": swapped["b"], "b": swapped["a"]}), tensor_stats(swapped))
        assert pairs == 2

        square = torch.randn(4, 4, generator=torch.Generator().manual_seed(3))
        transposed, _, _ = match_direct(tensor_stats({"w": square}), tensor_stats({"w": square.t().contiguous()}))
        assert transposed == 1


class TestFusedMatch:
    def test_grouping_takes_runs_of_equal_shape_in_file_order(self):
        stats = tensor_stats({f"t{index}": torch.ones(2, 2) for index in range(6)})
        groups = group_for_fusion(list(stats.items()), QKV_GROUP_SIZE)
        assert [names for names, _ in groups] == [["t0", "t1", "t2"], ["t3", "t4", "t5"]]

    def test_an_incomplete_group_is_dropped_rather_than_guessed(self):
        stats = tensor_stats({"a": torch.ones(2), "b": torch.ones(2), "c": torch.ones(3)})
        assert group_for_fusion(list(stats.items()), QKV_GROUP_SIZE) == []

    def test_interleaved_shapes_do_not_break_a_group(self):
        """Ports interleave `query.weight`, `query.bias`, `key.weight`, ... so grouping is per shape."""
        tensors = {"a": torch.ones(2), "b": torch.ones(3), "c": torch.ones(2), "d": torch.ones(2), "e": torch.ones(2)}
        groups = group_for_fusion(list(tensor_stats(tensors).items()), QKV_GROUP_SIZE)
        assert [names for names, _ in groups] == [["a", "c", "d"]], "trailing 'e' has no group of its own"

    def test_a_group_whose_members_do_not_add_up_is_reported(self):
        official = official_like(layers=1)
        broken = port_like(official, layers=1)
        broken["layer.0.attention.key.weight"] = broken["layer.0.attention.key.weight"] * 1.01
        report = compare(tensor_stats(official), tensor_stats(broken), "official", "broken")
        assert not report.ok
        assert report.unmatched_left == ["blocks.0.attn.qkv.weight"]

    def test_the_fused_pass_never_reuses_a_group(self):
        stat = tensor_stats({"w": torch.ones(6, 2)})["w"]
        parts = tensor_stats({"p": torch.ones(2, 2)})["p"]
        groups = [(["a", "b", "c"], parts.merged_with([parts, parts]))]
        pairs, unmatched_left, matched_right = match_fused([("x", stat), ("y", stat)], groups)
        assert (pairs, unmatched_left, matched_right) == (1, ["y"], ["a", "b", "c"])

    def test_a_port_tensor_that_never_formed_a_group_is_still_reported(self):
        """An ungrouped leftover must not vanish between the two passes."""
        official = official_like(layers=1)
        port = port_like(official, layers=1)
        port["layer.0.attention.extra"] = torch.ones(5)
        report = compare(tensor_stats(official), tensor_stats(port), "official", "port")
        assert report.unmatched_right == ["layer.0.attention.extra"]


class TestEndToEndCompare:
    def test_a_faithful_port_matches_completely(self, official, port):
        report = compare(tensor_stats(official), tensor_stats(port), "official", "port")
        assert report.ok
        assert (report.direct, report.fused) == (LAYERS, 2 * LAYERS)
        assert "MATCH" in report.summary()

    def test_the_two_passes_are_reported_apart(self, official, port):
        """Fusion matches are weaker evidence than direct ones, so they must stay countable."""
        report = compare(tensor_stats(official), tensor_stats(port), "official", "port")
        assert report.direct == LAYERS, "only the plain projections match directly"
        assert report.fused == 2 * LAYERS, "one fused weight and one fused bias per layer"

    def test_a_port_with_an_extra_tensor_is_not_silently_accepted(self, official, port):
        port["layer.0.attention.extra"] = torch.ones(5)
        report = compare(tensor_stats(official), tensor_stats(port), "official", "port")
        assert not report.ok
        assert report.unmatched_right == ["layer.0.attention.extra"]

    def test_a_port_missing_a_tensor_is_not_silently_accepted(self, official, port):
        del port["layer.2.attention.output.weight"]
        report = compare(tensor_stats(official), tensor_stats(port), "official", "port")
        assert not report.ok
        assert report.unmatched_left == ["blocks.2.attn.proj.weight"]

    def test_comparing_two_faithful_ports_needs_no_fusion_pass(self, official):
        report = compare(tensor_stats(port_like(official)), tensor_stats(port_like(official)), "port_a", "port_b")
        assert report.ok and report.fused == 0

    def test_a_wrong_section_is_a_clear_mismatch(self, official, port):
        """Standing in for `encoder` vs `ema_encoder`: the wrong section must not half-match."""
        other = compare(tensor_stats(official_like(seed=99)), tensor_stats(port), "official/other", "port")
        assert not other.ok
        assert len(other.unmatched_left) == LAYERS * 3


class TestProvenanceOutput:
    def test_sha256_matches_hashlib(self, tmp_path):
        blob = tmp_path / "weights.bin"
        payload = bytes(range(256)) * 4096
        blob.write_bytes(payload)
        assert sha256_of(blob, chunk_size=1024) == hashlib.sha256(payload).hexdigest()

    def test_markdown_is_a_table_with_one_row_per_file(self):
        table = render_markdown([("a.pt", "1", "deadbeef", "reference"), ("b.safetensors", "2", "cafe", "encoder")])
        lines = table.strip().splitlines()
        assert lines[0].startswith("| File |") and lines[1].startswith("|---")
        assert len(lines) == 4
        assert "`b.safetensors`" in lines[3]
