"""Cache dataset contract for I3 (S1b).

Covers the two things that can silently poison every depth target downstream:

* RGB / depth / valid must come out of the cache in the layout the adapter consumes, with the axis
  permutation actually applied (a wrong permute is shape-compatible when V == T);
* crop, flip and view reorder must use one shared parameter draw across all three modalities
  (AGENTS.md section 7), while photometric jitter stays RGB-only.

Alignment is asserted structurally rather than by re-deriving the sampled parameters: the fixture
encodes the column index into both RGB and depth, so any geometric transform that is applied
identically keeps `depth == rgb[channel 0]` true, whatever parameters were drawn.

No simulator or GPU needed; the fixture writes a miniature cache in the recorder's own layout.
Run:  pytest tests/test_i3_depth_cache_dataset.py -v
"""

import json

import numpy as np
import pytest
import torch

from starVLA.dataloader.depth_cache_dataset import (
    ClipAugmentation,
    DepthClipCacheDataset,
    load_manifest,
)

FRAMES, VIEWS, SIZE = 8, 2, 8
VIEW_OFFSET = 100.0  # keeps `column + 100 * view` inside uint8 so RGB can carry the same code


def _clip_arrays(episode_index):
    """A clip whose RGB channel 0, depth and valid mask all encode `column + 100 * view`.

    Encoding the column makes horizontal flips and crops observable, and the per-view offset makes a
    view reorder observable in depth as well as in RGB. The episode index only shifts channel 1, so
    clips stay distinguishable without disturbing the relation the alignment tests rely on.
    """
    columns = np.arange(SIZE, dtype=np.float32)
    column_grid = np.broadcast_to(columns, (FRAMES, VIEWS, SIZE, SIZE))
    view_offset = (VIEW_OFFSET * np.arange(VIEWS, dtype=np.float32)).reshape(1, VIEWS, 1, 1)
    code = column_grid + view_offset

    rgb = np.zeros((FRAMES, VIEWS, SIZE, SIZE, 3), dtype=np.uint8)
    rgb[..., 0] = code.astype(np.uint8)
    rgb[..., 1] = episode_index
    return {
        "rgb": rgb,
        "depth_m": code.astype(np.float32),
        "valid": code > 0,
        "extrinsics": np.zeros((FRAMES, VIEWS, 4, 4), dtype=np.float32),
        "actions": np.zeros((FRAMES, 7), dtype=np.float32),
    }


def _write_cache(root, suites=("libero_goal",), episodes_per_suite=3, clips_per_episode=2):
    """Miniature cache in `record_geo_clips.py`'s layout: one manifest per suite, meta per episode."""
    splits = ["train", "val", "test"]
    for suite in suites:
        rows = []
        for episode_index in range(episodes_per_suite):
            split = splits[min(episode_index, len(splits) - 1)]
            episode_dir = root / suite / "00_task" / f"ep{episode_index:02d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            (episode_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "target_type": "sim_metric",
                        "depth_units": "meter",
                        "z_near": 0.0106,
                        "z_far": 530.49,
                        "suite": suite,
                        "episode_index": episode_index,
                        "split": split,
                        "views": ["agentview", "robot0_eye_in_hand"],
                    }
                )
            )
            arrays = _clip_arrays(episode_index)
            for clip_index in range(clips_per_episode):
                clip_path = episode_dir / f"clip{clip_index:02d}.npz"
                np.savez_compressed(clip_path, **arrays)
                rows.append(
                    {
                        "path": str(clip_path.relative_to(root)),
                        "suite": suite,
                        "task_id": 0,
                        "episode_index": episode_index,
                        "clip_index": clip_index,
                        "start": clip_index * 4,
                        "split": split,
                        "target_type": "sim_metric",
                        "success": True,
                        "valid_fraction": 0.875,
                    }
                )
        manifest = root / f"manifest_{suite}.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return root


@pytest.fixture(scope="module")
def cache_root(tmp_path_factory):
    return _write_cache(tmp_path_factory.mktemp("geo_clips"), suites=("libero_goal", "libero_object"))


def alignment_holds(sample):
    """`depth == rgb[channel 0]` and `valid == depth > 0`, i.e. no modality was transformed alone."""
    depth_from_rgb = sample["video"][:, :, :1].to(torch.float32)
    return bool(torch.equal(sample["depth"], depth_from_rgb)) and bool(torch.equal(sample["valid"], sample["depth"] > 0))


class TestManifestLookup:
    def test_all_suites_are_collected(self, cache_root):
        assert len(load_manifest(cache_root)) == 2 * 3 * 2

    def test_split_filter(self, cache_root):
        assert {row["split"] for row in load_manifest(cache_root, split="val")} == {"val"}
        assert len(load_manifest(cache_root, split="val")) == 2 * 2

    def test_suite_filter(self, cache_root):
        rows = load_manifest(cache_root, suites=["libero_object"])
        assert {row["suite"] for row in rows} == {"libero_object"}

    def test_order_is_by_path_not_by_recorder_finish_order(self, cache_root):
        paths = [row["path"] for row in load_manifest(cache_root)]
        assert paths == sorted(paths)

    def test_an_empty_selection_is_an_error_not_an_empty_dataset(self, cache_root):
        with pytest.raises(ValueError, match="no clips under"):
            DepthClipCacheDataset(cache_root, suites=["libero_90"])


class TestSampleLayout:
    def test_documented_shapes_and_dtypes(self, cache_root):
        sample = DepthClipCacheDataset(cache_root)[0]
        assert sample["video"].shape == (VIEWS, FRAMES, 3, SIZE, SIZE)
        assert sample["depth"].shape == (VIEWS, FRAMES, 1, SIZE, SIZE)
        assert sample["valid"].shape == (VIEWS, FRAMES, 1, SIZE, SIZE)
        assert sample["video"].dtype is torch.uint8
        assert sample["depth"].dtype is torch.float32
        assert sample["valid"].dtype is torch.bool

    def test_the_axis_permutation_is_really_applied(self, cache_root):
        """A wrong permute stays shape-valid here (V and T both index leading axes of the cache)."""
        sample = DepthClipCacheDataset(cache_root)[0]
        columns = torch.arange(SIZE, dtype=torch.float32).expand(SIZE, SIZE)
        assert torch.equal(sample["depth"][0, 0, 0], columns)
        assert torch.equal(sample["depth"][1, 0, 0], columns + VIEW_OFFSET)
        assert torch.equal(sample["video"][0, 0, 0].to(torch.float32), columns)

    def test_rgb_and_depth_start_out_aligned(self, cache_root):
        assert alignment_holds(DepthClipCacheDataset(cache_root)[0])

    def test_metadata_comes_from_both_manifest_and_episode_meta(self, cache_root):
        sample = DepthClipCacheDataset(cache_root, split="test")[0]
        assert sample["target_type"] == "sim_metric"
        assert sample["depth_units"] == "meter"
        assert sample["split"] == "test"
        assert sample["z_far"] == pytest.approx(530.49)

    def test_episode_meta_is_cached_per_episode(self, cache_root):
        dataset = DepthClipCacheDataset(cache_root, split="train")
        for index in range(len(dataset)):
            dataset[index]
        assert len(dataset._episode_meta) == 2, "one meta.json per episode, not per clip"


class TestSynchronizedAugmentation:
    def test_default_is_a_no_op(self, cache_root):
        plain = DepthClipCacheDataset(cache_root)[0]
        seeded = DepthClipCacheDataset(cache_root, seed=1234)[0]
        assert ClipAugmentation().is_identity
        assert torch.equal(plain["video"], seeded["video"])
        assert torch.equal(plain["depth"], seeded["depth"])

    @pytest.mark.parametrize(
        "augmentation",
        [
            ClipAugmentation(horizontal_flip=True),
            ClipAugmentation(crop_size=5),
            ClipAugmentation(swap_views=True),
            ClipAugmentation(crop_size=6, horizontal_flip=True, swap_views=True, brightness=0.0),
        ],
        ids=["flip", "crop", "swap_views", "combined"],
    )
    def test_geometry_stays_aligned_across_modalities(self, cache_root, augmentation):
        dataset = DepthClipCacheDataset(cache_root, augmentation=augmentation, seed=17)
        for index in range(len(dataset)):
            assert alignment_holds(dataset[index]), f"modalities diverged at sample {index}"

    def test_flip_actually_fires_on_some_samples(self, cache_root):
        """Guard against the alignment test passing because nothing was ever transformed."""
        dataset = DepthClipCacheDataset(cache_root, augmentation=ClipAugmentation(horizontal_flip=True), seed=3)
        columns = torch.arange(SIZE, dtype=torch.float32).expand(SIZE, SIZE)
        flipped = [
            bool(torch.equal(dataset[index]["depth"][0, 0, 0], columns.flip(-1))) for index in range(len(dataset))
        ]
        assert any(flipped) and not all(flipped), "a p=0.5 flip should vary across samples"

    def test_crop_shrinks_every_modality_by_the_same_box(self, cache_root):
        sample = DepthClipCacheDataset(cache_root, augmentation=ClipAugmentation(crop_size=5), seed=2)[0]
        assert sample["video"].shape[-2:] == (5, 5)
        assert sample["depth"].shape[-2:] == (5, 5)
        assert sample["valid"].shape[-2:] == (5, 5)
        assert alignment_holds(sample)

    def test_swap_views_moves_both_modalities_together(self, cache_root):
        dataset = DepthClipCacheDataset(cache_root, augmentation=ClipAugmentation(swap_views=True), seed=5)
        swapped = []
        for index in range(len(dataset)):
            sample = dataset[index]
            assert alignment_holds(sample), "a view swap must move RGB, depth and valid together"
            # View 0 carries no offset unless the two views were exchanged.
            swapped.append(float(sample["depth"][0, 0, 0, 0, 0]) >= VIEW_OFFSET)
        assert any(swapped) and not all(swapped), "a p=0.5 swap should vary across samples"

    def test_a_crop_larger_than_the_frame_is_rejected(self, cache_root):
        dataset = DepthClipCacheDataset(cache_root, augmentation=ClipAugmentation(crop_size=SIZE + 1))
        with pytest.raises(ValueError, match="exceeds"):
            dataset[0]

    def test_brightness_touches_rgb_only(self, cache_root):
        plain = DepthClipCacheDataset(cache_root)[0]
        jittered = DepthClipCacheDataset(cache_root, augmentation=ClipAugmentation(brightness=0.5), seed=11)[0]
        assert not torch.equal(plain["video"], jittered["video"])
        assert torch.equal(plain["depth"], jittered["depth"])
        assert torch.equal(plain["valid"], jittered["valid"])


class TestDeterminism:
    def test_the_same_index_reads_bitwise_identically(self, cache_root):
        """Gate condition b on the IO side: probe inputs are fixed given the cache and the seed."""
        augmentation = ClipAugmentation(crop_size=6, horizontal_flip=True, brightness=0.3)
        first = DepthClipCacheDataset(cache_root, augmentation=augmentation, seed=7)
        second = DepthClipCacheDataset(cache_root, augmentation=augmentation, seed=7)
        for index in range(len(first)):
            for key in ("video", "depth", "valid"):
                assert torch.equal(first[index][key], second[index][key])

    def test_augmentation_does_not_depend_on_iteration_order(self, cache_root):
        augmentation = ClipAugmentation(crop_size=6, horizontal_flip=True)
        dataset = DepthClipCacheDataset(cache_root, augmentation=augmentation, seed=7)
        forward = [dataset[index]["depth"] for index in range(len(dataset))]
        backward = [dataset[index]["depth"] for index in reversed(range(len(dataset)))]
        for expected, actual in zip(forward, reversed(backward), strict=True):
            assert torch.equal(expected, actual)

    def test_a_different_seed_gives_a_different_draw(self, cache_root):
        augmentation = ClipAugmentation(crop_size=5)
        samples = [
            [
                DepthClipCacheDataset(cache_root, augmentation=augmentation, seed=seed)[index]["depth"]
                for index in range(4)
            ]
            for seed in (0, 100)
        ]
        assert any(not torch.equal(left, right) for left, right in zip(*samples, strict=True))
