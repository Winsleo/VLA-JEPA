# Vendored V-JEPA 2.1 modeling code

## Upstream

| Field | Value |
|---|---|
| Source | HuggingFace repo [`Dev-Jahn/vjepa2.1-vitl-fpc64-384`](https://huggingface.co/Dev-Jahn/vjepa2.1-vitl-fpc64-384) |
| Revision | `712c91d96b561e658385aac4ec52942e5b7fcaa6` |
| Converter | [`github.com/Dev-Jahn/vjepa2-hf`](https://github.com/Dev-Jahn/vjepa2-hf) |
| Original model | Meta V-JEPA 2.1 ViT-L/16 @384, distilled from ViT-G (`vjepa2_1_vitl_dist_vitG_384.pt`) |
| Licence | Apache-2.0 (port code and conversion); original from [`facebookresearch/vjepa2`](https://github.com/facebookresearch/vjepa2) |

## Vendored files

Byte-identical copies of the upstream revision:

| File | Local sha256 | Upstream git blob id |
|---|---|---|
| `configuration_vjepa21.py` | `edbf49b33d9e11df441b28a884b3eb14dbbc35b61fdc24154f51e4d9ef019608` | `f5e3559218e767524fec6c42a9088aee57b83493` |
| `modeling_vjepa21.py` | `f5cf2a7c6c10ed2f0b6bccd66cb4952c8acb6a4ffd7ddb132100af5689e9da02` | `f45ebe83db17f403cff1e8ae5894d1a7a99d6859` |

The blob ids are what the HuggingFace API reports for these (non-LFS) files at the pinned revision;
the sha256 column is the local copy, so the pair lets either identity be re-checked independently.

`__init__.py` and this file are ours. **No other change was made to the upstream sources** - no
reformatting, no lint fixes, no renames. That is deliberate: an unmodified copy can be re-diffed
against upstream at any time, and the repository formatter must not touch it
(`AGENTS.md` section 8 already forbids batch-rewriting vendored/upstream code).

## Why vendored instead of `trust_remote_code=True`

The model definition is an experimental variable of I3: it decides what the teacher computes. With
remote code, a silent upstream edit would change what the probes measure with no diff and no version
bump. Vendoring pins it, makes it reviewable, and keeps the pinned environment free of a runtime
code-download path.

## Compatibility with the pinned environment (measured 2026-08-02)

The upstream `config.json` declares `transformers_version: 5.12.1`, but the code only needs symbols
that already exist in the pinned `transformers==4.57.0`
(`ACT2FN`, `BaseModelOutput`, `ImageClassifierOutput`, `ALL_ATTENTION_FUNCTIONS`, `PreTrainedModel`,
`Unpack`, `ModelOutput`, `TransformersKwargs`). Measured in `envs/dynaweave`
(transformers 4.57.0, torch 2.6.0+cu124):

```text
VJEPA21Model.from_pretrained(port_dev_jahn, config=VJEPA21Config.from_pretrained(port_dev_jahn))
  -> 327,654,016 parameters
  -> missing_keys 0, unexpected_keys 0, mismatched_keys 0, error_msgs 0
  -> get_vision_features present
```

So I3 does **not** need the pinned environment upgraded - which matters because an upgrade would
invalidate the I1/I2 parity conclusions.

## Config geometry (the one adapter-relevant difference)

| Key | V-JEPA 2 (`vjepa2-vitl-fpc64-256`) | V-JEPA 2.1 (this port) |
|---|---|---|
| `image_size` | `256` | **absent** |
| `crop_size` | `256` | `384` |
| `patch_size` / `tubelet_size` | `16` / `2` | `16` / `2` |
| `hidden_size` | `1024` | `1024` |
| token grid | 16x16 | 24x24 |

`image_size` being absent (not merely different) is why `vj_backbone_adapter.py` needs a geometry
fallback: read `image_size` first, fall back to `crop_size`, error if neither is present. That change
lands with the adapter work, not here.

## The second port

`apiantonio/vjepa2.1-vit-large-384` @ `d6cfdbdd818754f22eaa72e5320d97724765f099` publishes a
*byte-identical* `model.safetensors` (see `docs/provenance/teachers.md`) but its own, substantially
different modeling code (`modeling_vjepa21.py` 60,655 bytes vs 40,562 here). It is kept downloaded as
a second reading of how these weights should be wired - the axis the tensor check cannot cover - and
because it is the only one of the two that ships a video processor
(`video_processing_vjepa21.py` + `video_preprocessor_config.json`, short side and centre crop both
384, no antialias on resize). Dev-Jahn's code is the vendored one because it is the smaller surface
and comes with a published converter.

## Authenticity of the weights

The weights are **not** vendored; they stay outside the repository under
`/vepfs/wangshilong/models/dynaweave/vjepa21/`. `tests/tools/check_vjepa21_weights.py` verifies them
against Meta's own `.pt` without a name mapping. See `docs/provenance/teachers.md` for the result and
for the boundary of what that check does and does not prove.
