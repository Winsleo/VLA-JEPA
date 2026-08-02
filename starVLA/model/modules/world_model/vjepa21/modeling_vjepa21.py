"""V-JEPA 2.1 model implementation for HuggingFace Transformers.

Key differences from V-JEPA 2:
- Multi-modality: separate patch embedding for images (tubelet_size=1) + modality embeddings
- Hierarchical output: intermediate layer features with per-layer norms
- Interpolatable RoPE: variable input resolution support
- Dense predictor: hierarchical input fusion + context token prediction
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, ImageClassifierOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import ModelOutput, TransformersKwargs, logging

from .configuration_vjepa21 import VJEPA21Config

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VJEPA21EncoderOutput(ModelOutput):
    """Encoder output with optional hierarchical features.

    Attributes:
        last_hidden_state: Final layer output `(B, N, hidden_size)`.
        hierarchical_hidden_state: Concatenated intermediate features
            `(B, N, n_distillation_layers * hidden_size)` when requested.
        hidden_states: All layer outputs when `output_hidden_states=True`.
        attentions: All attention weights when `output_attentions=True`.
    """

    last_hidden_state: torch.FloatTensor
    hierarchical_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class VJEPA21PredictorOutput(ModelOutput):
    """Predictor output.

    Attributes:
        last_hidden_state: Predicted target tokens `(B, N_target, proj_dim)`.
        context_hidden_state: Predicted context tokens when `return_all_tokens=True`.
        hidden_states: All predictor layer outputs.
        attentions: All predictor attention weights.
    """

    last_hidden_state: torch.FloatTensor
    context_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class VJEPA21ModelOutput(ModelOutput):
    """Full model output combining encoder and predictor.

    Attributes:
        last_hidden_state: Encoder output `(B, N, hidden_size)`.
        hierarchical_hidden_state: Hierarchical encoder features.
        masked_hidden_state: Masked encoder output (context tokens only).
        predictor_output: Predictor output when not skipped.
        hidden_states: Encoder hidden states.
        attentions: Encoder attention weights.
    """

    last_hidden_state: torch.FloatTensor
    hierarchical_hidden_state: torch.FloatTensor | None = None
    masked_hidden_state: torch.FloatTensor | None = None
    predictor_output: VJEPA21PredictorOutput | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def apply_masks(tensor: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    """Gather tokens at mask indices.

    Args:
        tensor: `(B, N, D)` tensor.
        masks: List of `(B, K)` index tensors.
    Returns:
        `(len(masks)*B, K, D)` gathered tensor.
    """
    parts = []
    for mask in masks:
        mask = mask.to(tensor.device)
        idx = mask.unsqueeze(-1).expand(-1, -1, tensor.size(-1))
        parts.append(torch.gather(tensor, dim=1, index=idx))
    return torch.cat(parts, dim=0)


def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    return x.div(keep) * mask


class VJEPA21DropPath(nn.Module):
    def __init__(self, p: float | None = None):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.p, self.training)


def rotate_queries_or_keys(
    x: torch.Tensor,
    pos: torch.Tensor,
    n_registers: int = 0,
    has_cls_first: bool = False,
) -> torch.Tensor:
    """Apply rotary position embeddings with register/CLS token handling.

    Args:
        x: `(B, H, N, D)` query or key tensor.
        pos: Position ids broadcastable to `(..., N_ctx)`.
        n_registers: Number of register tokens at end of sequence (not rotated).
        has_cls_first: Whether first token is CLS (not rotated).
    """
    B, num_heads, N, D = x.size()

    n_cls = 1 if has_cls_first else 0
    start_ctx = n_cls
    end_ctx = N - n_registers

    x_cls = x[..., :n_cls, :] if n_cls else None
    x_ctx = x[..., start_ctx:end_ctx, :]
    x_reg = x[..., end_ctx:, :] if n_registers > 0 else None

    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega = omega / (D / 2.0)
    omega = 1.0 / (10000.0**omega)
    freq = torch.einsum("..., f -> ... f", pos, omega)

    emb_sin = freq.sin().repeat_interleave(2, dim=-1)
    emb_cos = freq.cos().repeat_interleave(2, dim=-1)

    y = x_ctx.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)

    out_ctx = x_ctx * emb_cos + y * emb_sin

    parts = []
    if x_cls is not None:
        parts.append(x_cls)
    parts.append(out_ctx)
    if x_reg is not None:
        parts.append(x_reg)
    return torch.cat(parts, dim=-2)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value).transpose(1, 2).contiguous()
    return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Patch Embeddings
# ---------------------------------------------------------------------------


class VJEPA21PatchEmbeddings3D(nn.Module):
    """3D patch embedding via Conv3d."""

    def __init__(self, config: VJEPA21Config, tubelet_size: int | None = None):
        super().__init__()
        ts = tubelet_size if tubelet_size is not None else config.tubelet_size
        ps = config.patch_size
        self.proj = nn.Conv3d(
            config.in_chans,
            config.hidden_size,
            kernel_size=(ts, ps, ps),
            stride=(ts, ps, ps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        return self.proj(x).flatten(2).transpose(1, 2)


class VJEPA21Embeddings(nn.Module):
    """Patch embeddings with modality-aware processing."""

    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.config = config
        # Video patch embedding (tubelet_size from config)
        self.patch_embeddings = VJEPA21PatchEmbeddings3D(config)
        # Image patch embedding (tubelet_size=1) if img_temporal_dim_size is set
        self.patch_embeddings_img = None
        if config.img_temporal_dim_size is not None:
            self.patch_embeddings_img = VJEPA21PatchEmbeddings3D(config, tubelet_size=1)

        # Modality embeddings
        self.img_mod_embed = None
        self.video_mod_embed = None
        if config.modality_embedding:
            self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(
        self, pixel_values_videos: torch.Tensor
    ) -> tuple[torch.Tensor, str]:
        """
        Args:
            pixel_values_videos: channels-first `(B, C, T, H, W)` video
                (or `(B, C, H, W)` image). This is the required layout — it maps
                directly onto the Conv3d/Conv2d patch embedder. Permute upstream
                if your data is channels-last.
        Returns:
            embeddings: `(B, N, hidden_size)`.
            mode: "img" or "video".
        """
        target_dtype = self.patch_embeddings.proj.weight.dtype
        pixel_values_videos = pixel_values_videos.to(dtype=target_dtype)

        T = pixel_values_videos.shape[2]

        # Determine if this is an image input
        is_image = (
            self.config.img_temporal_dim_size is not None
            and T == self.config.img_temporal_dim_size
        )

        if is_image and self.patch_embeddings_img is not None:
            embeddings = self.patch_embeddings_img(pixel_values_videos)
            mode = "img"
        else:
            # Ensure at least tubelet_size frames
            if T < self.config.tubelet_size:
                pixel_values_videos = pixel_values_videos.repeat(
                    1, 1, self.config.tubelet_size, 1, 1
                )
            embeddings = self.patch_embeddings(pixel_values_videos)
            mode = "video"

        # Add modality embedding
        if self.img_mod_embed is not None:
            if mode == "img":
                embeddings = embeddings + self.img_mod_embed
            else:
                embeddings = embeddings + self.video_mod_embed

        return embeddings, mode


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class VJEPA21RopeAttention(nn.Module):
    """RoPE-based multi-head attention with interpolation and register support."""

    def __init__(
        self,
        config: VJEPA21Config,
        hidden_size: int,
        num_attention_heads: int,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)

        self.dropout_prob = config.attention_probs_dropout_prob
        self.scaling = self.attention_head_size**-0.5
        self.is_causal = False

        # RoPE dimension split: depth, height, width
        self.d_dim = int(2 * ((self.attention_head_size // 3) // 2))
        self.h_dim = int(2 * ((self.attention_head_size // 3) // 2))
        self.w_dim = int(2 * ((self.attention_head_size // 3) // 2))

        self.grid_size = config.crop_size // config.patch_size
        self.n_registers = config.n_registers
        self.has_cls_first = config.has_cls_first
        self.interpolate_rope = config.interpolate_rope
        self.pretrained_grid_size = config.pretrained_grid_size

    def _separate_positions(
        self,
        ids: torch.Tensor,
        H_patches: int | None = None,
        W_patches: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompose flat token ids into (depth, height, width) components."""
        hp = H_patches if H_patches is not None else self.grid_size
        wp = W_patches if W_patches is not None else self.grid_size
        tokens_per_frame = hp * wp
        frame_ids = ids // tokens_per_frame
        remainder = ids - tokens_per_frame * frame_ids
        height_ids = remainder // wp
        width_ids = remainder - wp * height_ids
        return frame_ids.float(), height_ids.float(), width_ids.float()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_mask: torch.Tensor | None = None,
        T: int | None = None,
        H_patches: int | None = None,
        W_patches: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, N, C = hidden_states.shape

        q = (
            self.query(hidden_states)
            .view(B, N, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        k = (
            self.key(hidden_states)
            .view(B, N, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        v = (
            self.value(hidden_states)
            .view(B, N, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        # Compute position ids
        if position_mask is not None:
            ids = position_mask.unsqueeze(1).repeat(1, self.num_attention_heads, 1)
        else:
            ids = torch.arange(N, device=hidden_states.device)

        d_mask, h_mask, w_mask = self._separate_positions(ids, H_patches, W_patches)

        # Interpolate RoPE for variable resolution
        if self.interpolate_rope:
            hp = H_patches if H_patches is not None else self.grid_size
            wp = W_patches if W_patches is not None else self.grid_size
            h_mask = h_mask * (self.pretrained_grid_size - 1) / max(hp - 1, 1)
            w_mask = w_mask * (self.pretrained_grid_size - 1) / max(wp - 1, 1)

        # Apply RoPE to each dimension slice
        s = 0
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], d_mask, self.n_registers, self.has_cls_first)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], d_mask, self.n_registers, self.has_cls_first)
        s += self.d_dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], h_mask, self.n_registers, self.has_cls_first)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], h_mask, self.n_registers, self.has_cls_first)
        s += self.h_dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], w_mask, self.n_registers, self.has_cls_first)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], w_mask, self.n_registers, self.has_cls_first)
        s += self.w_dim

        if s < self.attention_head_size:
            q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
            k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        context_layer, attn_weights = attention_interface(
            self,
            q,
            k,
            v,
            None,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        output = self.proj(context_layer.reshape(B, N, self.all_head_size))
        return output, attn_weights


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class VJEPA21MLP(nn.Module):
    """Standard GELU MLP."""

    def __init__(self, config: VJEPA21Config, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_features = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, hidden_features)
        self.act = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(hidden_features, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VJEPA21SwiGLUMLP(nn.Module):
    """SwiGLU FFN as used in V-JEPA 2.1."""

    def __init__(self, config: VJEPA21Config, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_features = int(hidden_size * mlp_ratio)
        if config.wide_silu:
            swiglu_hidden = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden = (swiglu_hidden + align_as - 1) // align_as * align_as
        else:
            swiglu_hidden = hidden_features
        self.fc1 = nn.Linear(hidden_size, swiglu_hidden)
        self.fc2 = nn.Linear(hidden_size, swiglu_hidden)
        self.fc3 = nn.Linear(swiglu_hidden, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


def _make_mlp(config: VJEPA21Config, hidden_size: int, mlp_ratio: float) -> nn.Module:
    if config.hidden_act == "silu":
        return VJEPA21SwiGLUMLP(config, hidden_size, mlp_ratio)
    return VJEPA21MLP(config, hidden_size, mlp_ratio)


# ---------------------------------------------------------------------------
# Transformer Layer
# ---------------------------------------------------------------------------


class VJEPA21Layer(nn.Module):
    """Single transformer block: LN → Attention → DropPath + Residual → LN → MLP → DropPath + Residual."""

    def __init__(
        self,
        config: VJEPA21Config,
        hidden_size: int,
        num_attention_heads: int,
        mlp_ratio: float,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.attention = VJEPA21RopeAttention(config, hidden_size, num_attention_heads)
        self.drop_path = VJEPA21DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.mlp = _make_mlp(config, hidden_size, mlp_ratio)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_mask: torch.Tensor | None = None,
        T: int | None = None,
        H_patches: int | None = None,
        W_patches: int | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        h = self.norm1(hidden_states)
        attn_out, attn_weights = self.attention(h, position_mask, T, H_patches, W_patches)
        hidden_states = residual + self.drop_path(attn_out)

        residual = hidden_states
        hidden_states = residual + self.drop_path(self.mlp(self.norm2(hidden_states)))

        return hidden_states, attn_weights


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class VJEPA21Encoder(nn.Module):
    """V-JEPA 2.1 encoder with hierarchical output support."""

    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.config = config
        self.embeddings = VJEPA21Embeddings(config)

        dpr = [
            config.drop_path_rate * i / max(config.num_hidden_layers - 1, 1)
            for i in range(config.num_hidden_layers)
        ]
        self.layer = nn.ModuleList(
            [
                VJEPA21Layer(
                    config,
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path_rate=dpr[i],
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        # Per-layer norms for hierarchical outputs
        hier_layers = config.encoder_hierarchical_layers
        self.norms_block = nn.ModuleList(
            [nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) for _ in hier_layers]
        )

        self._hier_layers = hier_layers
        self._distill_layers = config.encoder_distillation_layers

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        return_hierarchical: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> VJEPA21EncoderOutput:
        embeddings, mode = self.embeddings(pixel_values_videos)

        # Compute spatial/temporal grid sizes for RoPE (channels-first contract)
        if pixel_values_videos.ndim == 5:
            B, C, T_raw, H, W = pixel_values_videos.shape   # (B, C, T, H, W)
        else:
            B, C, H, W = pixel_values_videos.shape          # (B, C, H, W) image
            T_raw = 1

        is_image = (
            self.config.img_temporal_dim_size is not None
            and T_raw == self.config.img_temporal_dim_size
        )
        if is_image:
            T_patches = T_raw  # tubelet_size=1 for images
        else:
            T_patches = T_raw // self.config.tubelet_size

        H_patches = H // self.config.patch_size
        W_patches = W // self.config.patch_size

        hidden_states = embeddings
        hier_outputs = []

        for i, layer_module in enumerate(self.layer):
            layer_out = layer_module(
                hidden_states,
                position_mask=None,
                T=T_patches,
                H_patches=H_patches,
                W_patches=W_patches,
                **kwargs,
            )
            hidden_states = layer_out[0]

            if i in self._distill_layers:
                idx = self._hier_layers.index(i)
                hier_outputs.append(self.norms_block[idx](hidden_states))

        # Final output: always use the last hierarchical norm
        last_hidden_state = self.norms_block[-1](hidden_states) if not hier_outputs else hier_outputs[-1]

        hierarchical_hidden_state = None
        if return_hierarchical and len(hier_outputs) > 1:
            hierarchical_hidden_state = torch.cat(hier_outputs, dim=2)

        return VJEPA21EncoderOutput(
            last_hidden_state=last_hidden_state,
            hierarchical_hidden_state=hierarchical_hidden_state,
        )


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class VJEPA21PredictorEmbeddings(nn.Module):
    """Predictor embeddings with hierarchical input fusion."""

    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.config = config

        n_hier = len(config.predictor_hierarchical_layers)
        if n_hier <= 1:
            self.predictor_embed = nn.Linear(config.hidden_size, config.pred_hidden_size)
        else:
            act = nn.SiLU if config.hidden_act == "silu" else nn.GELU
            self.predictor_embed = nn.Sequential(
                nn.Linear(config.hidden_size * n_hier, config.hidden_size),
                act(),
                nn.Linear(config.hidden_size, config.pred_hidden_size),
            )

        self.num_mask_tokens = config.pred_num_mask_tokens
        self.mask_tokens = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1, config.pred_hidden_size)) for _ in range(self.num_mask_tokens)]
        )

        # Modality embeddings in predictor
        self.img_mod_embed = None
        self.video_mod_embed = None
        if config.img_temporal_dim_size is not None and config.modality_embedding:
            self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, config.pred_hidden_size))
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, config.pred_hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_mask: list[torch.Tensor],
        target_mask: list[torch.Tensor],
        mask_index: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = hidden_states.size(0)
        context = self.predictor_embed(hidden_states)
        _, N_ctxt, D = context.shape

        mask_index = mask_index % self.num_mask_tokens
        pred_tokens = self.mask_tokens[mask_index].repeat(B, self._max_patches(target_mask), 1)
        pred_tokens = apply_masks(pred_tokens, target_mask)

        context = context.repeat(len(context_mask), 1, 1)
        embeddings = torch.cat([context, pred_tokens], dim=1)

        cm = torch.cat(context_mask, dim=0)
        tm = torch.cat(target_mask, dim=0)
        masks = torch.cat([cm, tm], dim=1)

        return embeddings, masks

    @staticmethod
    def _max_patches(masks: list[torch.Tensor]) -> int:
        return max(m.max().item() for m in masks) + 1


class VJEPA21Predictor(nn.Module):
    """V-JEPA 2.1 predictor with hierarchical input and context projection."""

    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.config = config
        self.embeddings = VJEPA21PredictorEmbeddings(config)

        dpr = [
            config.drop_path_rate * i / max(config.pred_num_hidden_layers - 1, 1)
            for i in range(config.pred_num_hidden_layers)
        ]
        self.layer = nn.ModuleList(
            [
                VJEPA21Layer(
                    config,
                    hidden_size=config.pred_hidden_size,
                    num_attention_heads=config.pred_num_attention_heads,
                    mlp_ratio=config.pred_mlp_ratio,
                    drop_path_rate=dpr[i],
                )
                for i in range(config.pred_num_hidden_layers)
            ]
        )
        self.layernorm = nn.LayerNorm(config.pred_hidden_size, eps=config.layer_norm_eps)

        # Output projection
        n_hier = len(config.predictor_hierarchical_layers)
        if config.pred_teacher_embed_dim is not None:
            out_dim = config.pred_teacher_embed_dim // n_hier
        else:
            out_dim = config.hidden_size
        proj_out_dim = n_hier * out_dim
        self.proj = nn.Linear(config.pred_hidden_size, proj_out_dim)

        self.proj_context = None
        if config.pred_return_all_tokens:
            self.proj_context = nn.Linear(config.pred_hidden_size, proj_out_dim)

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        context_mask: list[torch.Tensor],
        target_mask: list[torch.Tensor],
        mode: str = "video",
        **kwargs: Unpack[TransformersKwargs],
    ) -> VJEPA21PredictorOutput:
        masked_states = apply_masks(encoder_hidden_states, context_mask)
        _, N_ctxt, _ = masked_states.shape

        hidden_states, position_masks = self.embeddings(masked_states, context_mask, target_mask)

        # Sort tokens by position for attention
        argsort = torch.argsort(position_masks, dim=1)
        idx_expand = argsort.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        hidden_states = torch.gather(hidden_states, 1, idx_expand.to(hidden_states.device))
        position_masks = torch.gather(position_masks, 1, argsort.to(position_masks.device))

        # Add modality embedding
        if self.embeddings.img_mod_embed is not None:
            if mode == "img":
                hidden_states = hidden_states + self.embeddings.img_mod_embed
            else:
                hidden_states = hidden_states + self.embeddings.video_mod_embed

        for layer_module in self.layer:
            layer_out = layer_module(hidden_states, position_mask=position_masks, **kwargs)
            hidden_states = layer_out[0]

        hidden_states = self.layernorm(hidden_states)

        # Unsort
        reverse = torch.argsort(argsort, dim=1)
        rev_expand = reverse.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        hidden_states = torch.gather(hidden_states, 1, rev_expand.to(hidden_states.device))

        if not self.config.pred_return_all_tokens:
            pred = self.proj(hidden_states[:, N_ctxt:])
            return VJEPA21PredictorOutput(last_hidden_state=pred)
        else:
            pred = self.proj(hidden_states[:, N_ctxt:])
            ctx = self.proj_context(hidden_states[:, :N_ctxt])
            return VJEPA21PredictorOutput(last_hidden_state=pred, context_hidden_state=ctx)


# ---------------------------------------------------------------------------
# Attentive Pooler (for downstream tasks)
# ---------------------------------------------------------------------------


class VJEPA21PoolerSelfAttention(nn.Module):
    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_probs_dropout_prob
        self.is_causal = False
        self.config = config

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, N, C = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self, q, k, v, None, is_causal=False, scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )
        return self.out_proj(attn_output.reshape(B, N, C)), attn_weights


class VJEPA21PoolerCrossAttention(nn.Module):
    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_probs_dropout_prob
        self.config = config

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self, queries: torch.Tensor, kv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, Nq, C = queries.shape
        Nkv = kv.shape[1]
        q = self.q_proj(queries).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self, q, k, v, None, is_causal=False, scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )
        return attn_output.reshape(B, Nq, C), attn_weights


class VJEPA21PoolerSelfAttentionLayer(nn.Module):
    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = VJEPA21PoolerSelfAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VJEPA21MLP(config, hidden_size=config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(self.layer_norm1(hidden_states))
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = residual + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class VJEPA21PoolerCrossAttentionLayer(nn.Module):
    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cross_attn = VJEPA21PoolerCrossAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VJEPA21MLP(config, hidden_size=config.hidden_size)

    def forward(self, queries: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        residual = queries
        hidden, _ = self.cross_attn(queries, self.layer_norm1(kv))
        hidden = residual + hidden
        residual = hidden
        hidden = residual + self.mlp(self.layer_norm2(hidden))
        return hidden


class VJEPA21AttentivePooler(nn.Module):
    def __init__(self, config: VJEPA21Config):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.cross_attention_layer = VJEPA21PoolerCrossAttentionLayer(config)
        self.self_attention_layers = nn.ModuleList(
            [VJEPA21PoolerSelfAttentionLayer(config) for _ in range(config.num_pooler_layers)]
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        for layer in self.self_attention_layers:
            hidden_state = layer(hidden_state)
        queries = self.query_tokens.expand(hidden_state.shape[0], -1, -1)
        return self.cross_attention_layer(queries, hidden_state).squeeze(1)


# ---------------------------------------------------------------------------
# PreTrainedModel base
# ---------------------------------------------------------------------------


class VJEPA21PreTrainedModel(PreTrainedModel):
    config_class = VJEPA21Config
    base_model_prefix = "vjepa21"
    main_input_name = "pixel_values_videos"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VJEPA21Layer"]
    _supports_sdpa = True
    _supports_flash_attn = True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, VJEPA21AttentivePooler):
            nn.init.trunc_normal_(module.query_tokens, std=std)
        elif isinstance(module, VJEPA21PredictorEmbeddings):
            if self.config.pred_zero_init_mask_tokens:
                for mt in module.mask_tokens:
                    nn.init.zeros_(mt)
            else:
                for mt in module.mask_tokens:
                    nn.init.trunc_normal_(mt, std=std)
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        # Modality embeddings
        if isinstance(module, VJEPA21Embeddings):
            if module.img_mod_embed is not None:
                nn.init.normal_(module.img_mod_embed, std=1e-6)
                nn.init.normal_(module.video_mod_embed, std=1e-6)
        if isinstance(module, VJEPA21PredictorEmbeddings):
            if module.img_mod_embed is not None:
                nn.init.normal_(module.img_mod_embed, std=1e-6)
                nn.init.normal_(module.video_mod_embed, std=1e-6)


# ---------------------------------------------------------------------------
# Main Models
# ---------------------------------------------------------------------------


class VJEPA21Model(VJEPA21PreTrainedModel):
    """V-JEPA 2.1 model (encoder + predictor).

    Usage for feature extraction (e.g., as VLM teacher):
    ```python
    model = VJEPA21Model.from_pretrained("path/to/model")
    outputs = model(pixel_values_videos, skip_predictor=True)
    features = outputs.last_hidden_state
    ```
    """

    def __init__(self, config: VJEPA21Config):
        super().__init__(config)
        self.encoder = VJEPA21Encoder(config)
        self.predictor = VJEPA21Predictor(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        context_mask: list[torch.Tensor] | None = None,
        target_mask: list[torch.Tensor] | None = None,
        skip_predictor: bool = False,
        return_hierarchical: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> VJEPA21ModelOutput:
        """
        Args:
            pixel_values_videos: channels-first video `(B, C, T, H, W)`
                (or image `(B, C, H, W)`). Permute upstream if channels-last.
            context_mask: List of `(B, K)` index tensors for context tokens.
            target_mask: List of `(B, K)` index tensors for target tokens.
            skip_predictor: If True, skip predictor forward (encoder only).
            return_hierarchical: If True, return hierarchical intermediate features.
        """
        if pixel_values_videos is None:
            raise ValueError("pixel_values_videos is required")

        encoder_out = self.encoder(
            pixel_values_videos,
            return_hierarchical=return_hierarchical,
            **kwargs,
        )
        seq_output = encoder_out.last_hidden_state

        predictor_output = None
        masked_hidden_state = None

        if not skip_predictor:
            B = pixel_values_videos.size(0)
            N = seq_output.size(1)
            device = seq_output.device

            if context_mask is None:
                context_mask = [torch.arange(N, device=device).unsqueeze(0).expand(B, -1)]
            if target_mask is None:
                target_mask = [torch.arange(N, device=device).unsqueeze(0).expand(B, -1)]

            # Determine mode from embeddings
            mode = self._detect_mode(pixel_values_videos)

            predictor_out = self.predictor(
                seq_output, context_mask, target_mask, mode=mode, **kwargs,
            )
            predictor_output = predictor_out
            masked_hidden_state = apply_masks(seq_output, context_mask)

        return VJEPA21ModelOutput(
            last_hidden_state=seq_output,
            hierarchical_hidden_state=encoder_out.hierarchical_hidden_state,
            masked_hidden_state=masked_hidden_state,
            predictor_output=predictor_output,
        )

    def _detect_mode(self, pixel_values_videos: torch.Tensor) -> str:
        if pixel_values_videos.ndim == 5:
            T = pixel_values_videos.shape[2]   # (B, C, T, H, W)
        else:
            T = 1
        if (
            self.config.img_temporal_dim_size is not None
            and T == self.config.img_temporal_dim_size
        ):
            return "img"
        return "video"

    def get_vision_features(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        """Extract encoder features (convenience method for VLM integration)."""
        return self.forward(pixel_values_videos, skip_predictor=True).last_hidden_state


class VJEPA21ForVideoClassification(VJEPA21PreTrainedModel):
    """V-JEPA 2.1 with attentive pooler + classification head."""

    def __init__(self, config: VJEPA21Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vjepa21 = VJEPA21Model(config)
        self.pooler = VJEPA21AttentivePooler(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutput:
        outputs = self.vjepa21(pixel_values_videos, skip_predictor=True, **kwargs)
        pooled = self.pooler(outputs.last_hidden_state)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_function(pooled_logits=logits, labels=labels, config=self.config)

        return ImageClassifierOutput(loss=loss, logits=logits)


__all__ = [
    "VJEPA21Config",
    "VJEPA21Model",
    "VJEPA21PreTrainedModel",
    "VJEPA21ForVideoClassification",
]
