"""V-JEPA 2.1 model configuration"""

from transformers import PretrainedConfig


class VJEPA21Config(PretrainedConfig):
    r"""
    Configuration class for the V-JEPA 2.1 model.

    V-JEPA 2.1 extends V-JEPA 2 with:
    - Multi-modality support (image + video with modality embeddings)
    - Hierarchical output distillation across intermediate layers
    - Interpolatable RoPE for variable input resolutions
    - Dense predictive loss with context token prediction

    Args:
        patch_size (`int`, defaults to 16):
            Spatial patch size.
        crop_size (`int`, defaults to 384):
            Input resolution of the model.
        frames_per_clip (`int`, defaults to 64):
            Number of frames in a video clip.
        tubelet_size (`int`, defaults to 2):
            Temporal patch size (number of frames per tubelet).
        hidden_size (`int`, defaults to 1024):
            Encoder embedding dimension.
        in_chans (`int`, defaults to 3):
            Number of input channels.
        num_attention_heads (`int`, defaults to 16):
            Number of attention heads in the encoder.
        num_hidden_layers (`int`, defaults to 24):
            Number of encoder transformer layers.
        drop_path_rate (`float`, defaults to 0.0):
            Stochastic depth rate.
        mlp_ratio (`float`, defaults to 4.0):
            Ratio of MLP hidden dim to embedding dim.
        layer_norm_eps (`float`, defaults to 1e-6):
            Layer normalization epsilon.
        qkv_bias (`bool`, defaults to True):
            Whether to use bias in QKV projection.
        hidden_act (`str`, defaults to "gelu"):
            Activation function in MLP. "silu" enables SwiGLU.
        wide_silu (`bool`, defaults to True):
            Whether to use wide SwiGLU (2/3 hidden features) when hidden_act is "silu".
        initializer_range (`float`, defaults to 0.02):
            Standard deviation for weight initialization.
        attention_probs_dropout_prob (`float`, defaults to 0.0):
            Dropout probability for attention weights.
        img_temporal_dim_size (`int` or `None`, defaults to 1):
            Temporal dimension for image inputs. When set, a separate patch embedding
            with tubelet_size=1 is used for images. Set to None to disable.
        interpolate_rope (`bool`, defaults to True):
            Whether to interpolate RoPE frequencies for variable input resolutions.
        modality_embedding (`bool`, defaults to True):
            Whether to add learned modality embeddings (image vs video).
        n_output_distillation (`int`, defaults to 4):
            Number of intermediate encoder layers for hierarchical output.
            Set to 1 to only use the final layer output.
        n_registers (`int`, defaults to 0):
            Number of register tokens (appended to sequence).
        has_cls_first (`bool`, defaults to False):
            Whether the sequence starts with a CLS token.
        num_pooler_layers (`int`, defaults to 3):
            Number of self-attention layers in the attentive pooler.
        pred_hidden_size (`int`, defaults to 384):
            Predictor embedding dimension.
        pred_num_attention_heads (`int`, defaults to 12):
            Number of attention heads in the predictor.
        pred_num_hidden_layers (`int`, defaults to 12):
            Number of predictor transformer layers.
        pred_num_mask_tokens (`int`, defaults to 8):
            Number of learnable mask tokens in the predictor.
        pred_zero_init_mask_tokens (`bool`, defaults to True):
            Whether to zero-initialize mask tokens.
        pred_mlp_ratio (`float`, defaults to 4.0):
            MLP ratio in the predictor.
        pred_teacher_embed_dim (`int` or `None`, defaults to None):
            Teacher embedding dimension for predictor output projection.
            When set, predictor projects to teacher_embed_dim // n_hierarchical_layers per layer.
        pred_return_all_tokens (`bool`, defaults to False):
            Whether the predictor returns predictions for both masked and context tokens.
    """

    model_type = "vjepa21"

    def __init__(
        self,
        patch_size: int = 16,
        crop_size: int = 384,
        frames_per_clip: int = 64,
        tubelet_size: int = 2,
        hidden_size: int = 1024,
        in_chans: int = 3,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 24,
        drop_path_rate: float = 0.0,
        mlp_ratio: float = 4.0,
        layer_norm_eps: float = 1e-6,
        qkv_bias: bool = True,
        hidden_act: str = "gelu",
        wide_silu: bool = True,
        initializer_range: float = 0.02,
        attention_probs_dropout_prob: float = 0.0,
        # V-JEPA 2.1 specific
        img_temporal_dim_size: int | None = 1,
        interpolate_rope: bool = True,
        modality_embedding: bool = True,
        n_output_distillation: int = 4,
        n_registers: int = 0,
        has_cls_first: bool = False,
        # Pooler
        num_pooler_layers: int = 3,
        # Predictor
        pred_hidden_size: int = 384,
        pred_num_attention_heads: int = 12,
        pred_num_hidden_layers: int = 12,
        pred_num_mask_tokens: int = 8,
        pred_zero_init_mask_tokens: bool = True,
        pred_mlp_ratio: float = 4.0,
        pred_teacher_embed_dim: int | None = None,
        pred_return_all_tokens: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.crop_size = crop_size
        self.frames_per_clip = frames_per_clip
        self.tubelet_size = tubelet_size
        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.drop_path_rate = drop_path_rate
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.hidden_act = hidden_act
        self.wide_silu = wide_silu
        self.initializer_range = initializer_range
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        # V-JEPA 2.1 specific
        self.img_temporal_dim_size = img_temporal_dim_size
        self.interpolate_rope = interpolate_rope
        self.modality_embedding = modality_embedding
        self.n_output_distillation = n_output_distillation
        self.n_registers = n_registers
        self.has_cls_first = has_cls_first

        # Pooler
        self.num_pooler_layers = num_pooler_layers

        # Predictor
        self.pred_hidden_size = pred_hidden_size
        self.pred_num_attention_heads = pred_num_attention_heads
        self.pred_num_hidden_layers = pred_num_hidden_layers
        self.pred_num_mask_tokens = pred_num_mask_tokens
        self.pred_zero_init_mask_tokens = pred_zero_init_mask_tokens
        self.pred_mlp_ratio = pred_mlp_ratio
        self.pred_teacher_embed_dim = pred_teacher_embed_dim
        self.pred_return_all_tokens = pred_return_all_tokens

    @property
    def encoder_hierarchical_layers(self) -> list[int]:
        """Layer indices for hierarchical output collection in the encoder."""
        return _get_hierarchical_layers(self.num_hidden_layers)

    @property
    def encoder_distillation_layers(self) -> list[int]:
        """Layer indices used for distillation output in the encoder."""
        all_layers = _get_hierarchical_layers(self.num_hidden_layers)
        return all_layers[-self.n_output_distillation :]

    @property
    def predictor_hierarchical_layers(self) -> list[int]:
        """Layer indices for hierarchical output in the predictor."""
        all_layers = _get_hierarchical_layers(self.pred_num_hidden_layers)
        # Predictor uses n_output_distillation from its own kwargs
        return all_layers[-self.n_output_distillation :]

    @property
    def pretrained_grid_size(self) -> int:
        """Grid size used during pre-training (for RoPE interpolation)."""
        if self.patch_size == 14:
            return int(252 / self.patch_size)
        return int(256 / self.patch_size)


def _get_hierarchical_layers(depth: int) -> list[int]:
    """Get hierarchical layer indices based on model depth."""
    _LAYER_MAP = {
        4: [0, 1, 2, 3],
        8: [1, 3, 5, 7],
        12: [2, 5, 8, 11],
        20: [4, 9, 14, 19],
        24: [5, 11, 17, 23],
        40: [9, 19, 29, 39],
        48: [11, 23, 37, 47],
    }
    if depth not in _LAYER_MAP:
        raise ValueError(
            f"Unsupported depth {depth}. Supported depths: {list(_LAYER_MAP.keys())}"
        )
    return _LAYER_MAP[depth]
