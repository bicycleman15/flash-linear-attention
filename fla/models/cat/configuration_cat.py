# CAT (Compress And Attend Transformers) implementation based on:
# https://github.com/rajesh-lab/cat-transformer

from __future__ import annotations

import warnings

from transformers.configuration_utils import PretrainedConfig


class CATConfig(PretrainedConfig):
    """
    Configuration class for CAT (Compress And Attend Transformers).

    CAT compresses chunks of tokens using a compressor and attends over
    compressed chunk representations. 
    Varying chunk size enables controllable efficiency at test-time.

    Args:
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the decoder hidden states.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of decoder layers.
        num_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for the decoder.
        num_kv_heads (`int`, *optional*):
            Number of key-value heads for grouped query attention.
        max_chunk_size (`int`, *optional*, defaults to 16):
            Maximum chunk size for training. 
            At inference, any chunk_size <= max_chunk_size can be used 
            Note that one must train for chunk sizes they wish to use during inference. 
            CAT does not support extrapolation to new chunk sizes not seen during training.
        dim_fx (`int`, *optional*):
            Dimension of compressed chunk representations. Defaults to hidden_size.
        compressor_hidden_size (`int`, *optional*):
            Hidden size for the compressor. Defaults to hidden_size // 2.
        compressor_num_layers (`int`, *optional*):
            Number of layers in the compressor. Defaults to num_hidden_layers // 4.
        compressor_num_heads (`int`, *optional*):
            Number of attention heads in the compressor. Defaults to num_heads // 2.
        qkv_bias (`bool`, *optional*, defaults to False):
            Whether to use bias in QKV projections.
        qk_norm (`bool`, *optional*, defaults to False):
            Whether to apply QK normalization.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base value for rotary position embeddings.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            Maximum sequence length.
        hidden_ratio (`int`, *optional*, defaults to 4):
            Ratio for MLP hidden size.
        intermediate_size (`int`, *optional*):
            Intermediate size for MLP. If None, computed from hidden_ratio.
        hidden_act (`str`, *optional*, defaults to "swish"):
            Activation function for MLP.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        elementwise_affine (`bool`, *optional*, defaults to True):
            Whether to use elementwise affine in layer norms.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for layer normalization.
        use_cache (`bool`, *optional*, defaults to True):
            Whether to use KV cache for generation.
        fuse_norm (`bool`, *optional*, defaults to True):
            Whether to use fused normalization operations.
        fuse_swiglu (`bool`, *optional*, defaults to True):
            Whether to use fused SwiGLU activation.
        fuse_cross_entropy (`bool`, *optional*, defaults to True):
            Whether to use fused cross entropy loss.
        fuse_linear_cross_entropy (`bool`, *optional*, defaults to False):
            Whether to use fused linear cross entropy (memory efficient).
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size.
        use_staircase_attention (`bool`, *optional*, defaults to True):
            Whether to use staircase attention pattern with globals-first sequence structure.
            This is better suited for taking advantage of FlexAttention to provide speedups in pre-training.
            When True, sequence is [fx_0, ..., fx_K, tok_0, ..., tok_L]
            When False, sequence is interleaved [fx_0, sep, tok_0, fx_1, sep, tok_1, ...].
    """

    model_type = 'cat'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        max_chunk_size: int = 16,
        dim_fx: int | None = None,
        compressor_hidden_size: int | None = None,
        compressor_num_layers: int | None = None,
        compressor_num_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        rope_theta: float | None = 10000.,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        initializer_range: float = 0.02,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        vocab_size: int = 32000,
        use_staircase_attention: bool = True,
        **kwargs,
    ):
        # Decoder parameters
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        # CAT-specific parameters
        self.max_chunk_size = max_chunk_size
        self.dim_fx = dim_fx if dim_fx is not None else hidden_size

        # Compressor parameters (defaults follow paper recommendations)
        self.compressor_hidden_size = compressor_hidden_size if compressor_hidden_size is not None else hidden_size // 2
        self.compressor_num_layers = compressor_num_layers if compressor_num_layers is not None else max(1, num_hidden_layers // 4)
        self.compressor_num_heads = compressor_num_heads if compressor_num_heads is not None else max(1, num_heads // 2)

        # Validate that dim_fx matches hidden_size (required for decoder input)
        if self.dim_fx != hidden_size:
            raise ValueError(
                f"dim_fx ({self.dim_fx}) must equal hidden_size ({hidden_size}) "
                "since compressed representations are fed directly to the decoder."
            )

        # Attention parameters
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # MLP parameters
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        # General parameters
        self.initializer_range = initializer_range
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache

        # Fusion flags
        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.vocab_size = vocab_size
        
        # Staircase attention (globals-first sequence structure)
        self.use_staircase_attention = use_staircase_attention

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time.",
            )
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which can improve memory efficiency "
                "at the potential cost of reduced precision. "
                "If you observe issues like loss divergence, consider disabling this setting.",
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
