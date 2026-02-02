# CAT (Compress And Attend Transformers) implementation based on:
# https://github.com/rajesh-lab/cat-transformer
# Paper: "Attention and Compression is all you need for Controllably Efficient Language Models"

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.models.cat.configuration_cat import CATConfig
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP as CATMLP
from fla.modules import RotaryEmbedding

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer


from torch.nn.attention.flex_attention import create_block_mask, flex_attention, BlockMask

create_block_mask = torch.compile(create_block_mask)

_flex_attention_compiled = torch.compile(flex_attention, dynamic=False, mode="default")


@torch.compiler.disable(recursive=False)
def flex_attention_compiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: Optional[BlockMask] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    return _flex_attention_compiled(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa)

logger = logging.get_logger(__name__)


def get_cat_mask_mod(block_size: int):
    """
    Creates the CAT attention mask. This mask allows:
    - tokens within the same chunk to attend to each other (causal)
    - tokens to attend previous compressed representations
    """
    def cat_mask(b, h, q_idx, kv_idx):
        within_block = (q_idx // block_size) == (kv_idx // block_size)
        divides_block = (kv_idx % block_size) == 0
        causal_mask = q_idx >= kv_idx
        return (divides_block | within_block) & causal_mask
    return cat_mask


def get_block_diagonal_mask_mod(block_size: int):
    # used in compressor
    def block_diagonal(b, h, q_idx, kv_idx):
        return (q_idx // block_size) == (kv_idx // block_size)
    return block_diagonal


class CATCompressorBlock(GradientCheckpointingLayer):
    """Transformer block for the CAT compressor."""

    def __init__(self, config: CATConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        hidden_size = config.compressor_hidden_size

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(hidden_size, eps=config.norm_eps)
        
        # Simple self-attention for compressor (no causal mask needed - bidirectional)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=config.qkv_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=config.qkv_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=config.qkv_bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.num_heads = config.compressor_num_heads
        self.head_dim = hidden_size // self.num_heads
        
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(hidden_size, eps=config.norm_eps)
        self.mlp = CATMLP(
            hidden_size=hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        block_mask: BlockMask,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        
        batch_size, seq_len, _ = hidden_states.shape
        
        q = rearrange(self.q_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_heads)
        k = rearrange(self.k_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_heads)
        v = rearrange(self.v_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_heads)
        
        # Apply rotary embeddings
        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)
        
        q = rearrange(q, 'b t h d -> b h t d')
        k = rearrange(k, 'b t h d -> b h t d')
        v = rearrange(v, 'b t h d -> b h t d')
        
        # Use flex_attention with block mask
        attn_output = flex_attention_compiled(q, k, v, block_mask=block_mask)
        attn_output = rearrange(attn_output, 'b h t d -> b t (h d)')
        hidden_states = self.o_proj(attn_output)
        
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings."""
        # x: [batch, seq, heads, dim]
        # cos, sin: 2D [seq, dim//2] from cache or 3D [1, seq, dim//2]
        seq_len = x.shape[1]
        if cos.dim() == 2:
            cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(2)  # [1, seq, 1, dim//2]
            sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(2)
        else:
            cos = cos[:, :seq_len, :].unsqueeze(2)  # [1, seq, 1, dim//2]
            sin = sin[:, :seq_len, :].unsqueeze(2)
        
        # Split x into two halves
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return rotated


class CATCompressor(nn.Module):
    """
    Compressor module for CAT that compresses chunks of tokens into fixed-size representations.
    
    The compressor processes each chunk independently and produces a single compressed
    representation (f(chunk)) for each chunk.
    """

    def __init__(self, config: CATConfig):
        super().__init__()
        self.config = config
        
        hidden_size = config.compressor_hidden_size
        self.hidden_size = hidden_size
        self.max_chunk_size = config.max_chunk_size
        
        # Token embeddings (shared with decoder via weight tying later)
        self.wte = nn.Embedding(config.vocab_size, hidden_size)
        
        # Position tokens for each chunk position
        self.pos_tokens = nn.Embedding(config.max_position_embeddings, hidden_size)
        
        # Adaptive tokens to encode the current chunk size
        self.adaptive_tokens = nn.Embedding(config.max_chunk_size + 1, hidden_size)
        
        # Transformer layers for compression
        self.layers = nn.ModuleList([
            CATCompressorBlock(config, layer_idx=i) 
            for i in range(config.compressor_num_layers)
        ])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(hidden_size, eps=config.norm_eps)
        
        # Projection from flattened chunk to dim_fx
        # Weight is interpolated for different chunk sizes
        self.proj_fx = nn.Linear(hidden_size * config.max_chunk_size, config.dim_fx, bias=False)
        
        # RoPE for compressor (positions within chunk)
        self.rotary = RotaryEmbedding(
            dim=hidden_size // config.compressor_num_heads,
            base=config.rope_theta,
        )

    def compress(
        self,
        input_ids: torch.LongTensor,
        chunk_idx: torch.LongTensor,
        chunk_size: int,
    ) -> torch.Tensor:
        """
        Compress a single chunk of tokens.
        
        Args:
            input_ids: Token IDs for the chunk [batch_size, chunk_size]
            chunk_idx: Index of this chunk in the sequence [1]
            chunk_size: Current chunk size being used
            
        Returns:
            Compressed representation [batch_size, dim_fx]
        """
        batch_size, seq_len = input_ids.shape
        assert seq_len == chunk_size
        
        # Embed tokens
        x = self.wte(input_ids)  # [batch, chunk_size, hidden]
        
        # Position token tells compressor which position this chunk is in
        pos_token = self.pos_tokens(chunk_idx)  # [hidden]
        pos_token = repeat(pos_token, 'd -> b 1 d', b=batch_size)  # [batch, 1, hidden]
        
        # Adaptive token encodes the current chunk size
        chunk_size_idx = torch.tensor([chunk_size], device=input_ids.device, dtype=torch.long)
        adaptive_token = self.adaptive_tokens(chunk_size_idx)  # [1, hidden]
        adaptive_token = repeat(adaptive_token, '1 d -> b 1 d', b=batch_size)  # [batch, 1, hidden]
        
        # Concatenate: [adaptive, position, tokens]
        x = torch.cat([adaptive_token, pos_token, x], dim=1)  # [batch, 2+chunk_size, hidden]
        
        # Get RoPE embeddings
        seq_len_with_special = x.shape[1]
        self.rotary._update_cos_sin_cache(seq_len_with_special, device=x.device, dtype=x.dtype)
        cos = self.rotary._cos_cached
        sin = self.rotary._sin_cached
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        
        # Remove special tokens and flatten
        x = x[:, 2:, :]  # [batch, chunk_size, hidden]
        x = rearrange(x, 'b l d -> b (l d)')  # [batch, chunk_size * hidden]
        
        # Adaptive projection for variable chunk sizes
        if chunk_size != self.max_chunk_size:
            # Interpolate projection weights for current chunk size
            target_in_features = chunk_size * self.hidden_size
            new_weight = F.interpolate(
                self.proj_fx.weight.unsqueeze(0).unsqueeze(0),
                size=(self.config.dim_fx, target_in_features),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            x = F.linear(x, new_weight, bias=None)
        else:
            x = self.proj_fx(x)
        
        return x  # [batch, dim_fx]

    def compress_batched(
        self,
        input_ids_chunked: torch.LongTensor,
        chunk_size: int,
    ) -> torch.Tensor:
        """
        Compress all chunks in one forward pass using a block-diagonal attention mask
        so the encoder attends only to tokens within each chunk.

        Args:
            input_ids_chunked: [batch_size, num_chunks, chunk_size]
            chunk_size: Current chunk size

        Returns:
            Compressed representations [batch_size, num_chunks, dim_fx]
        """
        batch_size, num_chunks, _chunk_size = input_ids_chunked.shape
        # sanity check
        assert chunk_size == _chunk_size, f"chunk_size ({chunk_size}) != input chunk size ({_chunk_size})"

        device = input_ids_chunked.device
        dtype = self.wte.weight.dtype
        block_len = 2 + chunk_size

        # Embed tokens: [batch, num_chunks, chunk_size, hidden]
        token_embeds = self.wte(input_ids_chunked)

        # Adaptive token (same for all chunks - encodes chunk_size)
        chunk_size_idx = torch.tensor([chunk_size], device=device, dtype=torch.long)
        adaptive_token = self.adaptive_tokens(chunk_size_idx)  # [1, hidden]
        adaptive_token = repeat(adaptive_token, '1 d -> b k 1 d', b=batch_size, k=num_chunks)

        # Position token per chunk (chunk 0 -> pos 0, chunk 1 -> pos 1, ...)
        chunk_indices = torch.arange(num_chunks, device=device, dtype=torch.long)
        pos_tokens = self.pos_tokens(chunk_indices)  # [num_chunks, hidden]
        pos_tokens = repeat(pos_tokens, 'k d -> b k 1 d', b=batch_size)  # [batch, num_chunks, 1, hidden]

        # Concatenate [adaptive, position, tokens] per chunk -> [batch, num_chunks, 2+chunk_size, hidden]
        x = torch.cat([adaptive_token, pos_tokens, token_embeds], dim=2)
        total_len = num_chunks * block_len
        x = rearrange(x, 'b k l d -> b (k l) d')  # [batch, total_len, hidden]

        # Block-diagonal mask: attend only within each chunk block
        compressor_block_mask = create_block_mask(
            get_block_diagonal_mask_mod(block_len),
            B=None,
            H=None,
            Q_LEN=total_len,
            KV_LEN=total_len,
        )

        # RoPE: positions repeat per block (position i has index i % block_len)
        self.rotary._update_cos_sin_cache(block_len, device=device, dtype=dtype)
        cos_base = self.rotary._cos_cached  # [block_len, dim//2]
        sin_base = self.rotary._sin_cached
        position_ids = torch.arange(total_len, device=device) % block_len
        cos = cos_base[position_ids].unsqueeze(0)   # [1, total_len, dim//2]
        sin = sin_base[position_ids].unsqueeze(0)

        # Transformer layers with block-diagonal attention mask
        for layer in self.layers:
            x = layer(x, cos, sin, block_mask=compressor_block_mask)
        x = self.norm(x)

        # Reshape back: [batch, total_len, hidden] -> [batch, num_chunks, block_len, hidden]
        x = rearrange(x, 'b (k l) d -> b k l d', k=num_chunks, l=block_len)
        # Drop adaptive + position; keep tokens only
        x = x[:, :, 2:, :]  # [batch, num_chunks, chunk_size, hidden]
        x = rearrange(x, 'b k l d -> b k (l d)')  # [batch, num_chunks, chunk_size * hidden]

        # Project to dim_fx (same logic as single-chunk: interpolate if chunk_size != max_chunk_size)
        if chunk_size != self.max_chunk_size:
            target_in_features = chunk_size * self.hidden_size
            new_weight = F.interpolate(
                self.proj_fx.weight.unsqueeze(0).unsqueeze(0),
                size=(self.config.dim_fx, target_in_features),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            x = F.linear(x, new_weight, bias=None)  # [batch, num_chunks, dim_fx]
        else:
            x = self.proj_fx(x)  # [batch, num_chunks, dim_fx]

        return x


class CATDecoderAttention(nn.Module):
    """
    This is same as default Attention, just with added support for FlexAttention and custom BlockMask.
    """

    def __init__(self, config: CATConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads if config.num_kv_heads is not None else config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        if config.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, dtype=torch.float32)
            self.k_norm = RMSNorm(self.head_dim, dtype=torch.float32)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        block_mask: BlockMask,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        q = rearrange(self.q_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_heads)
        k = rearrange(self.k_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_kv_heads)
        v = rearrange(self.v_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_kv_heads)
        
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Apply rotary embeddings using position indices from cos/sin
        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)
        
        # Expand KV heads for GQA
        if self.num_kv_groups > 1:
            k = repeat(k, 'b t h d -> b t (h g) d', g=self.num_kv_groups)
            v = repeat(v, 'b t h d -> b t (h g) d', g=self.num_kv_groups)
        
        # Rearrange for attention
        q = rearrange(q, 'b t h d -> b h t d')
        k = rearrange(k, 'b t h d -> b h t d')
        v = rearrange(v, 'b t h d -> b h t d')
        
        # Use flex_attention with CAT block mask
        attn_output = flex_attention_compiled(q, k, v, block_mask=block_mask)
        
        attn_output = rearrange(attn_output, 'b h t d -> b t (h d)')
        output = self.o_proj(attn_output)
        
        return output

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings."""
        seq_len = x.shape[1]
        cos = cos[:, :seq_len, :]
        sin = sin[:, :seq_len, :]
        
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return rotated


class CATBlock(GradientCheckpointingLayer):
    """Transformer block for the CAT decoder."""

    def __init__(self, config: CATConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = CATDecoderAttention(config, layer_idx)
        
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = CATMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        block_mask: BlockMask,
        **kwargs: Unpack[Any],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(
            hidden_states,
            cos=cos,
            sin=sin,
            block_mask=block_mask,
        )
        
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states
        
        return hidden_states


class CATPreTrainedModel(PreTrainedModel):
    """Base class for CAT models."""

    config_class = CATConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['CATBlock', 'CATCompressorBlock']
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if rescale_prenorm_residual:
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class CATModel(CATPreTrainedModel):
    """
    CAT (Compress And Attend Transformer) model.
    """

    def __init__(self, config: CATConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Compressor
        self.compressor = CATCompressor(config)
        
        # Decoder embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Separator token (separates fx from tokens in each chunk)
        self.separator = nn.Embedding(1, config.hidden_size)
        
        # Adaptive token -- informs the decoder about the current chunk size it is operating at
        # changing this allows the decoder to operate at multiple chunk sizes
        self.adaptive_token = nn.Embedding(config.max_chunk_size + 1, config.hidden_size)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            CATBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        
        # RoPE for decoder
        self.rotary = RotaryEmbedding(
            dim=config.hidden_size // config.num_heads,
            base=config.rope_theta,
        )
        
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def _build_cat_positions(
        self,
        num_chunks: int,
        chunk_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build position indices for CAT decoder where positions repeat per chunk.
        
        Each chunk block has positions [0, 1, ..., chunk_size+1] repeated.
        """
        block_len = 2 + chunk_size  # fx + separator + chunk_tokens
        chunk_positions = torch.arange(block_len, device=device)
        positions = chunk_positions.repeat(num_chunks)
        # Add final fx and separator positions
        positions = torch.cat([positions, torch.arange(2, device=device)])
        return positions

    def _build_cat_rope_cache(
        self,
        num_chunks: int,
        chunk_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build RoPE cos/sin cache for CAT decoder positions."""
        block_len = 2 + chunk_size
        max_pos = block_len + 2  # Maximum position needed
        
        # Update rotary cache
        self.rotary._update_cos_sin_cache(max_pos, device=device, dtype=dtype)
        
        # Get base cos/sin
        cos_base = self.rotary._cos_cached  # [max_pos, dim//2]
        sin_base = self.rotary._sin_cached  # [max_pos, dim//2]
        
        # Build positions that repeat per chunk
        positions = self._build_cat_positions(num_chunks, chunk_size, device)
        
        # Index into base cache
        cos = cos_base[positions]  # [total_len, dim//2]
        sin = sin_base[positions]  # [total_len, dim//2]
        
        return cos.unsqueeze(0), sin.unsqueeze(0)  # [1, total_len, dim//2]

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        chunk_size: int | None = None,
        **kwargs: Unpack[Any],
    ) -> tuple | BaseModelOutputWithPast:
        if output_attentions:
            logger.warning_once("`CATModel` does not support output_attentions, setting to False.")
            output_attentions = False
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Default to max_chunk_size if not specified
        if chunk_size is None:
            chunk_size = self.config.max_chunk_size
        
        if chunk_size > self.config.max_chunk_size:
            raise ValueError(
                f"chunk_size ({chunk_size}) cannot exceed max_chunk_size ({self.config.max_chunk_size})"
            )
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")
        
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
        else:
            batch_size, seq_len, _ = inputs_embeds.shape
            device = inputs_embeds.device
        
        # Pad sequence to a multiple of 512 (and chunk_size) to prevent flex_attention recompilation
        # The padding multiple must be divisible by chunk_size
        original_seq_len = seq_len
        pad_multiple = max(512, chunk_size)
        # Ensure pad_multiple is a multiple of chunk_size
        if pad_multiple % chunk_size != 0:
            pad_multiple = ((pad_multiple // chunk_size) + 1) * chunk_size
        
        if seq_len % pad_multiple != 0:
            pad_len = pad_multiple - (seq_len % pad_multiple)
            if input_ids is not None:
                pad_token = self.padding_idx if self.padding_idx is not None else 0
                input_ids = F.pad(input_ids, (0, pad_len), value=pad_token)
            else:
                inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, pad_len), value=0)
            seq_len = seq_len + pad_len
        
        num_chunks = seq_len // chunk_size
        
        if input_ids is not None:
            # Reshape for chunk processing
            input_ids_chunked = input_ids.view(batch_size, num_chunks, chunk_size)
            
            # Compress all chunks in one forward pass with block-diagonal attention mask
            # (compressor attends only to tokens within each chunk)
            fx = self.compressor.compress_batched(input_ids_chunked, chunk_size)  # [batch, num_chunks, dim_fx]
            
            # Get last chunk's fx for the last token's prediction in the input sequence
            fx_last = fx[:, -1:, :]  # [batch, 1, dim_fx]
            
            # Embed tokens
            token_embeds = self.embeddings(input_ids_chunked)  # [batch, num_chunks, chunk_size, hidden]
        else:
            raise NotImplementedError("inputs_embeds not yet supported for CAT")
        
        # Build decoder input sequence for parallel processing
        # Structure: [adaptive_token, sep, tokens_0, fx_0, sep, tokens_1, ..., fx_{n-1}, sep]
        
        # Adaptive token
        chunk_size_idx = torch.tensor([chunk_size], device=device, dtype=torch.long)
        adaptive_tok = self.adaptive_token(chunk_size_idx)  # [1, hidden]
        adaptive_tok = repeat(adaptive_tok, '1 d -> b 1 d', b=batch_size)
        
        # Separator tokens
        sep_idx = torch.zeros(1, device=device, dtype=torch.long)
        sep_token = self.separator(sep_idx)  # [1, hidden]
        sep_token = repeat(sep_token, '1 d -> b k 1 d', b=batch_size, k=num_chunks)
        
        # Combine: [fx, sep, tokens] for each chunk
        # For first chunk, use adaptive_token; for others, use fx from previous chunk
        # adaptive_tok: [b, 1, d] -> unsqueeze(1) -> [b, 1, 1, d]
        # fx[:, :-1, :]: [b, k-1, d] -> unsqueeze(2) -> [b, k-1, 1, d]
        fx_shifted = torch.cat([adaptive_tok.unsqueeze(1), fx[:, :-1, :].unsqueeze(2)], dim=1)  # [batch, num_chunks, 1, hidden]
        
        # Build sequence: [fx, sep, tokens] per chunk
        decoder_input = torch.cat([fx_shifted, sep_token, token_embeds], dim=2)  # [batch, num_chunks, 2+chunk_size, hidden]
        decoder_input = rearrange(decoder_input, 'b k l d -> b (k l) d')  # [batch, num_chunks*(2+chunk_size), hidden]
        
        # Add final fx and separator for last token's prediction
        last_sep = repeat(self.separator(sep_idx), '1 d -> b 1 d', b=batch_size)
        decoder_input = torch.cat([decoder_input, fx_last, last_sep], dim=1)
        
        hidden_states = decoder_input
        total_seq_len = hidden_states.shape[1]
        
        # Build RoPE cache for CAT positions
        cos, sin = self._build_cat_rope_cache(num_chunks, chunk_size, device, hidden_states.dtype)
        
        # Build CAT attention mask using flex_attention
        block_size = 2 + chunk_size
        block_mask = create_block_mask(
            get_cat_mask_mod(block_size),
            B=None, H=None,
            Q_LEN=total_seq_len,
            KV_LEN=total_seq_len,
        )
        
        all_hidden_states = () if output_hidden_states else None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            hidden_states = layer(
                hidden_states,
                cos=cos,
                sin=sin,
                block_mask=block_mask,
                **kwargs,
            )
        
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # Rearrange output back to [batch, seq_len, hidden]
        # Extract the last token position from each chunk for next-token prediction
        block_size = 2 + chunk_size
        
        # Get the prediction for the last token (from final separator position)
        last_pred = hidden_states[:, -1:, :]  # [batch, 1, hidden]
        
        # Get predictions from within chunks
        # The structure is: [fx, sep, tok_0, tok_1, ..., tok_{l-1}] per chunk
        # We want predictions at positions: sep (predicts tok_0), tok_0 (predicts tok_1), ..., tok_{l-2} (predicts tok_{l-1})
        # And from the last fx+sep we get prediction for first token of next chunk
        
        hidden_states_chunks = rearrange(
            hidden_states[:, :-2, :],  # Remove final fx and sep
            'b (k l) d -> b k l d',
            k=num_chunks,
            l=block_size,
        )
        
        # For first chunk: need predictions from positions [fx, sep, tok_0, ..., tok_{l-2}] (predict tok_0 to tok_{l-1})
        # Actually we need: position sep predicts tok_0, tok_0 predicts tok_1, ..., tok_{l-2} predicts tok_{l-1}
        # So we take positions 1 (sep) through l-1 (tok_{l-2}) = positions [1, l-1]
        
        # For simplicity, let's restructure:
        # Each chunk contributes predictions for its tokens
        # Position fx_{i} (at start of chunk i+1) predicts tok_0 of chunk i+1
        # Position sep predicts tok_0, tok_0 predicts tok_1, etc.
        
        # Take positions [1, 2, ..., l] from each chunk (sep through last token)
        # But the last token's prediction is for the next chunk's first token
        first_chunk_preds = hidden_states_chunks[:, 0, 1:-1, :]  # [batch, chunk_size-1, hidden]
        middle_chunk_preds = hidden_states_chunks[:, 1:, :-1, :]  # [batch, num_chunks-1, chunk_size+1, hidden]
        middle_chunk_preds = rearrange(middle_chunk_preds, 'b k l d -> b (k l) d')
        
        # Combine predictions
        output_hidden = torch.cat([first_chunk_preds, middle_chunk_preds, last_pred], dim=1)
        
        # Trim to original sequence length
        output_hidden = output_hidden[:, :original_seq_len, :]
        
        if not return_dict:
            return tuple(v for v in [output_hidden, None, all_hidden_states, None] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=output_hidden,
            past_key_values=None,  # CAT doesn't use standard KV cache
            hidden_states=all_hidden_states,
            attentions=None,
        )


class CATForCausalLM(CATPreTrainedModel, FLAGenerationMixin):
    """CAT model with a language modeling head for causal LM."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: CATConfig):
        super().__init__(config)
        self.model = CATModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = 0,
        chunk_size: int | None = None,
        **kwargs: Unpack[Any],
    ) -> tuple | CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            chunk_size=chunk_size,
            **kwargs,
        )

        hidden_states = outputs[0]

        logits = None if self.config.fuse_linear_cross_entropy else self.lm_head(
            hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:]
        )

        loss = None
        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss()
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            
            if self.config.fuse_linear_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
