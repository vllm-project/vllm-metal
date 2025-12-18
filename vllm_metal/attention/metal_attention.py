# SPDX-License-Identifier: Apache-2.0
"""Metal attention implementation using MLX."""

from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from vllm_metal._compat import AttentionImpl, AttentionType, init_logger
from vllm_metal.attention.backend import MetalAttentionMetadata
from vllm_metal.mlx import (
    TensorBridge,
    mlx_paged_attention,
    mlx_scaled_dot_product_attention,
    to_mlx,
    to_torch,
)

logger = init_logger(__name__)


class MetalAttentionImpl(AttentionImpl):
    """Metal attention implementation using MLX.

    This implementation provides efficient attention computation on
    Apple Silicon by using MLX's optimized SDPA. It handles both
    prefill (variable length) and decode (single token) phases.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = "decoder",
    ):
        """Initialize Metal attention.

        Args:
            num_heads: Number of attention heads.
            head_size: Size of each head.
            scale: Attention scale factor.
            num_kv_heads: Number of key-value heads (for GQA).
            alibi_slopes: Optional ALiBi slopes.
            sliding_window: Optional sliding window size.
            kv_cache_dtype: Data type for KV cache.
            blocksparse_params: Block sparse parameters (not supported).
            logits_soft_cap: Logits soft cap (not supported).
            attn_type: Attention type (decoder, encoder, etc.).
        """
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type

        # GQA setup
        self.num_queries_per_kv = num_heads // num_kv_heads

        # Validate unsupported features
        if blocksparse_params is not None:
            logger.warning("Block sparse attention not supported on Metal")
        if logits_soft_cap is not None:
            logger.warning("Logits soft cap not supported on Metal")
        if sliding_window is not None:
            logger.warning(
                "Sliding window attention has limited support on Metal"
            )

        logger.debug(
            f"MetalAttentionImpl initialized: "
            f"num_heads={num_heads}, head_size={head_size}, "
            f"num_kv_heads={num_kv_heads}, scale={scale}"
        )

    def forward(
        self,
        layer: Any,  # AttentionLayer, but we avoid import cycle
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: MetalAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for attention.

        Args:
            layer: The attention layer (unused on Metal).
            query: Query tensor [num_tokens, num_heads, head_size].
            key: Key tensor [num_tokens, num_kv_heads, head_size].
            value: Value tensor [num_tokens, num_kv_heads, head_size].
            kv_cache: Tuple of (key_cache, value_cache).
            attn_metadata: Attention metadata.
            output: Optional pre-allocated output tensor.

        Returns:
            Attention output [num_tokens, num_heads, head_size].
        """
        key_cache, value_cache = kv_cache

        # Store new K/V into cache
        if attn_metadata.slot_mapping is not None:
            self._store_kv_cache(
                key, value, key_cache, value_cache, attn_metadata.slot_mapping
            )

        # Route to appropriate attention implementation
        if attn_metadata.is_all_prefill:
            attn_output = self._prefill_attention(
                query,
                key,
                value,
                attn_metadata,
            )
        elif attn_metadata.is_all_decode:
            attn_output = self._decode_attention(
                query,
                key_cache,
                value_cache,
                attn_metadata,
            )
        else:
            # Mixed batch - handle prefill and decode separately
            attn_output = self._mixed_attention(
                query,
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata,
            )

        # Copy to output if provided
        if output is not None:
            output.copy_(attn_output)
            return output

        return attn_output

    def _store_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Store key and value tensors into the cache.

        Args:
            key: Key tensor [num_tokens, num_kv_heads, head_size].
            value: Value tensor [num_tokens, num_kv_heads, head_size].
            key_cache: Key cache [num_blocks, block_size, num_kv_heads, head_size].
            value_cache: Value cache [num_blocks, block_size, num_kv_heads, head_size].
            slot_mapping: Slot indices [num_tokens].
        """
        block_size = key_cache.shape[1]

        # Compute block indices and offsets
        block_indices = slot_mapping // block_size
        block_offsets = slot_mapping % block_size

        # Store using advanced indexing
        # This works on MPS since PyTorch handles it
        for i in range(key.shape[0]):
            block_idx = int(block_indices[i])
            offset = int(block_offsets[i])
            key_cache[block_idx, offset] = key[i]
            value_cache[block_idx, offset] = value[i]

    def _prefill_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: MetalAttentionMetadata,
    ) -> torch.Tensor:
        """Compute attention for prefill phase using MLX.

        Args:
            query: Query tensor [total_tokens, num_heads, head_size].
            key: Key tensor [total_tokens, num_kv_heads, head_size].
            value: Value tensor [total_tokens, num_kv_heads, head_size].
            attn_metadata: Attention metadata.

        Returns:
            Attention output [total_tokens, num_heads, head_size].
        """
        import mlx.core as mx

        # Convert to MLX
        q_mlx = to_mlx(query)
        k_mlx = to_mlx(key)
        v_mlx = to_mlx(value)

        # Process each sequence
        outputs = []
        seq_lens = attn_metadata.seq_lens
        query_start_loc = attn_metadata.query_start_loc

        for i in range(len(seq_lens)):
            start = int(query_start_loc[i])
            end = int(query_start_loc[i + 1])
            seq_len = int(seq_lens[i])

            if end <= start:
                continue

            # Get Q, K, V for this sequence
            q = q_mlx[start:end]  # [seq_len, num_heads, head_size]
            k = k_mlx[start:end]
            v = v_mlx[start:end]

            # Expand KV for GQA
            if self.num_kv_heads != self.num_heads:
                k = mx.repeat(k, self.num_queries_per_kv, axis=1)
                v = mx.repeat(v, self.num_queries_per_kv, axis=1)

            # Reshape for SDPA: [1, num_heads, seq_len, head_size]
            q = q.transpose(1, 0, 2)[None, ...]
            k = k.transpose(1, 0, 2)[None, ...]
            v = v.transpose(1, 0, 2)[None, ...]

            # Create causal mask
            mask = mx.triu(
                mx.full((seq_len, seq_len), float("-inf")),
                k=1,
            )

            # Compute attention
            out = mlx_scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=mask
            )

            # Reshape back: [seq_len, num_heads, head_size]
            out = out[0].transpose(1, 0, 2)
            outputs.append(out)

        # Concatenate and convert back to PyTorch
        if outputs:
            result = mx.concatenate(outputs, axis=0)
            return to_torch(result, device=query.device, dtype=query.dtype)
        else:
            return query.new_zeros(query.shape)

    def _decode_attention(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: MetalAttentionMetadata,
    ) -> torch.Tensor:
        """Compute attention for decode phase using MLX paged attention.

        Args:
            query: Query tensor [batch, num_heads, head_size].
            key_cache: Key cache [num_blocks, block_size, num_kv_heads, head_size].
            value_cache: Value cache [num_blocks, block_size, num_kv_heads, head_size].
            attn_metadata: Attention metadata.

        Returns:
            Attention output [batch, num_heads, head_size].
        """
        import mlx.core as mx

        # Convert to MLX
        q_mlx = to_mlx(query)
        k_cache_mlx = to_mlx(key_cache)
        v_cache_mlx = to_mlx(value_cache)
        block_table_mlx = to_mlx(attn_metadata.block_table)
        seq_lens_mlx = to_mlx(attn_metadata.seq_lens)

        # Add sequence dimension: [batch, num_heads, 1, head_size]
        if q_mlx.ndim == 3:
            q_mlx = q_mlx[:, :, None, :]

        # ALiBi slopes if present
        alibi_mlx = None
        if self.alibi_slopes is not None:
            alibi_mlx = mx.array(self.alibi_slopes)

        # Compute paged attention
        output = mlx_paged_attention(
            q_mlx,
            k_cache_mlx,
            v_cache_mlx,
            block_table_mlx,
            seq_lens_mlx,
            scale=self.scale,
            alibi_slopes=alibi_mlx,
        )

        # Remove sequence dimension: [batch, num_heads, head_size]
        if output.shape[2] == 1:
            output = output[:, :, 0, :]

        return to_torch(output, device=query.device, dtype=query.dtype)

    def _mixed_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: MetalAttentionMetadata,
    ) -> torch.Tensor:
        """Handle mixed prefill/decode batch.

        This is less common but can happen during continuous batching.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            key_cache: Key cache.
            value_cache: Value cache.
            attn_metadata: Attention metadata.

        Returns:
            Attention output.
        """
        # For simplicity, fall back to PyTorch SDPA for mixed batches
        # This is rare in practice
        logger.debug("Mixed batch attention - using PyTorch SDPA fallback")

        return self._pytorch_attention(query, key, value, attn_metadata)

    def _pytorch_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: MetalAttentionMetadata,
    ) -> torch.Tensor:
        """Fallback to PyTorch SDPA.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            attn_metadata: Attention metadata.

        Returns:
            Attention output.
        """
        # Reshape for PyTorch SDPA
        # [total_tokens, num_heads, head_size] -> [batch, num_heads, seq, head_size]
        # This is a simplified implementation for edge cases

        # Process per sequence
        outputs = []
        seq_lens = attn_metadata.seq_lens
        query_start_loc = attn_metadata.query_start_loc

        if query_start_loc is not None:
            for i in range(len(seq_lens)):
                start = int(query_start_loc[i])
                end = int(query_start_loc[i + 1])

                if end <= start:
                    continue

                q = query[start:end].transpose(0, 1).unsqueeze(0)
                k = key[start:end].transpose(0, 1).unsqueeze(0)
                v = value[start:end].transpose(0, 1).unsqueeze(0)

                # Expand for GQA
                if self.num_kv_heads != self.num_heads:
                    k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
                    v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, scale=self.scale, is_causal=True
                )

                out = out.squeeze(0).transpose(0, 1)
                outputs.append(out)

            if outputs:
                return torch.cat(outputs, dim=0)

        return query.new_zeros(query.shape)
