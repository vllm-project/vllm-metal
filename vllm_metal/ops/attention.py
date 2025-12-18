# SPDX-License-Identifier: Apache-2.0
"""Metal paged attention operations."""

from typing import Optional

import torch

from vllm_metal.mlx import (
    mlx_paged_attention,
    to_mlx,
    to_torch,
)


def paged_attention_v1(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor] = None,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Paged attention v1 using MLX.

    This is the standard paged attention used during decode phase.

    Args:
        output: Output tensor to write results [batch, num_heads, head_size].
        query: Query tensor [batch, num_heads, head_size].
        key_cache: Key cache [num_blocks, block_size, num_kv_heads, head_size].
        value_cache: Value cache [num_blocks, block_size, num_kv_heads, head_size].
        num_kv_heads: Number of key-value heads.
        scale: Attention scale factor.
        block_tables: Block table [batch, max_blocks].
        seq_lens: Sequence lengths [batch].
        block_size: Tokens per block.
        max_seq_len: Maximum sequence length.
        alibi_slopes: Optional ALiBi slopes.
        kv_cache_dtype: Cache data type.
        k_scale: Key scale factor (for FP8).
        v_scale: Value scale factor (for FP8).
    """
    import mlx.core as mx

    # Convert to MLX
    q_mlx = to_mlx(query)
    k_cache_mlx = to_mlx(key_cache)
    v_cache_mlx = to_mlx(value_cache)
    block_table_mlx = to_mlx(block_tables)
    seq_lens_mlx = to_mlx(seq_lens)

    # Add sequence dimension for SDPA: [batch, num_heads, 1, head_size]
    if q_mlx.ndim == 3:
        q_mlx = q_mlx[:, :, None, :]

    # ALiBi slopes
    alibi_mlx = None
    if alibi_slopes is not None:
        alibi_mlx = to_mlx(alibi_slopes)

    # Compute paged attention
    result = mlx_paged_attention(
        q_mlx,
        k_cache_mlx,
        v_cache_mlx,
        block_table_mlx,
        seq_lens_mlx,
        scale=scale,
        alibi_slopes=alibi_mlx,
    )

    # Remove sequence dimension
    if result.shape[2] == 1:
        result = result[:, :, 0, :]

    # Convert back and copy to output
    result_torch = to_torch(result, device=output.device, dtype=output.dtype)
    output.copy_(result_torch)


def paged_attention_v2(
    output: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor] = None,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Paged attention v2 using MLX.

    This is the split-k version for very long sequences.
    On Metal, we use the same implementation as v1 since MLX
    handles the optimization internally.

    Args:
        output: Output tensor [batch, num_heads, head_size].
        exp_sums: Intermediate exp sums (unused on Metal).
        max_logits: Intermediate max logits (unused on Metal).
        tmp_output: Temporary output (unused on Metal).
        query: Query tensor [batch, num_heads, head_size].
        key_cache: Key cache.
        value_cache: Value cache.
        num_kv_heads: Number of KV heads.
        scale: Attention scale.
        block_tables: Block table.
        seq_lens: Sequence lengths.
        block_size: Block size.
        max_seq_len: Max sequence length.
        alibi_slopes: Optional ALiBi slopes.
        kv_cache_dtype: Cache dtype.
        k_scale: Key scale.
        v_scale: Value scale.
    """
    # On Metal, v2 is the same as v1 - MLX handles optimization
    paged_attention_v1(
        output=output,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        num_kv_heads=num_kv_heads,
        scale=scale,
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=block_size,
        max_seq_len=max_seq_len,
        alibi_slopes=alibi_slopes,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
    )
