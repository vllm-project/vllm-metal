# SPDX-License-Identifier: Apache-2.0
"""Metal rotary embedding operations."""

from typing import Optional, Tuple

import torch

from vllm_metal.mlx import mlx_rotary_embedding, to_mlx, to_torch


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings using MLX.

    Args:
        positions: Position indices [batch_size, seq_len] or [seq_len].
        query: Query tensor [batch, seq_len, num_heads, head_size].
        key: Key tensor [batch, seq_len, num_kv_heads, head_size].
        head_size: Size of each attention head.
        cos_sin_cache: Precomputed cos/sin cache [max_seq_len, rotary_dim].
        is_neox: Whether to use NeoX-style rotation.

    Returns:
        Tuple of (rotated_query, rotated_key).
    """
    import mlx.core as mx

    # Extract cos and sin from cache
    # cos_sin_cache shape: [max_seq_len, rotary_dim] where rotary_dim = 2 * head_size // 2
    rotary_dim = cos_sin_cache.shape[-1] // 2

    # Get positions for indexing
    if positions.dim() == 1:
        pos_indices = positions
    else:
        pos_indices = positions.flatten()

    # Get cos/sin values for these positions
    cos_sin = cos_sin_cache[pos_indices.long()]
    cos = cos_sin[:, :rotary_dim]
    sin = cos_sin[:, rotary_dim:]

    # Convert to MLX
    q_mlx = to_mlx(query)
    k_mlx = to_mlx(key)
    cos_mlx = to_mlx(cos)
    sin_mlx = to_mlx(sin)

    # Reshape cos/sin to match query shape
    # Original shape: [num_tokens, rotary_dim]
    # Need: [num_tokens, 1, rotary_dim] or broadcastable shape
    if query.dim() == 4:
        # [batch, seq_len, num_heads, head_size]
        cos_mlx = cos_mlx.reshape(-1, 1, rotary_dim)
        sin_mlx = sin_mlx.reshape(-1, 1, rotary_dim)
    else:
        # [total_tokens, num_heads, head_size]
        cos_mlx = cos_mlx.reshape(-1, 1, rotary_dim)
        sin_mlx = sin_mlx.reshape(-1, 1, rotary_dim)

    # Expand cos/sin to full rotary_dim for the rotation
    cos_full = mx.concatenate([cos_mlx, cos_mlx], axis=-1)
    sin_full = mx.concatenate([sin_mlx, sin_mlx], axis=-1)

    # Apply RoPE
    rotated_q = mlx_rotary_embedding(q_mlx, cos_full, sin_full, is_neox)
    rotated_k = mlx_rotary_embedding(k_mlx, cos_full, sin_full, is_neox)

    # Convert back to PyTorch
    rotated_query = to_torch(rotated_q, device=query.device, dtype=query.dtype)
    rotated_key = to_torch(rotated_k, device=key.device, dtype=key.dtype)

    return rotated_query, rotated_key


def rotary_embedding_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> None:
    """Apply rotary embeddings in-place.

    Args:
        positions: Position indices.
        query: Query tensor (modified in-place).
        key: Key tensor (modified in-place).
        head_size: Size of each head.
        cos_sin_cache: Precomputed cos/sin cache.
        is_neox: Whether to use NeoX-style rotation.
    """
    rotated_q, rotated_k = rotary_embedding(
        positions, query, key, head_size, cos_sin_cache, is_neox
    )
    query.copy_(rotated_q)
    key.copy_(rotated_k)
