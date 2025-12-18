# SPDX-License-Identifier: Apache-2.0
"""MLX attention implementations for vLLM Metal backend."""

import mlx.core as mx
import mlx.core.fast as mx_fast

from vllm_metal.mlx.tensor_bridge import to_mlx, to_torch


def mlx_scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    """Compute scaled dot-product attention using MLX.

    This is a thin wrapper around MLX's optimized SDPA implementation
    which is highly optimized for Apple Silicon.

    Args:
        query: Query tensor of shape [batch, num_heads, seq_len, head_dim].
        key: Key tensor of shape [batch, num_kv_heads, seq_len, head_dim].
        value: Value tensor of shape [batch, num_kv_heads, seq_len, head_dim].
        scale: Attention scale factor. Defaults to 1/sqrt(head_dim).
        mask: Optional attention mask.

    Returns:
        Attention output of shape [batch, num_heads, seq_len, head_dim].
    """
    if scale is None:
        head_dim = query.shape[-1]
        scale = head_dim ** -0.5

    # MLX's fast SDPA is optimized for Apple Silicon
    output = mx_fast.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=scale,
        mask=mask,
    )

    return output


def mlx_paged_attention(
    query: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    block_table: mx.array,
    seq_lens: mx.array,
    scale: float | None = None,
    alibi_slopes: mx.array | None = None,
) -> mx.array:
    """Compute paged attention using MLX.

    This implements paged attention for efficient KV cache management
    during decoding. It gathers keys and values from the cache based
    on the block table, then computes attention.

    Args:
        query: Query tensor [batch, num_heads, 1, head_dim] (decode step).
        key_cache: Key cache [num_blocks, block_size, num_kv_heads, head_dim].
        value_cache: Value cache [num_blocks, block_size, num_kv_heads, head_dim].
        block_table: Block table mapping [batch, max_blocks].
        seq_lens: Sequence lengths [batch].
        scale: Attention scale factor.
        alibi_slopes: Optional ALiBi slopes [num_heads].

    Returns:
        Attention output [batch, num_heads, 1, head_dim].
    """
    batch_size = query.shape[0]
    num_heads = query.shape[1]
    head_dim = query.shape[-1]
    num_kv_heads = key_cache.shape[2]
    block_size = key_cache.shape[1]

    if scale is None:
        scale = head_dim ** -0.5

    # Process each sequence in the batch
    outputs = []
    for i in range(batch_size):
        seq_len = int(seq_lens[i])
        num_blocks_needed = (seq_len + block_size - 1) // block_size

        # Get block indices for this sequence
        blocks = block_table[i, :num_blocks_needed]

        # Gather keys and values from cache
        # Shape: [num_blocks_needed, block_size, num_kv_heads, head_dim]
        k_blocks = key_cache[blocks]
        v_blocks = value_cache[blocks]

        # Reshape to [seq_len, num_kv_heads, head_dim]
        k_flat = k_blocks.reshape(-1, num_kv_heads, head_dim)[:seq_len]
        v_flat = v_blocks.reshape(-1, num_kv_heads, head_dim)[:seq_len]

        # Expand for GQA if needed (repeat KV heads)
        if num_kv_heads != num_heads:
            repeats = num_heads // num_kv_heads
            k_flat = mx.repeat(k_flat, repeats, axis=1)
            v_flat = mx.repeat(v_flat, repeats, axis=1)

        # Reshape for attention: [1, num_heads, seq_len, head_dim]
        k = k_flat.transpose(1, 0, 2)[None, ...]  # [1, num_heads, seq_len, head_dim]
        v = v_flat.transpose(1, 0, 2)[None, ...]

        # Query: [1, num_heads, 1, head_dim]
        q = query[i : i + 1]

        # Compute attention scores: [1, num_heads, 1, seq_len]
        attn_weights = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale

        # Apply ALiBi if provided
        if alibi_slopes is not None:
            # Create position bias: [num_heads, 1, seq_len]
            positions = mx.arange(seq_len)[None, None, :]
            alibi_bias = alibi_slopes[:, None, None] * (positions - seq_len + 1)
            attn_weights = attn_weights + alibi_bias[None, ...]

        # Softmax
        attn_weights = mx.softmax(attn_weights, axis=-1)

        # Apply attention: [1, num_heads, 1, head_dim]
        output = mx.matmul(attn_weights, v)
        outputs.append(output)

    # Stack batch results
    return mx.concatenate(outputs, axis=0)


def mlx_prefill_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    seq_lens: mx.array,
    scale: float | None = None,
    is_causal: bool = True,
) -> mx.array:
    """Compute prefill attention using MLX.

    For prefill, we have variable length sequences and need to handle
    them efficiently. MLX's SDPA handles this natively.

    Args:
        query: Query tensor [total_tokens, num_heads, head_dim].
        key: Key tensor [total_tokens, num_kv_heads, head_dim].
        value: Value tensor [total_tokens, num_kv_heads, head_dim].
        seq_lens: Sequence lengths for each sequence in the batch.
        scale: Attention scale factor.
        is_causal: Whether to use causal masking.

    Returns:
        Attention output [total_tokens, num_heads, head_dim].
    """
    num_heads = query.shape[1]
    head_dim = query.shape[-1]
    num_kv_heads = key.shape[1]

    if scale is None:
        scale = head_dim ** -0.5

    # Process each sequence
    outputs = []
    start_idx = 0

    for seq_len in seq_lens:
        seq_len = int(seq_len)
        end_idx = start_idx + seq_len

        # Get this sequence's Q, K, V
        q = query[start_idx:end_idx]  # [seq_len, num_heads, head_dim]
        k = key[start_idx:end_idx]  # [seq_len, num_kv_heads, head_dim]
        v = value[start_idx:end_idx]

        # Expand KV for GQA
        if num_kv_heads != num_heads:
            repeats = num_heads // num_kv_heads
            k = mx.repeat(k, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)

        # Reshape for SDPA: [1, num_heads, seq_len, head_dim]
        q = q.transpose(1, 0, 2)[None, ...]
        k = k.transpose(1, 0, 2)[None, ...]
        v = v.transpose(1, 0, 2)[None, ...]

        # Create causal mask if needed
        mask = None
        if is_causal:
            mask = mx.triu(
                mx.full((seq_len, seq_len), float("-inf")),
                k=1,
            )

        # Compute attention
        out = mx_fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

        # Reshape back: [seq_len, num_heads, head_dim]
        out = out[0].transpose(1, 0, 2)
        outputs.append(out)

        start_idx = end_idx

    return mx.concatenate(outputs, axis=0)
