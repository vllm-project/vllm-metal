# SPDX-License-Identifier: Apache-2.0
"""Tests for Metal attention backend."""

import pytest
import numpy as np


@pytest.mark.mlx
class TestMetalAttention:
    """Tests for Metal attention implementation."""

    def test_attention_backend_name(self):
        """Test attention backend name."""
        from vllm_metal.attention import MetalAttentionBackend

        assert MetalAttentionBackend.get_name() == "metal"

    def test_attention_impl_class(self):
        """Test attention implementation class."""
        from vllm_metal.attention import MetalAttentionBackend, MetalAttentionImpl

        impl_cls = MetalAttentionBackend.get_impl_cls()
        assert impl_cls == MetalAttentionImpl

    def test_kv_cache_shape(self):
        """Test KV cache shape computation."""
        from vllm_metal.attention import MetalAttentionBackend

        shape = MetalAttentionBackend.get_kv_cache_shape(
            num_blocks=32,
            block_size=16,
            num_kv_heads=8,
            head_size=64,
        )

        assert shape == (32, 16, 8, 64)

    def test_mlx_sdpa(self, sample_tensors):
        """Test MLX scaled dot-product attention."""
        import mlx.core as mx
        from vllm_metal.mlx import mlx_scaled_dot_product_attention, to_mlx, to_torch

        query = sample_tensors["query"]
        key = sample_tensors["key"]
        value = sample_tensors["value"]

        # Convert to MLX
        q_mlx = to_mlx(query.cpu())
        k_mlx = to_mlx(key.cpu())
        v_mlx = to_mlx(value.cpu())

        # Reshape for SDPA: [batch, num_heads, seq_len, head_dim]
        q_mlx = q_mlx.transpose(0, 2, 1, 3)
        k_mlx = k_mlx.transpose(0, 2, 1, 3)
        v_mlx = v_mlx.transpose(0, 2, 1, 3)

        # Compute attention
        output = mlx_scaled_dot_product_attention(q_mlx, k_mlx, v_mlx)

        assert output.shape == q_mlx.shape

    def test_attention_impl_initialization(self):
        """Test MetalAttentionImpl initialization."""
        from vllm_metal.attention import MetalAttentionImpl

        impl = MetalAttentionImpl(
            num_heads=8,
            head_size=64,
            scale=1.0 / (64 ** 0.5),
            num_kv_heads=8,
        )

        assert impl.num_heads == 8
        assert impl.head_size == 64
        assert impl.num_kv_heads == 8
        assert impl.num_queries_per_kv == 1

    def test_attention_impl_gqa(self):
        """Test MetalAttentionImpl with GQA."""
        from vllm_metal.attention import MetalAttentionImpl

        impl = MetalAttentionImpl(
            num_heads=16,
            head_size=64,
            scale=1.0 / (64 ** 0.5),
            num_kv_heads=4,  # GQA: 4 groups of 4 heads
        )

        assert impl.num_heads == 16
        assert impl.num_kv_heads == 4
        assert impl.num_queries_per_kv == 4


@pytest.mark.mlx
class TestPagedAttention:
    """Tests for paged attention."""

    def test_paged_attention_v1(self, torch_device, kv_cache_tensors):
        """Test paged attention v1."""
        import torch
        from vllm_metal.ops import paged_attention_v1

        batch_size = 2
        num_heads = 8
        head_size = 64
        seq_lens = [10, 15]

        # Create query
        query = torch.randn(
            batch_size, num_heads, head_size,
            device=torch_device, dtype=torch.float16
        )

        # Create output tensor
        output = torch.zeros_like(query)

        # Create block table
        max_blocks = 4
        block_tables = torch.zeros(
            batch_size, max_blocks,
            device=torch_device, dtype=torch.int32
        )
        block_tables[0, :2] = torch.tensor([0, 1])
        block_tables[1, :2] = torch.tensor([2, 3])

        seq_lens_tensor = torch.tensor(seq_lens, device=torch_device, dtype=torch.int32)

        # Populate some cache data
        key_cache = kv_cache_tensors["key_cache"]
        value_cache = kv_cache_tensors["value_cache"]

        # Just verify it runs without error
        paged_attention_v1(
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=8,
            scale=1.0 / (64 ** 0.5),
            block_tables=block_tables,
            seq_lens=seq_lens_tensor,
            block_size=16,
            max_seq_len=64,
        )

        # Output should have changed
        assert not torch.allclose(output, torch.zeros_like(output))
