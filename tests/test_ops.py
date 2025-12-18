# SPDX-License-Identifier: Apache-2.0
"""Tests for Metal operations."""

import pytest
import numpy as np


@pytest.mark.mlx
class TestNormalization:
    """Tests for normalization operations."""

    def test_rms_norm(self, torch_device):
        """Test RMS normalization."""
        import torch
        from vllm_metal.mlx import mlx_rms_norm, to_mlx, to_torch

        hidden_size = 64
        batch_size = 4

        # Create inputs
        input_tensor = torch.randn(batch_size, hidden_size, dtype=torch.float32)
        weight = torch.ones(hidden_size, dtype=torch.float32)

        # Convert and compute
        input_mlx = to_mlx(input_tensor)
        weight_mlx = to_mlx(weight)
        output_mlx = mlx_rms_norm(input_mlx, weight_mlx, eps=1e-6)
        output = to_torch(output_mlx, device="cpu", dtype=torch.float32)

        # Verify shape
        assert output.shape == input_tensor.shape

        # Verify normalization (output should have ~unit variance)
        # This is approximate due to the RMS norm formula
        variance = output.var(dim=-1)
        assert torch.allclose(variance, torch.ones_like(variance), atol=0.5)

    def test_fused_add_rms_norm(self, torch_device):
        """Test fused residual add + RMS normalization."""
        import torch
        from vllm_metal.mlx import mlx_fused_add_rms_norm, to_mlx, to_torch

        hidden_size = 64
        batch_size = 4

        # Create inputs
        input_tensor = torch.randn(batch_size, hidden_size, dtype=torch.float32)
        residual = torch.randn(batch_size, hidden_size, dtype=torch.float32)
        weight = torch.ones(hidden_size, dtype=torch.float32)

        # Convert and compute
        input_mlx = to_mlx(input_tensor)
        residual_mlx = to_mlx(residual)
        weight_mlx = to_mlx(weight)

        output_mlx, new_residual_mlx = mlx_fused_add_rms_norm(
            input_mlx, residual_mlx, weight_mlx, eps=1e-6
        )

        output = to_torch(output_mlx, device="cpu", dtype=torch.float32)
        new_residual = to_torch(new_residual_mlx, device="cpu", dtype=torch.float32)

        # Verify shapes
        assert output.shape == input_tensor.shape
        assert new_residual.shape == residual.shape

        # Verify residual is updated
        expected_residual = input_tensor + residual
        np.testing.assert_allclose(
            new_residual.numpy(),
            expected_residual.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.mlx
class TestActivations:
    """Tests for activation operations."""

    def test_silu_and_mul(self, torch_device):
        """Test SiLU activation with gated multiplication."""
        import torch
        from vllm_metal.ops import silu_and_mul

        hidden_size = 64
        batch_size = 4

        # Create input [batch, 2 * hidden]
        input_tensor = torch.randn(
            batch_size, 2 * hidden_size,
            device=torch_device, dtype=torch.float32
        )
        output = torch.zeros(
            batch_size, hidden_size,
            device=torch_device, dtype=torch.float32
        )

        silu_and_mul(output, input_tensor)

        # Verify shape
        assert output.shape == (batch_size, hidden_size)

        # Verify output is not all zeros
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_gelu_and_mul(self, torch_device):
        """Test GELU activation with gated multiplication."""
        import torch
        from vllm_metal.ops import gelu_and_mul

        hidden_size = 64
        batch_size = 4

        input_tensor = torch.randn(
            batch_size, 2 * hidden_size,
            device=torch_device, dtype=torch.float32
        )
        output = torch.zeros(
            batch_size, hidden_size,
            device=torch_device, dtype=torch.float32
        )

        gelu_and_mul(output, input_tensor)

        assert output.shape == (batch_size, hidden_size)
        assert not torch.allclose(output, torch.zeros_like(output))


@pytest.mark.mlx
class TestCache:
    """Tests for cache operations."""

    def test_reshape_and_cache(self, torch_device, kv_cache_tensors):
        """Test reshape_and_cache operation."""
        import torch
        from vllm_metal.ops import reshape_and_cache

        key_cache = kv_cache_tensors["key_cache"]
        value_cache = kv_cache_tensors["value_cache"]
        num_kv_heads = kv_cache_tensors["num_kv_heads"]
        head_size = kv_cache_tensors["head_size"]
        block_size = kv_cache_tensors["block_size"]

        # Create key/value to store
        num_tokens = 5
        key = torch.randn(
            num_tokens, num_kv_heads, head_size,
            device=torch_device, dtype=torch.float16
        )
        value = torch.randn(
            num_tokens, num_kv_heads, head_size,
            device=torch_device, dtype=torch.float16
        )

        # Create slot mapping (store in first block)
        slot_mapping = torch.arange(num_tokens, device=torch_device, dtype=torch.int64)

        reshape_and_cache(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
        )

        # Verify data was stored
        for i in range(num_tokens):
            np.testing.assert_allclose(
                key_cache[0, i].cpu().numpy(),
                key[i].cpu().numpy(),
                rtol=1e-3,
                atol=1e-3,
            )

    def test_copy_blocks(self, torch_device, kv_cache_tensors):
        """Test copy_blocks operation."""
        import torch
        from vllm_metal.ops import copy_blocks

        key_cache = kv_cache_tensors["key_cache"]
        value_cache = kv_cache_tensors["value_cache"]

        # Initialize source block with data
        key_cache[0] = torch.randn_like(key_cache[0])
        value_cache[0] = torch.randn_like(value_cache[0])

        # Copy block 0 to block 1
        block_mapping = torch.tensor([[0, 1]], device=torch_device, dtype=torch.int64)

        copy_blocks([key_cache], [value_cache], block_mapping)

        # Verify copy
        np.testing.assert_allclose(
            key_cache[0].cpu().numpy(),
            key_cache[1].cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.mlx
class TestRotary:
    """Tests for rotary embedding operations."""

    def test_mlx_rotary_embedding(self):
        """Test MLX rotary embedding."""
        import torch
        import mlx.core as mx
        from vllm_metal.mlx import mlx_rotary_embedding

        seq_len = 16
        num_heads = 8
        head_dim = 64

        # Create input
        x = mx.random.normal(shape=(seq_len, num_heads, head_dim))

        # Create cos/sin (half of head_dim)
        rotary_dim = head_dim // 2
        cos = mx.random.normal(shape=(seq_len, rotary_dim))
        sin = mx.random.normal(shape=(seq_len, rotary_dim))

        # Expand to full rotary_dim
        cos_full = mx.concatenate([cos, cos], axis=-1)
        sin_full = mx.concatenate([sin, sin], axis=-1)

        # Apply rotary embedding
        output = mlx_rotary_embedding(x, cos_full, sin_full, is_neox_style=True)

        assert output.shape == x.shape
