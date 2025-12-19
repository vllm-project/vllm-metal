# SPDX-License-Identifier: Apache-2.0
"""Tests for MLX backend operations."""

import mlx.core as mx

from vllm_metal.mlx_backend.ops import (
    attention,
    gelu,
    rms_norm,
    rotary_embedding,
    silu_and_mul,
)


class TestRMSNorm:
    """Tests for RMS normalization."""

    def test_rms_norm_basic(self) -> None:
        """Test basic RMS normalization."""
        x = mx.array([[1.0, 2.0, 3.0, 4.0]])
        weight = mx.ones((4,))

        result = rms_norm(x, weight)
        mx.eval(result)

        # RMS = sqrt(mean([1, 4, 9, 16])) = sqrt(7.5) ≈ 2.74
        # Normalized values should sum to approximately the same
        assert result.shape == x.shape
        assert result.dtype == x.dtype

    def test_rms_norm_with_weight(self) -> None:
        """Test RMS normalization with scaling weight."""
        x = mx.array([[1.0, 2.0, 3.0, 4.0]])
        weight = mx.array([2.0, 2.0, 2.0, 2.0])

        result = rms_norm(x, weight)
        mx.eval(result)

        # Result should be scaled by 2x
        result_unscaled = rms_norm(x, mx.ones((4,)))
        mx.eval(result_unscaled)

        # Compare approximately
        assert mx.allclose(result, result_unscaled * 2.0, atol=1e-5)

    def test_rms_norm_batch(self) -> None:
        """Test RMS normalization with batch dimension."""
        batch_size = 4
        hidden_size = 8
        x = mx.random.normal((batch_size, hidden_size))
        weight = mx.ones((hidden_size,))

        result = rms_norm(x, weight)
        mx.eval(result)

        assert result.shape == (batch_size, hidden_size)


class TestAttention:
    """Tests for attention operations."""

    def test_attention_basic(self) -> None:
        """Test basic self-attention."""
        batch = 1
        seq_len = 4
        num_heads = 2
        head_dim = 8

        q = mx.random.normal((batch, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch, seq_len, num_heads, head_dim))
        v = mx.random.normal((batch, seq_len, num_heads, head_dim))

        result = attention(q, k, v)
        mx.eval(result)

        assert result.shape == (batch, seq_len, num_heads, head_dim)

    def test_attention_with_mask(self) -> None:
        """Test attention with causal mask."""
        batch = 1
        seq_len = 4
        num_heads = 2
        head_dim = 8

        q = mx.random.normal((batch, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch, seq_len, num_heads, head_dim))
        v = mx.random.normal((batch, seq_len, num_heads, head_dim))

        # Create causal mask
        mask = mx.triu(mx.full((seq_len, seq_len), -1e9), k=1)
        mask = mask[None, None, :, :]  # (1, 1, seq_len, seq_len)

        result = attention(q, k, v, mask=mask)
        mx.eval(result)

        assert result.shape == (batch, seq_len, num_heads, head_dim)

    def test_attention_gqa(self) -> None:
        """Test grouped-query attention."""
        batch = 1
        seq_len = 4
        num_heads = 8
        num_kv_heads = 2
        head_dim = 8

        q = mx.random.normal((batch, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch, seq_len, num_kv_heads, head_dim))
        v = mx.random.normal((batch, seq_len, num_kv_heads, head_dim))

        result = attention(q, k, v)
        mx.eval(result)

        # Output should have same shape as queries
        assert result.shape == (batch, seq_len, num_heads, head_dim)


class TestRotaryEmbedding:
    """Tests for rotary position embeddings."""

    def test_rotary_embedding_basic(self) -> None:
        """Test basic RoPE application."""
        batch = 1
        seq_len = 4
        num_heads = 2
        head_dim = 8

        q = mx.random.normal((batch, seq_len, num_heads, head_dim))
        k = mx.random.normal((batch, seq_len, num_heads, head_dim))
        positions = mx.arange(seq_len)[None, :]

        q_rot, k_rot = rotary_embedding(q, k, positions, head_dim)
        mx.eval(q_rot)
        mx.eval(k_rot)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rotary_embedding_different_positions(self) -> None:
        """Test that different positions give different embeddings."""
        batch = 1
        seq_len = 1
        num_heads = 2
        head_dim = 8

        q = mx.ones((batch, seq_len, num_heads, head_dim))
        k = mx.ones((batch, seq_len, num_heads, head_dim))

        pos0 = mx.array([[0]])
        pos1 = mx.array([[1]])

        q0, k0 = rotary_embedding(q, k, pos0, head_dim)
        q1, k1 = rotary_embedding(q, k, pos1, head_dim)
        mx.eval(q0, q1)

        # Different positions should give different results
        assert not mx.allclose(q0, q1, atol=1e-5)


class TestActivations:
    """Tests for activation functions."""

    def test_silu_and_mul(self) -> None:
        """Test SiLU activation with gating."""
        hidden_dim = 8
        x = mx.random.normal((2, hidden_dim * 2))

        result = silu_and_mul(x)
        mx.eval(result)

        assert result.shape == (2, hidden_dim)

    def test_gelu_approximate(self) -> None:
        """Test approximate GELU activation."""
        x = mx.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        result = gelu(x, approximate=True)
        mx.eval(result)

        # GELU(0) should be 0
        assert abs(float(result[2].item())) < 1e-5
        # GELU is approximately x for large positive x
        assert float(result[4].item()) > 1.9

    def test_gelu_exact(self) -> None:
        """Test exact GELU activation."""
        x = mx.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        result = gelu(x, approximate=False)
        mx.eval(result)

        # GELU(0) should be 0
        assert abs(float(result[2].item())) < 1e-5
