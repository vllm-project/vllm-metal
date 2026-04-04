# SPDX-License-Identifier: Apache-2.0
"""Test Metal platform's page size unification for hybrid models.

This test verifies that update_block_size_for_backend() properly aligns
block_size and sets mamba_page_size_padded to ensure vLLM's
unify_kv_cache_spec_page_size() can unify page sizes between
FullAttentionSpec and MambaSpec layers.
"""

import pytest

from vllm.config import CacheConfig, ModelConfig, VllmConfig


@pytest.fixture
def hybrid_vllm_config():
    """Create a VllmConfig for Qwen3.5 hybrid model."""
    model_config = ModelConfig(
        model="Qwen/Qwen3.5-0.8B",
        tokenizer="Qwen/Qwen3.5-0.8B",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="auto",
        seed=0,
        max_model_len=65534,
        enforce_eager=True,
    )

    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
    )

    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
    )


def test_update_block_size_for_backend_increases_block_size(hybrid_vllm_config):
    """Test that update_block_size_for_backend() increases block_size for hybrid models."""
    from vllm.platforms import current_platform

    original_block_size = hybrid_vllm_config.cache_config.block_size
    assert original_block_size == 16

    # Call the method
    current_platform.update_block_size_for_backend(hybrid_vllm_config)

    # Block size should be increased
    new_block_size = hybrid_vllm_config.cache_config.block_size
    assert new_block_size > original_block_size, (
        f"block_size should increase from {original_block_size}, got {new_block_size}"
    )

    # Block size should be aligned to 32 for performance
    assert new_block_size % 32 == 0, (
        f"block_size {new_block_size} should be aligned to 32"
    )


def test_update_block_size_for_backend_sets_mamba_padding(hybrid_vllm_config):
    """Test that update_block_size_for_backend() sets mamba_page_size_padded."""
    from vllm.platforms import current_platform
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    # Call the method
    current_platform.update_block_size_for_backend(hybrid_vllm_config)

    cache_config = hybrid_vllm_config.cache_config
    model_config = hybrid_vllm_config.model_config

    # mamba_page_size_padded should be set
    assert cache_config.mamba_page_size_padded is not None, (
        "mamba_page_size_padded should be set"
    )

    # Compute SDPA page_size with updated block_size
    attn_page_size = FullAttentionSpec(
        block_size=cache_config.block_size,
        num_kv_heads=model_config.get_num_kv_heads(
            hybrid_vllm_config.parallel_config
        ),
        head_size=model_config.get_head_size(),
        dtype=model_config.dtype,
    ).page_size_bytes

    # mamba_page_size_padded should equal SDPA page_size
    assert cache_config.mamba_page_size_padded == attn_page_size, (
        f"mamba_page_size_padded {cache_config.mamba_page_size_padded} "
        f"should equal SDPA page_size {attn_page_size}"
    )


def test_update_block_size_for_backend_page_size_divisibility(hybrid_vllm_config):
    """Test that page sizes are divisible after update_block_size_for_backend().

    This is the key test that verifies vLLM's unify_kv_cache_spec_page_size()
    will succeed.
    """
    from vllm.platforms import current_platform
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    # Call the method
    current_platform.update_block_size_for_backend(hybrid_vllm_config)

    cache_config = hybrid_vllm_config.cache_config
    model_config = hybrid_vllm_config.model_config

    # Compute SDPA page_size
    attn_page_size = FullAttentionSpec(
        block_size=cache_config.block_size,
        num_kv_heads=model_config.get_num_kv_heads(
            hybrid_vllm_config.parallel_config
        ),
        head_size=model_config.get_head_size(),
        dtype=model_config.dtype,
    ).page_size_bytes

    # Get MambaSpec page_size (with padding)
    # Note: We can't easily get the actual MambaSpec here, but we can verify
    # that mamba_page_size_padded is set and equals attn_page_size
    mamba_page_size = cache_config.mamba_page_size_padded

    # Verify divisibility: max_page_size % layer_page_size == 0
    max_page_size = max(attn_page_size, mamba_page_size)

    # SDPA page_size should divide max_page_size
    assert max_page_size % attn_page_size == 0, (
        f"max_page_size {max_page_size} should be divisible by "
        f"SDPA page_size {attn_page_size}"
    )

    # Mamba page_size (padded) should divide max_page_size
    assert max_page_size % mamba_page_size == 0, (
        f"max_page_size {max_page_size} should be divisible by "
        f"Mamba page_size {mamba_page_size}"
    )


def test_update_block_size_for_backend_skips_non_hybrid():
    """Test that update_block_size_for_backend() skips non-hybrid models."""
    from vllm.platforms import current_platform

    # Use a public model that doesn't require authentication
    model_config = ModelConfig(
        model="microsoft/phi-2",
        tokenizer="microsoft/phi-2",
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="auto",
        seed=0,
        max_model_len=2048,
    )

    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
    )

    original_block_size = vllm_config.cache_config.block_size

    # Call the method
    current_platform.update_block_size_for_backend(vllm_config)

    # Block size should not change for non-hybrid models
    assert vllm_config.cache_config.block_size == original_block_size, (
        "block_size should not change for non-hybrid models"
    )

    # mamba_page_size_padded should not be set
    assert vllm_config.cache_config.mamba_page_size_padded is None, (
        "mamba_page_size_padded should not be set for non-hybrid models"
    )
