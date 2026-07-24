# SPDX-License-Identifier: Apache-2.0
"""Guard against an engine KV cache config larger than the allocated pool."""

from __future__ import annotations

import mlx.core as mx
import pytest
from vllm.v1.kv_cache_interface import KVCacheConfig

from tests.stub_runner import make_stub_runner
from vllm_metal.attention.runtime.mla import MLAPagedAttentionRuntime


def _make_policy(allocated_blocks: int | None):
    """Build a stub runner whose paged runtime reports ``allocated_blocks``."""
    runtime = None
    if allocated_blocks is not None:
        runtime = MLAPagedAttentionRuntime(
            num_layers=1,
            latent_dim=1,
            block_size=1,
            dtype=mx.bfloat16,
        )
        runtime.initialize(num_blocks=allocated_blocks)
    runner = make_stub_runner(
        model_args={"kv_lora_rank": 1},
        _paged_attention_runtime=runtime,
    )
    return runner._cache_policy


def _kv_config(num_blocks: int) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[],
    )


def test_rejects_more_blocks_than_allocated() -> None:
    """--num-gpu-blocks-override above the profiled capacity must fail fast."""
    policy = _make_policy(1846)

    with pytest.raises(ValueError, match="cannot grow it afterwards"):
        policy.initialize_kv_cache(_kv_config(999999))


def test_accepts_matching_block_count() -> None:
    """The normal round-trip, where the engine echoes the allocated count."""
    policy = _make_policy(1846)

    policy.initialize_kv_cache(_kv_config(1846))


def test_accepts_fewer_blocks_than_allocated() -> None:
    """A smaller override only wastes memory; it is not unsafe."""
    policy = _make_policy(1846)

    policy.initialize_kv_cache(_kv_config(512))


def test_noop_without_paged_runtime() -> None:
    """Non-paged paths have no pool to validate against."""
    policy = _make_policy(None)

    policy.initialize_kv_cache(_kv_config(999999))
