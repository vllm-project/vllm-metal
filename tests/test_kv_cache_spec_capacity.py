# SPDX-License-Identifier: Apache-2.0
"""The scheduler-visible spec must describe the pool that was allocated."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import torch
from vllm.v1.core.kv_cache_utils import get_uniform_page_size
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec

from tests.stub_runner import make_stub_runner

BLOCK_SIZE = 16
DTYPE = torch.bfloat16
POOL_BLOCKS = 1000


def _engine_blocks(specs: dict, metal_per_block: int) -> int:
    """Replicate vLLM's sizing: available_memory // page_size // group_size."""
    available = POOL_BLOCKS * metal_per_block
    return available // get_uniform_page_size(list(specs.values())) // len(specs)


def _policy(*, num_layers, is_mla=False, yoco=None, num_kv_heads=4, head_dim=256):
    # ``is_mla``/``is_hybrid`` are derived from the model config, not settable.
    model_args = {"kv_lora_rank": head_dim} if is_mla else {}
    runner = make_stub_runner(
        model_args=model_args,
        num_layers=num_layers,
        _yoco_cache_mapping=yoco,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_cache_dtype=mx.bfloat16,
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
        kv_heads_per_layer=None,
        head_dim_per_layer=None,
    )
    return runner._cache_policy


def test_mla_spec_matches_single_latent_cache() -> None:
    """MLA stores one latent tensor, so its spec must not bill K and V."""
    layers, latent = 61, 576
    policy = _policy(num_layers=layers, is_mla=True, head_dim=latent, num_kv_heads=1)
    specs = policy.get_kv_cache_spec()

    assert all(isinstance(s, MLAAttentionSpec) for s in specs.values())

    # _kv_factor() == 1 for MLA
    metal_per_block = 1 * BLOCK_SIZE * DTYPE.itemsize * layers * 1 * latent
    assert _engine_blocks(specs, metal_per_block) == POOL_BLOCKS


def test_yoco_shared_layers_get_no_spec() -> None:
    """Only the layers that own a cache may appear in the spec."""
    layers, owned = 35, 15
    policy = _policy(num_layers=layers, yoco=(owned, list(range(owned))))
    specs = policy.get_kv_cache_spec()

    assert len(specs) == owned

    metal_per_block = 2 * BLOCK_SIZE * DTYPE.itemsize * owned * 4 * 256
    assert _engine_blocks(specs, metal_per_block) == POOL_BLOCKS


def test_plain_attention_spec_is_unchanged() -> None:
    """Non-MLA, non-YOCO models keep emitting one FullAttentionSpec per layer."""
    layers = 36
    policy = _policy(num_layers=layers)
    specs = policy.get_kv_cache_spec()

    assert len(specs) == layers
    assert all(type(s) is FullAttentionSpec for s in specs.values())

    metal_per_block = 2 * BLOCK_SIZE * DTYPE.itemsize * layers * 4 * 256
    assert _engine_blocks(specs, metal_per_block) == POOL_BLOCKS
