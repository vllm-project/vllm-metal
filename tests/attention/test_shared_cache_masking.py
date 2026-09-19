# SPDX-License-Identifier: Apache-2.0
"""Unread shared-page NaNs must not leak into production attention outputs.

A page reassigned from recurrent state can contain arbitrary floating-point
bit patterns. Unlike an initially zeroed KV pool, an unwritten tail can contain
NaNs; masking its attention weight to zero alone is insufficient for values.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)

from vllm_metal.attention.attention_contracts import AttentionContract
from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.impls.sdpa import _pick_kernel_block_size, sdpa_forward
from vllm_metal.metal import get_ops

_NUM_BLOCKS = 5
_KV_HEADS = 2
_QUERY_HEADS = 4
_HEAD_DIM = 64
_LAYER = 1
_ATTENTION_NAMES = ("attention.0", "attention.1")
_STATE_NAMES = ("state.0", "state.1")
_BLOCK_TABLE = [3, 1]
_TAIL_TOKENS = 13


class _SliceAttention(nn.Module):
    """Expose supplied Q/K/V through real projection and SDPA preparation code."""

    n_heads = _QUERY_HEADS
    n_kv_heads = _KV_HEADS
    head_dim = _HEAD_DIM
    scale = _HEAD_DIM**-0.5

    def q_proj(self, x: mx.array) -> mx.array:
        return x[..., : _QUERY_HEADS * _HEAD_DIM]

    def k_proj(self, x: mx.array) -> mx.array:
        start = _QUERY_HEADS * _HEAD_DIM
        return x[..., start : start + _KV_HEADS * _HEAD_DIM]

    def v_proj(self, x: mx.array) -> mx.array:
        return x[..., (_QUERY_HEADS + _KV_HEADS) * _HEAD_DIM :]

    def o_proj(self, x: mx.array) -> mx.array:
        return x


def _config(block_size: int, dtype: mx.Dtype) -> KVCacheConfig:
    torch_dtype = torch.float16 if dtype == mx.float16 else torch.bfloat16
    attention = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=_KV_HEADS,
        head_size=_HEAD_DIM,
        dtype=torch_dtype,
    )
    state = MambaSpec(
        block_size=block_size,
        shapes=((2, 8), (2, 4, 16)),
        dtypes=(torch_dtype, torch.float32),
        page_size_padded=attention.page_size_bytes,
        mamba_cache_mode="align",
    )
    groups = [
        KVCacheGroupSpec(list(_STATE_NAMES), state),
        KVCacheGroupSpec(list(_ATTENTION_NAMES), attention),
    ]
    page = attention.page_size_bytes
    return KVCacheConfig(
        num_blocks=_NUM_BLOCKS,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            KVCacheTensor(
                size=2 * _NUM_BLOCKS * page,
                layers=group.layer_names,
                layer_stride=_NUM_BLOCKS * page,
                block_stride=page,
            )
            for group in groups
        ],
        kv_cache_layout="LBNHC",
    )


def _poisoned_shared_cache(block_size: int, dtype: mx.Dtype) -> MetalPagedKVCache:
    storage = KVCacheStorage(_config(block_size, dtype))
    # Poison the actual upstream-owned bytes in place, retaining the shared
    # import. Replacing storage.buffer would test a different allocation.
    for raw in storage._region_storages:
        raw.fill_(0xFF)
    cache = MetalPagedKVCache.from_upstream(storage, _ATTENTION_NAMES, dtype=dtype)
    _, recurrent = storage.state_views(_STATE_NAMES)
    assert recurrent[1].dtype == mx.float32
    assert np.isnan(np.array(recurrent[1][3])).all()
    # Per-layer regions: every layer view bases its own buffer at offset 0.
    assert storage.tensors[_ATTENTION_NAMES[1]].storage_offset() == 0
    assert (
        storage.tensors[_ATTENTION_NAMES[0]].untyped_storage()._cdata
        != storage.tensors[_ATTENTION_NAMES[1]].untyped_storage()._cdata
    )
    return cache


def _slots(block_size: int, start: int, stop: int) -> list[int]:
    return [
        _BLOCK_TABLE[token // block_size] * block_size + token % block_size
        for token in range(start, stop)
    ]


def _write_prefix(
    cache: MetalPagedKVCache,
    slots: list[int],
    keys: mx.array,
    values: mx.array,
) -> None:
    new_keys, new_values = get_ops().reshape_and_cache(
        mx.contiguous(keys),
        mx.contiguous(values),
        cache.key_caches[_LAYER],
        cache.value_caches[_LAYER],
        mx.array(slots, dtype=mx.int64),
    )
    cache.replace_layer_cache(_LAYER, new_keys, new_values)


def _numpy_causal_attention(
    queries: mx.array, keys: mx.array, values: mx.array
) -> np.ndarray:
    # Convert the already-rounded model inputs, including bfloat16, to fp32.
    q = np.array(queries.astype(mx.float32))
    k = np.repeat(np.array(keys.astype(mx.float32)), _QUERY_HEADS // _KV_HEADS, axis=1)
    v = np.repeat(
        np.array(values.astype(mx.float32)), _QUERY_HEADS // _KV_HEADS, axis=1
    )
    query_len, sequence_len = len(q), len(k)
    scores = np.einsum("qhd,khd->qhk", q, k) * _HEAD_DIM**-0.5
    query_positions = sequence_len - query_len + np.arange(query_len)
    causal = np.arange(sequence_len)[None, :] <= query_positions[:, None]
    scores = np.where(causal[:, None, :], scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum("qhk,khd->qhd", weights, v).reshape(
        1, query_len, _QUERY_HEADS * _HEAD_DIM
    )


def _assert_partial_tail_masking(
    block_size: int, query_len: int, dtype: mx.Dtype
) -> None:
    # Cross a scheduler-page boundary and leave three NaN rows in the final
    # visited 16-token kernel page. The remaining scheduler-page tail is dirty.
    assert _pick_kernel_block_size(block_size) == 16
    sequence_len = block_size + _TAIL_TOKENS
    prefix_len = sequence_len - query_len
    rng = np.random.default_rng(98071)
    queries = mx.array(
        rng.normal(0, 0.6, (query_len, _QUERY_HEADS, _HEAD_DIM)), dtype=dtype
    )
    keys = mx.array(
        rng.normal(0, 0.7, (sequence_len, _KV_HEADS, _HEAD_DIM)), dtype=dtype
    )
    values = mx.array(
        rng.normal(0, 0.8, (sequence_len, _KV_HEADS, _HEAD_DIM)), dtype=dtype
    )
    x = mx.concatenate(
        [
            queries.reshape(query_len, -1),
            keys[prefix_len:].reshape(query_len, -1),
            values[prefix_len:].reshape(query_len, -1),
        ],
        axis=-1,
    )[None]
    shared = _poisoned_shared_cache(block_size, dtype)
    zeroed = MetalPagedKVCache(
        num_layers=2,
        num_kv_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        num_blocks=_NUM_BLOCKS,
        block_size=block_size,
        dtype=dtype,
    )
    outputs = []
    for cache in (shared, zeroed):
        _write_prefix(
            cache,
            _slots(block_size, 0, prefix_len),
            keys[:prefix_len],
            values[:prefix_len],
        )
        ctx = PagedAttentionContext(
            slot_mapping=_slots(block_size, prefix_len, sequence_len),
            block_tables=[_BLOCK_TABLE.copy()],
            context_lens=[sequence_len],
            offsets=[prefix_len],
            cu_seqlens=[0, query_len],
            num_decode_requests=int(query_len == 1),
        )
        output, _ = sdpa_forward(
            _SliceAttention(),
            x,
            ctx,
            cache,
            _LAYER,
            attention_contract=AttentionContract(use_rope=False),
        )
        outputs.append(output)
    mx.eval(*outputs)
    actual, reference = [np.array(output.astype(mx.float32)) for output in outputs]
    assert np.isfinite(reference).all(), "the zero-backed control must be finite"
    assert np.isfinite(actual).all(), "unwritten shared-page NaNs leaked into attention"
    np.testing.assert_array_equal(actual, reference)
    np.testing.assert_allclose(
        actual,
        _numpy_causal_attention(queries, keys, values),
        atol=3e-3 if dtype == mx.bfloat16 else 5e-4,
        rtol=2e-2 if dtype == mx.bfloat16 else 3e-3,
    )
    # Check original upstream bytes, including the entire unused tail, layer,
    # and unmapped block. A clear/copy workaround must not hide bad masking.
    mx.eval(shared._storage.buffer)
    untouched = shared._storage.tensors[_ATTENTION_NAMES[0]]
    written = shared._storage.tensors[_ATTENTION_NAMES[1]]
    for region in (untouched, written[0], written[_BLOCK_TABLE[-1], :, _TAIL_TOKENS:]):
        assert torch.all(region.view(torch.uint8) == 0xFF)
    np.testing.assert_array_equal(
        np.array(shared.key_caches[_LAYER][1, :_TAIL_TOKENS].astype(mx.float32)),
        np.array(keys[-_TAIL_TOKENS:].astype(mx.float32)),
    )


@pytest.mark.parametrize("query_len", [1, 65], ids=["decode", "prefill"])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16], ids=["fp16", "bf16"])
def test_dirty_shared_page_tail_matches_zero_backed_sdpa(
    query_len: int, dtype: mx.Dtype
) -> None:
    # Native dispatch is unchanged: multi-query fp16/bf16 uses NAX where it is
    # enabled and available, otherwise tiled attention on the same inputs.
    _assert_partial_tail_masking(784, query_len, dtype)


def test_dirty_shared_page_tail_matches_tiled_prefill(force_tiled_prefill) -> None:
    _assert_partial_tail_masking(784, 65, mx.float16)
