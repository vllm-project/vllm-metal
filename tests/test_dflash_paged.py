# SPDX-License-Identifier: Apache-2.0
"""Real paged kernels agree with full-context DFlash across cache reuse."""

from dataclasses import replace

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
)

from tests.test_dflash import _config, _torch_forward
from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.v1.dflash import DFlashModel
from vllm_metal.v1.dflash_paged import DFlashPagedCache


def make_cache(dtype=mx.float16, *, block_size=16):
    cfg = replace(
        _config(),
        hidden_size=64,
        head_dim=64,
        max_position_embeddings=64,
        block_size=16,
    )
    model = DFlashModel(cfg)
    model.set_dtype(dtype)
    embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
    embed.set_dtype(dtype)
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=cfg.num_key_value_heads,
        head_size=cfg.head_dim,
        dtype=torch.float16 if dtype == mx.float16 else torch.bfloat16,
    )
    # Target and draft layers occupy distinct regions of ONE allocation.
    names = ("target", "dflash_layers.0.self_attn", "dflash_layers.1.self_attn")
    size = 12 * spec.page_size_bytes
    storage = KVCacheStorage(
        KVCacheConfig(
            num_blocks=12,
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=list(names), kv_cache_spec=spec)
            ],
            kv_cache_tensors=[
                KVCacheTensor(
                    size=len(names) * size,
                    layers=list(names),
                    layer_stride=size,
                    block_stride=spec.page_size_bytes,
                )
            ],
            kv_cache_layout="LBNHC",
        )
    )
    for tensor in storage.tensors.values():
        tensor.fill_(7)
    torch.mps.synchronize()
    cache = DFlashPagedCache(model, storage, names[1:], max_model_len=64)
    return model, embed, cache


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("poison", [False, True])
@pytest.mark.parametrize("block_size", [8, 16, 32])
@pytest.mark.parametrize("num_draft_tokens", [1, 3, 15])
def test_compiled_ragged_blocks_rejection_and_page_reuse(
    dtype, poison, block_size, num_draft_tokens
):
    model, embed, cache = make_cache(dtype, block_size=block_size)
    if poison:
        for name in ("dflash_layers.0.self_attn", "dflash_layers.1.self_attn"):
            cache.storage.tensors[name].fill_(float("nan"))
        torch.mps.synchronize()
    draft = cache.compile_draft(
        num_draft_tokens=num_draft_tokens, embed=embed, project=embed.as_linear
    )
    tables = [[5, 2, 7, 1], [9, 3, 8, 4]]
    # Repeated shapes replay the compiled graph with different positions and
    # contents. Shrinking the prefix simulates rejected tails and page reuse.
    before_boundary = (block_size - 1, min(2 * block_size - 2, 47))
    at_boundary = (block_size, min(2 * block_size - 1, 48))
    for lengths in [before_boundary, at_boundary, before_boundary, (1, 2)]:
        features = [
            [mx.random.normal((1, length, 64)).astype(dtype) for _ in range(3)]
            for length in lengths
        ]
        packed = [mx.concatenate([f[i][0] for f in features]) for i in range(3)]
        cache.write_context(
            packed, [(blocks, 0, n) for blocks, n in zip(tables, lengths, strict=True)]
        )
        anchors = mx.array([11, 12])
        actual = draft(anchors, list(zip(tables, lengths, strict=True)))
        mx.eval(actual, *cache.storage.buffers)
        for row, feature in enumerate(features):
            embedding = model._draft_embeddings(
                anchors[row : row + 1], num_draft_tokens, embed
            )
            reference = _torch_forward(model, embedding, feature)[:, 1:]
            expected = embed.as_linear(mx.array(reference).astype(dtype))
            tolerance = 0.04 if dtype == mx.bfloat16 else 0.006
            np.testing.assert_allclose(
                np.array(actual[row : row + 1].astype(mx.float32)),
                np.array(expected.astype(mx.float32)),
                atol=tolerance,
                rtol=tolerance,
            )
        # No write may touch target bytes or unallocated physical pages.
        assert torch.all(cache.storage.tensors["target"] == 7)
        for name in ("dflash_layers.0.self_attn", "dflash_layers.1.self_attn"):
            untouched = cache.storage.tensors[name][[0, 6, 10, 11]]
            assert torch.all(torch.isnan(untouched) if poison else untouched == 7)


def test_missing_lookahead_is_rejected_before_cache_write():
    model, embed, cache = make_cache()
    draft = cache.compile_draft(
        num_draft_tokens=3, embed=embed, project=embed.as_linear
    )
    with pytest.raises(RuntimeError, match="scheduler supplied"):
        draft(mx.array([11]), [([2], 15)])
    assert all(torch.all(t == 7) for t in cache.storage.tensors.values())


@pytest.mark.parametrize(
    "blocks,length", [([-1, 2], 15), ([12, 2], 15), ([2, 3, 4, 5], 61)]
)
def test_invalid_page_or_context_bound_is_rejected(blocks, length):
    model, embed, cache = make_cache()
    draft = cache.compile_draft(
        num_draft_tokens=3, embed=embed, project=embed.as_linear
    )
    with pytest.raises(ValueError):
        draft(mx.array([11]), [(blocks, length)])
    assert all(torch.all(t == 7) for t in cache.storage.tensors.values())
