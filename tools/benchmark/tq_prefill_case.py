# SPDX-License-Identifier: Apache-2.0
"""Small production SDPA fixture shared by the TQ benchmark and regressions."""

from dataclasses import dataclass
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import record_kv_cache_layout
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import KVCacheGroupSpec

from vllm_metal.attention.block_tables import build_block_tables
from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.caches.placement import KV_CACHE_LAYOUT
from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.attention.caches.turboquant import (
    QUANT_PARAMS,
    get_v_centroids,
    prefill_workspace_bytes,
)
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.impls import sdpa
from vllm_metal.metal import get_ops
from vllm_metal.v1.cache_policy import (
    TurboQuantAttentionSpec,
    turboquant_page_size_bytes,
)


class Projection:
    def __init__(self, inputs: int, outputs: int, dtype: mx.Dtype, seed: int):
        self.weight = (
            mx.random.normal((outputs, inputs), key=mx.random.key(seed)) / inputs**0.5
        ).astype(dtype)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.T


@dataclass
class AttentionCase:
    inner: SimpleNamespace
    x: mx.array
    ctx: PagedAttentionContext
    cache: MetalPagedKVCache

    def forward(self) -> mx.array:
        return sdpa.sdpa_forward(self.inner, self.x, self.ctx, self.cache, 0)[0]

    def reference(self) -> mx.array:
        """Old compressed primitive on the same cache, after current writes."""
        c = self.cache
        q, *_ = sdpa.prepare_sdpa_qkv(
            self.inner, self.x, self.ctx, self.inner.n_heads, self.inner.n_kv_heads
        )
        q = mx.contiguous(q[0].transpose(1, 0, 2).astype(c.dtype))
        table, kb = build_block_tables(self.ctx.block_tables, c.block_size)

        def kernel_view(array):
            return array.reshape(-1, kb, c.num_kv_heads, array.shape[-1])

        out = mx.array(0)
        get_ops().paged_attention_primitive(
            q,
            kernel_view(c.key_caches[0]),
            kernel_view(c.value_caches[0]),
            c.num_kv_heads,
            self.inner.scale,
            self.inner.attn_logit_softcapping,
            table,
            mx.array(self.ctx.context_lens, mx.int32),
            mx.array(self.ctx.cu_seqlens, mx.int32),
            kb,
            max(self.ctx.context_lens),
            c.sliding_window_per_layer[0],
            out,
            use_turboquant=True,
            quant_type=c.k_quant,
            v_bits=c.v_bits,
            key_scale_cache=kernel_view(c.key_scale_caches[0]),
            value_scale_cache=kernel_view(c.value_scale_caches[0]),
            key_zero_cache=kernel_view(c.key_zero_caches[0]),
            v_centroids=get_v_centroids(c.v_bits),
            window_seqlen_q=self.ctx.verify_window_q,
            sinks=getattr(self.inner, "sinks", None),
        )
        return out.reshape(1, self.x.shape[1], -1)


def build_case(
    *,
    qlens=(128,),
    context_lens=(257,),
    head_dim=128,
    block_size=16,
    n_heads=8,
    n_kv_heads=2,
    dtype=mx.bfloat16,
    k_quant="q8_0",
    v_quant="q3_0",
    page_padding=0,
    pool_blocks=None,
    shared_prefix=False,
    extra_table_pages=0,
    sliding_window=-1,
    softcap=0.0,
) -> AttentionCase:
    """Bind upstream storage, seed only past KV, and leave current writes lazy.

    Tables are deliberately nonidentity and physical pages are separated from
    logical sequence order. Padding/tail pages are zeroed but never requested.
    Shared prefixes reuse all complete pages common to the cached histories.
    """
    counts = [(n + block_size - 1) // block_size for n in context_lens]
    minimum = sum(counts) + extra_table_pages * len(counts) + 4
    pool_blocks = max(pool_blocks or minimum, minimum)
    order = np.random.default_rng(73).permutation(np.arange(1, minimum))
    tables, cursor = [], 0
    for n in counts:
        tables.append(order[cursor : cursor + n + extra_table_pages].tolist())
        cursor += n + extra_table_pages
    if shared_prefix:
        assert all(
            n - q >= block_size for n, q in zip(context_lens, qlens, strict=True)
        )
        prefix_pages = min(
            (n - q) // block_size for n, q in zip(context_lens, qlens, strict=True)
        )
        for row in tables[1:]:
            row[:prefix_pages] = tables[0][:prefix_pages]
    size = turboquant_page_size_bytes(
        block_size, n_kv_heads, head_dim, k_quant, v_quant
    )
    spec = TurboQuantAttentionSpec(
        block_size=block_size,
        num_kv_heads=n_kv_heads,
        head_size=head_dim,
        dtype=torch.int8,
        k_quant=k_quant,
        v_quant=v_quant,
        page_size_padded=size + page_padding if page_padding else None,
    )
    config = VllmConfig()
    record_kv_cache_layout(config.cache_config, KV_CACHE_LAYOUT)
    layout = get_kv_cache_config_from_groups(
        config,
        [KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=spec)],
        spec.page_size_bytes * pool_blocks,
    )
    layout.kv_cache_layout = config.cache_config.kv_cache_layout
    storage = KVCacheStorage(layout)
    storage.zero_blocks(list(range(pool_blocks)))
    cache = MetalPagedKVCache.from_upstream(storage, ["attn"], dtype=dtype)
    cache.sliding_window_per_layer[0] = sliding_window
    past_slots, new_slots = [], []
    for table, length, qlen in zip(tables, context_lens, qlens, strict=True):
        past_slots.extend(
            table[t // block_size] * block_size + t % block_size
            for t in range(length - qlen)
        )
        new_slots.extend(
            table[t // block_size] * block_size + t % block_size
            for t in range(length - qlen, length)
        )
    # Shared pages are written once, with identical history for every owner.
    past_slots = list(dict.fromkeys(past_slots))
    if past_slots:
        k = mx.random.normal(
            (len(past_slots), n_kv_heads, head_dim), key=mx.random.key(81)
        ).astype(dtype)
        v = mx.random.normal(k.shape, key=mx.random.key(82)).astype(dtype)
        updated = get_ops().tq_encode(
            k,
            v,
            cache.key_caches[0],
            cache.value_caches[0],
            cache.key_scale_caches[0],
            cache.value_scale_caches[0],
            cache.key_zero_caches[0],
            mx.array(past_slots, mx.int64),
            get_v_centroids(cache.v_bits),
            cache.v_bits,
            cache.k_bits,
            bool(QUANT_PARAMS[k_quant]["signed"]),
        )
        for arrays, value in zip(
            (
                cache.key_caches,
                cache.value_caches,
                cache.key_scale_caches,
                cache.value_scale_caches,
                cache.key_zero_caches,
            ),
            updated,
            strict=True,
        ):
            arrays[0] = value
        mx.eval(*updated)
    else:
        mx.eval(*storage.buffers)
    hidden = 32
    inner = SimpleNamespace(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        scale=head_dim**-0.5,
        attn_logit_softcapping=softcap,
        q_proj=Projection(hidden, n_heads * head_dim, dtype, 91),
        k_proj=Projection(hidden, n_kv_heads * head_dim, dtype, 92),
        v_proj=Projection(hidden, n_kv_heads * head_dim, dtype, 93),
        o_proj=lambda x: x,
        rope=lambda x, offset=0: x,
    )
    x = mx.random.normal((1, sum(qlens), hidden), key=mx.random.key(94)).astype(dtype)
    mx.eval(x, inner.q_proj.weight, inner.k_proj.weight, inner.v_proj.weight)
    ctx = PagedAttentionContext(
        tq_prefill_workspace_bytes=prefill_workspace_bytes(),
        slot_mapping=new_slots,
        block_tables=tables,
        context_lens=list(context_lens),
        offsets=[n - q for n, q in zip(context_lens, qlens, strict=True)],
        cu_seqlens=[0, *np.cumsum(qlens).tolist()],
    )
    return AttentionCase(inner, x, ctx, cache)
