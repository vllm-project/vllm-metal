# SPDX-License-Identifier: Apache-2.0
"""GLM-5-Next sparse MLA on vLLM's paged latent and pooled-key caches."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import torch
from mlx_vlm.models.cache import KVCache, PoolingCache
from mlx_vlm.models.linear import DECODE_BLOCK_SIZE, linear
from mlx_vlm.models.sparse_attention import indexed_sparse_attention
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SparseCacheRole

from vllm_metal.attention.caches.mla_cache import MLAPagedLatentCache
from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.attention.context import PagedAttentionContext, get_context
from vllm_metal.attention.impls.mla import _block_table_rows, _mla_metadata

# vLLM's name for this cache, built inline by its GLM-5-Next Indexer.
INDEXER_CACHE_SUFFIX = ".indexer.k_cache"


def is_sparse_mla(model_args: Mapping[str, Any]) -> bool:
    """Whether the model's MLA layers carry GLM-5-Next's pooled indexer."""
    return "index_kpool" in model_args


def is_dsa_attention(module: nn.Module) -> bool:
    """Return whether a module is an MLA layer with its own pooled indexer."""
    indexer = getattr(module, "indexer", None)
    return hasattr(indexer, "index_kpool") and hasattr(module, "kv_lora_rank")


def indexer_cache_spec(
    model_args: Mapping[str, Any], *, block_size: int, dtype: torch.dtype
) -> MLAAttentionSpec:
    """vLLM's pooled-key spec for one sparse layer.

    It is in the model dtype; vLLM's ``Glm5NextIndexerCache`` builds an fp8 one
    inline.
    """
    if model_args.get("qk_rope_head_dim") != 0:
        raise ValueError(
            "GLM-5-Next sparse MLA: needs qk_rope_head_dim 0 (NoPE), got "
            f"{model_args.get('qk_rope_head_dim')!r}."
        )
    if any(kind != "full" for kind in model_args.get("indexer_types") or ()):
        raise NotImplementedError(
            "GLM-5-Next sparse MLA: layers that reuse another layer's top-k "
            "(indexer_types 'shared') are not supported."
        )
    kpool = int(model_args["index_kpool"])
    head_dim = int(model_args["index_head_dim"])
    # vLLM asserts this in Glm5NextIndexerCache.__init__.
    if block_size % kpool:
        raise ValueError(
            f"GLM-5-Next sparse MLA: KV cache block_size {block_size} must be a "
            f"multiple of index_kpool {kpool}, so pools never straddle blocks."
        )
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_dim,
        dtype=dtype,
        cache_role=SparseCacheRole.INDEXER,
        tokens_per_state=kpool,
    )


def max_latent_blocks(*, block_size: int, latent_dim: int) -> int:
    """The most latent blocks mlx-vlm's kernel can address with 32-bit offsets."""
    return (2**31 - 1) // (block_size * latent_dim)


@dataclass(frozen=True)
class SparseIndexerCaches:
    """The sparse layers' pooled-key pages and their unfinished pools' rows."""

    # [num_blocks, block_size // index_kpool, index_head_dim], vLLM's pages
    pooled: MLAPagedLatentCache
    # [num_blocks, index_kpool, 2 * index_head_dim]: raw key | gate of the
    # unfinished pool in that latent block, at its position % index_kpool
    tail: MLAPagedLatentCache

    @classmethod
    def from_upstream(
        cls,
        storage: KVCacheStorage,
        attention_names: Sequence[str],
        *,
        latent: MLAPagedLatentCache,
    ) -> SparseIndexerCaches:
        """Bind vLLM's pooled keys and allocate the tail rows.

        The pooled keys sit next to the latent in vLLM's layout; the tail rows
        sit beside each latent block, in a table of their own.
        """
        limit = max_latent_blocks(
            block_size=latent.block_size, latent_dim=latent.latent_dim
        )
        if latent.num_blocks > limit:
            raise ValueError(
                f"GLM-5-Next sparse MLA: {latent.num_blocks} KV blocks exceed "
                f"the {limit} that mlx-vlm's indexed attention kernel can "
                "address with its 32-bit offsets; lower "
                "--num-gpu-blocks-override."
            )
        if not latent.has_dense_pages:
            raise ValueError("GLM-5-Next sparse MLA: requires dense latent pages.")
        pooled = MLAPagedLatentCache.from_upstream(
            storage, [name + INDEXER_CACHE_SUFFIX for name in attention_names]
        )
        kpool = latent.block_size // pooled.block_size
        # vLLM's ring is a KV group of its own, for KV transfer and speculative
        # tokens; neither runs here, so the rows sit beside their latent block.
        tail = MLAPagedLatentCache(
            num_layers=len(attention_names),
            latent_dim=2 * pooled.latent_dim,
            num_blocks=latent.num_blocks,
            block_size=kpool,
            dtype=latent.dtype,
        )
        return cls(pooled=pooled, tail=tail)


def _restore_pooling_cache(
    tail_rows: mx.array | None,
    pooled_pages: mx.array,
    block_table: mx.array,
    past: int,
    kpool: int,
) -> PoolingCache:
    """mlx-vlm's ``PoolingCache`` for one request, from its pages.

    ``tail_rows`` are those of the latent block holding position ``past - 1``,
    whose first ``past % kpool`` are the unfinished pool's raw keys and gates;
    ``None`` when every pool is complete.
    """
    num_pools, remainder = divmod(past, kpool)
    head_dim = pooled_pages.shape[-1]
    pooled = None
    if num_pools:
        blocks = block_table[: math.ceil(num_pools / pooled_pages.shape[1])]
        pooled = pooled_pages[blocks].reshape(1, -1, head_dim)[:, :num_pools]
    tail = None if tail_rows is None else tail_rows[None, :remainder]
    pool = PoolingCache(kpool)
    pool.state = (
        None if tail is None else tail[..., :head_dim],
        None if tail is None else tail[..., head_dim:],
        pooled,
    )
    return pool


def _key_validity(past: int) -> KVCache:
    """The indexer's key-validity cache: every paged key is valid."""
    valid = KVCache()
    if past:
        valid.state = (
            mx.ones((1, 1, past, 1), dtype=mx.bool_),
            mx.zeros((1, 1, past, 0), dtype=mx.bool_),
        )
    return valid


@dataclass
class _PageWrites:
    """The pools and tails a step's requests complete, scattered once per layer."""

    pool_rows: list[mx.array] = field(default_factory=list)
    pool_slots: list[int] = field(default_factory=list)
    tail_rows: list[mx.array] = field(default_factory=list)
    tail_slots: list[int] = field(default_factory=list)

    def add(
        self,
        pool: PoolingCache,
        *,
        past: int,
        latent_slots: Sequence[int],
        block_size: int,
    ) -> None:
        """Queue one request's new pools and tail after mlx-vlm's indexer ran.

        A pool's slot is its last token's latent slot divided by ``kpool``, as
        vLLM maps pools in ``get_compressed_slot_mapping``, a Triton kernel.
        """
        kpool, length = pool.ratio, len(latent_slots)
        new_pools = (past + length) // kpool - past // kpool
        if new_pools:
            self.pool_rows.append(pool.pooled[0, -new_pools:])
            self.pool_slots += [
                slot // kpool
                for j, slot in enumerate(latent_slots)
                if (past + j + 1) % kpool == 0
            ]
        if remainder := (past + length) % kpool:
            keys, gates, _ = pool.state
            # Only this step's rows are new: earlier ones are in place already.
            new = min(remainder, length)
            self.tail_rows.append(
                mx.concatenate([keys[0, -new:], gates[0, -new:]], axis=-1)
            )
            self.tail_slots += [
                slot // block_size * kpool + slot % kpool
                for slot in latent_slots[-new:]
            ]

    def write(self, caches: SparseIndexerCaches, layer: int) -> None:
        if self.pool_rows:
            caches.pooled.write_slots(
                layer,
                mx.array(self.pool_slots, dtype=mx.int32),
                mx.concatenate(self.pool_rows),
            )
        if self.tail_rows:
            caches.tail.write_slots(
                layer,
                mx.array(self.tail_slots, dtype=mx.int32),
                mx.concatenate(self.tail_rows).astype(caches.tail.dtype),
            )


def _global_slots(
    topk: mx.array, block_table: mx.array, block_size: int
) -> tuple[mx.array, mx.array]:
    """Top-k positions to latent slots and their mask.

    vLLM's ``triton_convert_req_index_to_global_index`` is Triton-only. An
    unselected entry (-1) reads the request's first latent slot, as mlx-vlm
    clips it.
    """
    safe = mx.maximum(topk, 0)
    slots = mx.take(block_table, safe // block_size) * block_size + safe % block_size
    return slots, topk >= 0


class DSAPagedAttentionWrapper(nn.Module):
    """Run a GLM-5-Next sparse attention layer on the paged caches.

    mlx-vlm's indexer runs per request; attention reads only the selected rows.
    """

    def __init__(
        self,
        inner: nn.Module,
        layer_idx: int,
        latent_cache: MLAPagedLatentCache,
        indexer_caches: SparseIndexerCaches,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        self.rebind_cache(
            latent_cache, cache_idx=layer_idx, indexer_caches=indexer_caches
        )

    def rebind_cache(
        self,
        latent_cache: MLAPagedLatentCache,
        *,
        cache_idx: int,
        indexer_caches: SparseIndexerCaches,
    ) -> None:
        """Refresh the caches and compact layer index."""
        object.__setattr__(self, "_dsa_layer_idx", cache_idx)
        object.__setattr__(self, "_dsa_latent_cache", latent_cache)
        object.__setattr__(self, "_dsa_indexer_caches", indexer_caches)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any = None,
        prev_topk_indices: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        ctx = get_context()
        if ctx is None:
            return self._inner(x, mask, cache, prev_topk_indices)

        inner = self._inner
        layer = self._dsa_layer_idx
        latent_cache = self._dsa_latent_cache
        # The projections of mlx-vlm's Glm5NextAttention.__call__.
        q_a, kv_a = mx.split(linear(inner.qkv_a_proj, x), (inner.q_lora_rank,), axis=-1)
        q_resid = inner.q_a_layernorm(q_a)
        q = (
            linear(inner.q_b_proj, q_resid)
            .reshape(*x.shape[:2], inner.num_heads, inner.q_head_dim)
            .transpose(0, 2, 1, 3)
        )
        new_latent = inner.kv_a_layernorm(kv_a)
        if new_latent.dtype != latent_cache.dtype:
            raise ValueError(
                f"GLM-5-Next sparse MLA: keeps {latent_cache.dtype} caches but "
                f"the layer computes {new_latent.dtype}; serve with --dtype "
                "matching it."
            )
        # Written first: the always-selected tail includes this step's rows.
        latent_cache.write_slots(layer, _mla_metadata(ctx).slot_mapping, new_latent[0])
        latent = latent_cache.latent_caches[layer]
        latent = latent.reshape(1, 1, -1, latent.shape[-1])

        rows, slots, selected = self._select(ctx, x, q_resid)
        # mlx-vlm's own split between its decode and prefill attention.
        decode = [
            i for i, request in enumerate(rows) if len(request) <= DECODE_BLOCK_SIZE
        ]
        prefill = [i for i in range(len(rows)) if i not in decode]
        outputs: list[mx.array] = []
        order: list[int] = []
        for requests, attend in (
            (decode, self._decode_attention),
            (prefill, self._prefill_attention),
        ):
            if requests:
                request_rows = [row for i in requests for row in rows[i]]
                outputs.append(
                    attend(
                        q,
                        latent,
                        request_rows,
                        mx.concatenate([slots[i] for i in requests]),
                        mx.concatenate([selected[i] for i in requests]),
                    )
                )
                order += request_rows
        out = mx.concatenate(outputs)
        if order != sorted(order):
            out = out[mx.argsort(mx.array(order))]
        return linear(inner.o_proj, out.reshape(1, len(order), -1)), None

    def _select(
        self, ctx: PagedAttentionContext, x: mx.array, q_resid: mx.array
    ) -> tuple[list[list[int]], list[mx.array], list[mx.array]]:
        """Each request's top-k as latent slots and a mask, writing back its pages."""
        kpool = self._inner.indexer.index_kpool
        layer = self._dsa_layer_idx
        caches = self._dsa_indexer_caches
        block_size = self._dsa_latent_cache.block_size
        pooled_pages = caches.pooled.latent_caches[layer]
        tail_pages = caches.tail.latent_caches[layer]
        block_tables = _block_table_rows(ctx)
        cu = ctx.cu_seqlens
        assert cu is not None
        rows: list[list[int]] = []
        slots: list[mx.array] = []
        selected: list[mx.array] = []
        writes = _PageWrites()
        for i, context_len in enumerate(ctx.context_lens):
            start, end = cu[i], cu[i + 1]
            past = context_len - (end - start)
            tail_rows = None
            if past % kpool:
                tail_rows = tail_pages[ctx.block_tables[i][(past - 1) // block_size]]
            pool = _restore_pooling_cache(
                tail_rows, pooled_pages, block_tables[i], past, kpool
            )
            topk = self._request_topk(
                x[:, start:end], q_resid[:, start:end], pool, past
            )
            rows.append(list(range(start, end)))
            request_slots, request_selected = _global_slots(
                topk[0], block_tables[i], block_size
            )
            slots.append(request_slots)
            selected.append(request_selected)
            writes.add(
                pool,
                past=past,
                latent_slots=ctx.slot_mapping[start:end],
                block_size=block_size,
            )
        writes.write(caches, layer)
        return rows, slots, selected

    def _request_topk(
        self, x: mx.array, q_resid: mx.array, pool: PoolingCache, past: int
    ) -> mx.array:
        """mlx-vlm's indexer for one request, on its restored caches."""
        return self._inner.indexer(x, q_resid, None, _key_validity(past), pool, past)

    def _decode_attention(
        self,
        q: mx.array,
        latent: mx.array,
        rows: list[int],
        slots: mx.array,
        selected: mx.array,
    ) -> mx.array:
        """The decode branch of mlx-vlm's attention, over the rows read by slot."""
        inner = self._inner
        query = inner.embed_q(mx.take(q, mx.array(rows), axis=2).transpose(2, 1, 0, 3))
        keys = latent[0, 0][slots][:, None]
        out = mx.fast.scaled_dot_product_attention(
            query,
            keys,
            keys,
            scale=inner.scale,
            mask=selected[:, None, None],
        )
        return inner.unembed_out(out)[:, :, 0]

    def _prefill_attention(
        self,
        q: mx.array,
        latent: mx.array,
        rows: list[int],
        slots: mx.array,
        selected: mx.array,
    ) -> mx.array:
        """vLLM's latent-space prefill, reading the selected rows in place."""
        inner = self._inner
        query = inner.embed_q(mx.take(q, mx.array(rows), axis=2))
        indices = mx.where(selected, slots, -1)[None]
        out = indexed_sparse_attention(query, latent, latent, indices, inner.scale)
        if out is None:
            raise RuntimeError(
                "GLM-5-Next sparse MLA: mlx-vlm's indexed sparse attention "
                "kernel declined "
                f"{query.dtype} queries {tuple(query.shape)} over "
                f"{tuple(latent.shape)}; it needs bf16 or fp16 on the GPU and "
                "latent widths that are multiples of 32."
            )
        return inner.unembed_out(out)[0].transpose(1, 0, 2)
