# SPDX-License-Identifier: Apache-2.0
"""GLM-5-Next sparse MLA on vLLM's paged latent cache."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import torch
from mlx_vlm.models.cache import KVCache, PoolingCache
from mlx_vlm.models.linear import linear
from mlx_vlm.models.sparse_attention import indexed_sparse_attention
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_metal.attention.caches.mla_cache import MLAPagedLatentCache
from vllm_metal.attention.caches.storage import CacheViews, KVCacheStorage
from vllm_metal.attention.context import PagedAttentionContext, get_context
from vllm_metal.attention.impls.mla import _block_table_rows, _mla_metadata

# The most per-head float32 indexer scores one prefill query part may hold:
# the default of vLLM's VLLM_SPARSE_INDEXER_MAX_LOGITS_MB, which bounds the
# logits summed over the heads.
_MAX_SCORES_BYTES = 512 * 2**20


def is_sparse_mla(model_args: Mapping[str, Any]) -> bool:
    """Whether the model's MLA layers carry GLM-5-Next's pooled indexer."""
    return "index_kpool" in model_args


def is_dsa_attention(module: nn.Module) -> bool:
    """Return whether a module is an MLA layer with its own pooled indexer."""
    indexer = getattr(module, "indexer", None)
    return hasattr(indexer, "index_kpool") and hasattr(module, "kv_lora_rank")


def latent_page_rows(model_args: Mapping[str, Any], *, block_size: int) -> int:
    """A sparse layer's latent rows per block, its pooled keys' rows included."""
    pooled = block_size // int(model_args["index_kpool"])
    return block_size + pooled * int(model_args["index_head_dim"]) // int(
        model_args["kv_lora_rank"]
    )


def sparse_latent_spec(
    spec: MLAAttentionSpec, model_args: Mapping[str, Any]
) -> MLAAttentionSpec:
    """A sparse layer's latent spec, with the pooled keys in its page padding.

    vLLM's GLM-5-Next cache keeps the pooled keys in the latent's block, which
    its block-outer layout packs. A pooled-key spec of its own would, on Metal's
    layer-outer layout, become a KV group whose block vLLM grows to the latent
    page (16 times at GLM-5.3-Flash's sizes), and prefix-cache hits would land
    only on that block's multiples. In the padding they share the latent's
    block table. They are in the model dtype; vLLM's are fp8.
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
    if not model_args.get("index_kpool_always_select_tail", True):
        raise NotImplementedError(
            "GLM-5-Next sparse MLA: requires index_kpool_always_select_tail=True."
        )
    kpool = int(model_args["index_kpool"])
    pooled_size = spec.block_size // kpool * int(model_args["index_head_dim"])
    if spec.block_size % kpool or pooled_size % spec.head_size:
        raise ValueError(
            f"GLM-5-Next sparse MLA: KV cache block_size {spec.block_size} must "
            f"be a multiple of index_kpool {kpool}, so pools never straddle "
            "blocks, and its pooled keys must fill whole latent rows."
        )
    rows = latent_page_rows(model_args, block_size=spec.block_size)
    return replace(spec, page_size_padded=rows * spec.state_content_size_bytes)


def sparse_mla_reserve_bytes(
    model_args: Mapping[str, Any],
    *,
    max_model_len: int,
    max_decode_rows: int,
    itemsize: int,
) -> int:
    """What a step's sparse layers hold that profile_run's unpaged run misses."""
    heads = int(model_args["index_n_heads"])
    head_dim = int(model_args["index_head_dim"])
    kpool = int(model_args["index_kpool"])
    cap = _MAX_SCORES_BYTES
    # One query part: per-head float32 scores at the cap, plus four [rows, pools]
    # float32 temporaries (head sum, mask, argpartition's values and indices).
    part = cap + 4 * cap // heads
    # Per pool of one max_model_len request: its key in the cache dtype and as
    # float32, its per-head float32 decode scores, and about 16 B of per-pool
    # rows (head-summed score, end position, validity, argpartition index).
    per_pool = (itemsize + 4) * head_dim + 4 * heads + 16
    # Each decode row's index_topk selected rows plus kpool - 1 tail rows.
    width = int(model_args["index_topk"]) + kpool - 1
    decode_rows = max_decode_rows * width * int(model_args["kv_lora_rank"]) * itemsize
    return part + max_model_len // kpool * per_pool + decode_rows


def max_latent_blocks(*, rows_per_block: int, latent_dim: int) -> int:
    """The most latent blocks mlx-vlm's kernel can address with 32-bit offsets."""
    return (2**31 - 1) // (rows_per_block * latent_dim)


@dataclass(frozen=True)
class SparseIndexerCaches:
    """The sparse layers' latent rows, pooled keys and unfinished pools' rows."""

    # [num_blocks * rows_per_block, latent_dim]: each latent page as rows, its
    # pooled keys after the block's latent rows
    rows: CacheViews
    rows_per_block: int
    # [num_blocks, block_size // index_kpool, index_head_dim], in the padding
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
        kpool: int,
        head_dim: int,
    ) -> SparseIndexerCaches:
        """View the pooled keys in the latent pages and allocate the tail rows.

        The tail rows sit beside each latent block, in a table of their own.
        """
        block_size, latent_dim = latent.block_size, latent.latent_dim
        page = storage.specs[attention_names[0]].page_size_bytes
        rows_per_block = page // (latent_dim * latent.dtype.size)
        assert (rows_per_block - block_size) * latent_dim >= (
            block_size // kpool * head_dim
        ), "GLM-5-Next sparse MLA: the latent pages have no room for the pooled keys"
        limit = max_latent_blocks(rows_per_block=rows_per_block, latent_dim=latent_dim)
        if latent.num_blocks > limit:
            raise ValueError(
                f"GLM-5-Next sparse MLA: {latent.num_blocks} KV blocks exceed "
                f"the {limit} that mlx-vlm's indexed attention kernel can "
                "address with its 32-bit offsets; lower "
                "--num-gpu-blocks-override."
            )
        tensors = [storage.tensors[name] for name in attention_names]
        num_blocks = latent.num_blocks
        rows = [
            torch.as_strided(
                t, (num_blocks * rows_per_block, latent_dim), (latent_dim, 1)
            )
            for t in tensors
        ]
        pooled = [
            torch.as_strided(
                t,
                (num_blocks, block_size // kpool, head_dim),
                (rows_per_block * latent_dim, head_dim, 1),
                t.storage_offset() + block_size * latent_dim,
            )
            for t in tensors
        ]
        # vLLM's ring is a KV group of its own, for KV transfer and speculative
        # tokens; neither runs here, so the rows sit beside their latent block.
        tail = MLAPagedLatentCache(
            num_layers=len(attention_names),
            latent_dim=2 * head_dim,
            num_blocks=num_blocks,
            block_size=kpool,
            dtype=latent.dtype,
        )
        return cls(
            rows=storage.views(rows),
            rows_per_block=rows_per_block,
            pooled=MLAPagedLatentCache.from_views(storage, pooled),
            tail=tail,
        )


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


def query_parts(length: int, heads: int, pools: int) -> list[int]:
    """Row bounds of a chunk's indexer query parts.

    vLLM's ``_split_indexer_prefill_chunks`` for one request; private, so not
    imported. It bounds logits summed over the heads, while mlx-vlm keeps a
    float32 score per head, so it gets the per-head cap divided by the heads.
    """
    cap = _MAX_SCORES_BYTES // heads
    if length * pools * 4 <= cap:
        return [0, length]
    step = max(1, cap // 4 // pools)
    return [*range(0, length, step), length]


def _global_slots(
    topk: mx.array, block_table: mx.array, block_size: int, rows_per_block: int
) -> tuple[mx.array, mx.array]:
    """Top-k positions to latent rows and their mask.

    vLLM's ``triton_convert_req_index_to_global_index`` is Triton-only. An
    unselected entry (-1) reads the request's first latent row, as mlx-vlm
    clips it.
    """
    safe = mx.maximum(topk, 0)
    blocks = mx.take(block_table, safe // block_size)
    return blocks * rows_per_block + safe % block_size, topk >= 0


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
        from mlx_vlm.models.glm5_next import language

        # Query parts call these private mlx-vlm names; binding them here
        # fails at wrap time on an install that lacks one.
        indexer = inner.indexer
        object.__setattr__(self, "_inner", inner)
        try:
            object.__setattr__(self, "_pool_select", language._exact_pool_select)
            object.__setattr__(self, "_project_keys", indexer._project_keys)
            object.__setattr__(self, "_project_queries", indexer._project_queries)
            object.__setattr__(self, "_compress_pools", indexer._compress_pools)
        except AttributeError as error:
            raise RuntimeError(
                f"GLM-5-Next sparse MLA: needs mlx-vlm's {error.name}."
            ) from error
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
        latent = self._dsa_indexer_caches.rows[layer][None, None]

        rows, slots, selected = self._select(ctx, x, q_resid)
        # One-token requests take mlx-vlm's decode attention, which gathers
        # their selected rows: one row per request, as the reserve counts.
        decode = [i for i, request in enumerate(rows) if len(request) == 1]
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
        out = linear(inner.o_proj, out.reshape(1, len(order), -1))
        if inner.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=inner.sharding_group)
        return out, None

    def _select(
        self, ctx: PagedAttentionContext, x: mx.array, q_resid: mx.array
    ) -> tuple[list[list[int]], list[mx.array], list[mx.array]]:
        """Each request's top-k as latent rows and a mask, writing back its pages."""
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
                topk[0], block_tables[i], block_size, caches.rows_per_block
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
        """mlx-vlm's indexer for one request, in one call or in query parts.

        One row is always one part, so a decode-only step evaluates nothing.
        """
        indexer = self._inner.indexer
        length = x.shape[1]
        pools = (past + length) // indexer.index_kpool
        bounds = query_parts(length, indexer.n_heads, pools)
        if len(bounds) == 2:
            return indexer(x, q_resid, None, _key_validity(past), pool, past)
        return self._topk_in_parts(x, q_resid, pool, past, bounds)

    def _topk_in_parts(
        self,
        x: mx.array,
        q_resid: mx.array,
        pool: PoolingCache,
        past: int,
        bounds: Sequence[int],
    ) -> mx.array:
        """mlx-vlm's single-call top-k of one prefill chunk, scored in parts.

        The chunk is pooled and projected once; each part is selected against
        that pool set and evaluated before the next.
        """
        indexer = self._inner.indexer
        kpool, length = indexer.index_kpool, x.shape[1]
        kv_length = past + length
        keys, gates = self._project_keys(x)
        ready_keys, ready_gates, _ = pool.accumulate_windows(
            keys, gates.astype(keys.dtype), past
        )
        pool_keys = pool.update_and_fetch(self._compress_pools(ready_keys, ready_gates))
        count = pool_keys.shape[1]
        pool_end = mx.minimum(mx.arange(1, count + 1) * kpool - 1, kv_length - 1)[None]
        pool_valid = mx.ones((1, count), dtype=mx.bool_)
        positions = past + mx.arange(length)
        select_k = min(indexer.index_topk // kpool, count)
        # mlx-vlm scores pools only when it must choose among them.
        scored = select_k < count
        q, weights = self._project_queries(x, q_resid)
        parts = []
        for start, end in itertools.pairwise(bounds):
            part = self._pool_select(
                q[:, start:end] if scored else None,
                pool_keys,
                weights[:, start:end] if scored else None,
                pool_ends=pool_end,
                pool_valid=pool_valid,
                query_positions=positions[start:end],
                select_k=select_k,
                scale=indexer.softmax_scale,
                chunk_size=end - start,
            )
            mx.eval(part)
            mx.synchronize()
            parts.append(part)
        selected = mx.concatenate([part[0] for part in parts], axis=1)
        selected_valid = mx.concatenate([part[1] for part in parts], axis=1)

        order = mx.argsort(mx.where(selected_valid, selected, count), axis=-1)
        selected = mx.take_along_axis(selected, order, axis=-1)
        selected_valid = mx.take_along_axis(selected_valid, order, axis=-1)
        safe = mx.clip(selected, 0, count - 1)
        topk = (safe[..., None] * kpool + mx.arange(kpool)).reshape(1, length, -1)
        valid = mx.repeat(selected_valid[..., None], kpool, axis=-1)
        topk = mx.where(valid.reshape(1, length, -1), topk, -1)
        width = indexer.index_topk
        if topk.shape[-1] < width:
            pad = [(0, 0), (0, 0), (0, width - topk.shape[-1])]
            topk = mx.pad(topk, pad, constant_values=-1)
        topk = topk[..., :width]
        # sparse_latent_spec refuses index_kpool_always_select_tail other than True.
        if kpool > 1:
            visible = positions + 1
            tail_count = visible % kpool
            tail = (visible - tail_count)[:, None] + mx.arange(kpool - 1)
            tail_valid = (mx.arange(kpool - 1) < tail_count[:, None]) & (
                tail < kv_length
            )
            topk = mx.concatenate([topk, mx.where(tail_valid, tail, -1)[None]], axis=-1)
        return topk.astype(mx.int32)

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
