# SPDX-License-Identifier: Apache-2.0
"""LMCache KV-cache integration for the vLLM Metal (Apple Silicon) backend.

vllm-metal keeps its paged KV cache as MLX arrays in Apple unified memory
(:class:`~vllm_metal.attention.caches.kv_cache.MetalPagedKVCache`), and does
*not* drive vLLM's standard v1 ``KVConnector`` hooks. This module provides a
thin bridge so LMCache can store and retrieve KV against that MLX cache.

Design
------
On Apple Silicon, an MLX array and a ``torch`` tensor can share the same bytes
(unified memory). vllm-metal's ``tensor_bridge`` exposes this as a near
zero-copy conversion. We present each layer's ``key_cache`` / ``value_cache``
(shape ``[num_blocks, block_size, num_kv_heads, head_dim]``, i.e. NHD) as a
single stacked ``torch`` tensor ``[2, num_blocks, block_size, num_kv_heads,
head_dim]`` -- exactly LMCache's ``NL_X_TWO_NB_BS_NH_HS`` (flash-attention NHD)
layout. The physical scatter/gather is done by LMCache's device-agnostic
``python_ops_fallback.multi_layer_kv_transfer`` via
:class:`~lmcache.v1.gpu_connector.cpu_connectors.VLLMPagedMemCPUConnectorV2`.

This is a *direct-engine* integration (LMCacheEngine.store/retrieve at the
prefill boundary), not the scheduler-driven vLLM connector. It is intended as
a working, benchmarkable local-dev path and the basis for a future upstreamable
connector.

Enable with ``VLLM_METAL_ENABLE_LMCACHE=1``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    import mlx.core as mx

    from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache

logger = init_logger(__name__)


def lmcache_enabled() -> bool:
    return os.environ.get("VLLM_METAL_ENABLE_LMCACHE", "0") == "1"


def _mlx_to_torch_cpu(array: mx.array) -> torch.Tensor:
    """Bridge an MLX array to a torch CPU tensor (unified-memory friendly).

    Uses vllm-metal's tensor_bridge, forcing the CPU device so LMCache's
    CPUCacheContext / python fallback path is selected.
    """
    from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch

    return mlx_to_torch(array, device="cpu")


class MetalLMCacheConnector:
    """Bridges a live :class:`MetalPagedKVCache` to an LMCache engine.

    One connector per model runner. Lazily initializes the LMCache engine on
    first use, once the paged cache dimensions are known.
    """

    def __init__(self, instance_id: str = "vllm-metal") -> None:
        self.instance_id = instance_id
        self._engine = None
        self._connector = None
        self._num_layers: int | None = None
        self._block_size: int | None = None
        self._chunk_size = int(os.environ.get("VLLM_METAL_LMCACHE_CHUNK_SIZE", "256"))
        # Stats
        self.n_store_calls = 0
        self.n_retrieve_calls = 0
        self.n_tokens_stored = 0
        self.n_tokens_hit = 0

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    # ------------------------------------------------------------------
    # KV bridging
    # ------------------------------------------------------------------
    def _stacked_kv_views(self, cache: MetalPagedKVCache) -> list[torch.Tensor]:
        """Return per-layer torch tensors [2, num_blocks, block_size, nh, hd].

        Layer i = stack(key_caches[i], value_caches[i]).

        The MLX paged cache is written by a lazy graph and its per-layer arrays
        are *rebound* to fresh objects after each forward/scatter. We force
        evaluation and read the current arrays here so the bridge sees fully
        materialized, current KV (a stale/lazy read yields wrong bytes).
        """
        import mlx.core as mx

        mx.eval(*cache.key_caches, *cache.value_caches)
        mx.synchronize()
        views: list[torch.Tensor] = []
        for i in range(cache.num_layers):
            k = _mlx_to_torch_cpu(cache.key_caches[i])
            v = _mlx_to_torch_cpu(cache.value_caches[i])
            views.append(torch.stack([k, v], dim=0))
        return views

    def _ensure_engine(self, cache: MetalPagedKVCache) -> None:
        if self._engine is not None:
            return
        from lmcache.v1.cache_engine import LMCacheEngineBuilder
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.gpu_connector.cpu_connectors import (
            VLLMPagedMemCPUConnectorV2,
        )
        from lmcache.v1.metadata import LMCacheMetadata

        num_layers = cache.num_layers
        num_kv_heads = cache.kv_heads_per_layer[0]
        head_dim = cache.head_dim_per_layer[0]
        block_size = cache.block_size
        # Infer torch dtype from an actual bridged view.
        probe = _mlx_to_torch_cpu(cache.key_caches[0])
        kv_dtype = probe.dtype

        self._num_layers = num_layers
        self._block_size = block_size

        hidden = num_kv_heads * head_dim
        self._connector = VLLMPagedMemCPUConnectorV2(
            hidden, num_layers, layout_hints={"kv_layout": "NHD"}
        )
        kv_shape = (num_layers, 2, self._chunk_size, num_kv_heads, head_dim)
        metadata = LMCacheMetadata(
            model_name=os.environ.get("VLLM_METAL_LMCACHE_MODEL", "vllm-metal"),
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=kv_dtype,
            kv_shape=kv_shape,
        )
        cfg = LMCacheEngineConfig.from_legacy(
            chunk_size=self._chunk_size,
            remote_url=None,
            save_unfull_chunk=True,
        )

        def _noop(*a, **k):
            return None

        self._engine = LMCacheEngineBuilder.get_or_create(
            self.instance_id, cfg, metadata, self._connector, _noop, _noop
        )
        self._engine.post_init()
        logger.info(
            "MetalLMCacheConnector: engine ready (layers=%d, block_size=%d, "
            "num_kv_heads=%d, head_dim=%d, dtype=%s, chunk=%d)",
            num_layers,
            block_size,
            num_kv_heads,
            head_dim,
            kv_dtype,
            self._chunk_size,
        )

    @staticmethod
    def _slot_mapping(
        block_ids: list[int], num_tokens: int, block_size: int, device="cpu"
    ) -> torch.Tensor:
        """Compute per-token physical slot indices from paged block_ids.

        slot(token_j) = block_ids[j // block_size] * block_size + (j % block_size)
        """
        slots = [
            block_ids[j // block_size] * block_size + (j % block_size)
            for j in range(num_tokens)
        ]
        return torch.tensor(slots, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _touched_blocks(self, token_ids_len: int, block_ids: list[int]) -> list[int]:
        """Unique paged blocks spanned by the first ``token_ids_len`` tokens,
        in first-seen order (matches ``_compact_slot_mapping`` dedup)."""
        assert self._block_size is not None
        bs = self._block_size
        n_blocks = -(-token_ids_len // bs)  # ceil
        return list(dict.fromkeys(block_ids[:n_blocks]))

    def _gather_compact(self, cache, blocks: list[int]):
        """Bridge ONLY *blocks* of the live MLX cache into a compact per-layer
        stacked buffer ``[2, len(blocks), bs, nh, hd]`` (torch cpu).

        This is the key perf fix: touch only the request's blocks, not the
        whole (multi-GB) cache.
        """
        import mlx.core as mx
        import torch as _t

        mx.eval(*cache.key_caches, *cache.value_caches)
        mx.synchronize()
        idx = mx.array(blocks, dtype=mx.uint32)
        out = []
        for i in range(cache.num_layers):
            k = _mlx_to_torch_cpu(mx.take(cache.key_caches[i], idx, axis=0))
            v = _mlx_to_torch_cpu(mx.take(cache.value_caches[i], idx, axis=0))
            out.append(_t.stack([k, v], dim=0))
        return out

    def _scatter_compact(self, cache, blocks: list[int], compact) -> None:
        """Write compact per-layer ``[2, len(blocks), bs, nh, hd]`` buffers back
        into the live MLX cache at *blocks* (only those blocks touched)."""
        import mlx.core as mx

        from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx

        idx = mx.array(blocks, dtype=mx.uint32)
        for i in range(cache.num_layers):
            k_src = torch_to_mlx(compact[i][0].contiguous())
            v_src = torch_to_mlx(compact[i][1].contiguous())
            # Scatter the compact blocks into the full cache at `blocks`.
            cache.key_caches[i][idx] = k_src.astype(cache.key_caches[i].dtype)
            cache.value_caches[i][idx] = v_src.astype(cache.value_caches[i].dtype)
        mx.eval(*cache.key_caches, *cache.value_caches)
        mx.synchronize()

    @staticmethod
    def _compact_slot_mapping(
        block_ids: list[int], num_tokens: int, block_size: int, device="cpu"
    ) -> torch.Tensor:
        """Slot mapping into the COMPACT block space (block position 0..k-1)."""
        # token j lives in block_ids[j // bs]; in compact space that block's
        # index is its position within the unique touched-block list.
        uniq = list(dict.fromkeys(block_ids[: -(-num_tokens // block_size)]))
        pos = {b: p for p, b in enumerate(uniq)}
        slots = [
            pos[block_ids[j // block_size]] * block_size + (j % block_size)
            for j in range(num_tokens)
        ]
        return torch.tensor(slots, dtype=torch.long, device=device)

    def store(
        self, cache: MetalPagedKVCache, token_ids: list[int], block_ids: list[int]
    ) -> None:
        """Store the KV for a computed prompt into LMCache.

        Only the request's own blocks are bridged (not the whole cache).
        """
        self._ensure_engine(cache)
        assert self._block_size is not None
        n = len(token_ids)
        if n == 0:
            return
        blocks = self._touched_blocks(n, block_ids)
        compact = self._gather_compact(cache, blocks)
        slot_mapping = self._compact_slot_mapping(block_ids, n, self._block_size)
        tokens = torch.tensor(token_ids, dtype=torch.long)
        self._connector.kv_cache_pointers_on_gpu = {}
        self._engine.store(tokens=tokens, kvcaches=compact, slot_mapping=slot_mapping)
        self.n_store_calls += 1
        self.n_tokens_stored += n
        logger.info(
            "LMCache store: %d tokens / %d blocks (call #%d)",
            n,
            len(blocks),
            self.n_store_calls,
        )

    def lookup(self, token_ids: list[int]) -> int:
        if self._engine is None:
            return 0
        tokens = torch.tensor(token_ids, dtype=torch.long)
        return int(self._engine.lookup(tokens))

    def retrieve(
        self, cache: MetalPagedKVCache, token_ids: list[int], block_ids: list[int]
    ) -> int:
        """Retrieve KV for a prompt from LMCache into the live MLX paged cache.

        Only the request's own blocks are bridged and scattered back (not the
        whole cache), so cost scales with the prompt, not the cache size.
        Returns the number of tokens whose KV was loaded.
        """
        self._ensure_engine(cache)
        assert self._block_size is not None
        n = len(token_ids)
        if n == 0:
            return 0
        blocks = self._touched_blocks(n, block_ids)
        # Snapshot the compact blocks (retrieve overwrites only matched slots;
        # unmatched slots in the same blocks must retain their live values).
        compact = self._gather_compact(cache, blocks)
        slot_mapping = self._compact_slot_mapping(block_ids, n, self._block_size)
        tokens = torch.tensor(token_ids, dtype=torch.long)
        self._connector.kv_cache_pointers_on_gpu = {}
        mask = self._engine.retrieve(
            tokens, kvcaches=compact, slot_mapping=slot_mapping
        )
        hits = int(mask.sum())
        if hits > 0:
            self._scatter_compact(cache, blocks, compact)
        self.n_retrieve_calls += 1
        self.n_tokens_hit += hits
        logger.info(
            "LMCache retrieve: %d/%d tokens hit / %d blocks (call #%d)",
            hits,
            n,
            len(blocks),
            self.n_retrieve_calls,
        )
        return hits

    def stats(self) -> dict:
        return {
            "store_calls": self.n_store_calls,
            "retrieve_calls": self.n_retrieve_calls,
            "tokens_stored": self.n_tokens_stored,
            "tokens_hit": self.n_tokens_hit,
        }
