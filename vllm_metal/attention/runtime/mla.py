# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import mlx.core as mx
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_metal.attention.caches.mla_cache import MLAPagedLatentCache
from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.attention.impls.mla import MLAPagedAttentionWrapper
from vllm_metal.attention.patching import walk_and_wrap
from vllm_metal.attention.runtime.base import PagedAttentionRuntimeBase

logger = init_logger(__name__)


class MLAPagedAttentionRuntime(PagedAttentionRuntimeBase):
    """Paged attention runtime for MLA models.

    Implements the PagedAttentionRuntime protocol. Uses MLX-native
    scatter/gather (cache I/O only; attention is MLX SDPA by default
    or an opt-in single-pass Metal kernel, RFC #360) because MLA
    latents do not fit the standard (num_heads, head_dim) kernel
    layout.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        latent_dim: int,
        block_size: int,
        dtype: mx.Dtype,
    ) -> None:
        self._num_layers = num_layers
        self._latent_dim = latent_dim
        self._block_size = block_size
        self._dtype = dtype
        self._cache = None
        self._storage: KVCacheStorage | None = None
        self._scheduler_group_indices = (0,)
        self._wrappers: list[MLAPagedAttentionWrapper] = []

    def initialize(self, num_blocks: int) -> None:
        # TODO: Move draft-model speculation's target KV to initialize_from_config.
        # Then remove this initializer and MLAPagedLatentCache's count-based
        # allocation.
        self._cache = MLAPagedLatentCache(
            num_layers=self._num_layers,
            latent_dim=self._latent_dim,
            num_blocks=num_blocks,
            block_size=self._block_size,
            dtype=self._dtype,
        )

    def initialize_from_config(
        self, config: KVCacheConfig, layer_names: tuple[str, ...]
    ) -> None:
        """Use the physical allocation and views planned by vLLM."""
        groups = tuple(
            i
            for i, group in enumerate(config.kv_cache_groups)
            if any(name in layer_names for name in group.layer_names)
        )
        # The wrapper reads one block table for every MLA layer.
        if len(groups) != 1:
            raise NotImplementedError(
                f"MLA layers must share one scheduler KV group, got groups {groups}"
            )
        self._storage = KVCacheStorage(config)
        self._cache = MLAPagedLatentCache.from_upstream(self._storage, layer_names)
        self._scheduler_group_indices = groups
        logger.info(
            "Shared MLA latent cache: %d blocks, %.2f GiB across %d Metal regions",
            config.num_blocks,
            self._storage.nbytes / (1 << 30),
            len(self._storage.buffers),
        )

    def kv_scheduler_group_indices(self) -> tuple[int, ...]:
        """Return scheduler KV groups consumed by this runtime."""
        self._require_initialized("kv_scheduler_group_indices")
        return self._scheduler_group_indices

    def extend_forward_eval_outputs(self, outputs: list[mx.array]) -> None:
        if self._storage is not None:
            self._storage.depend(outputs)
            outputs.extend(self._storage.buffers)

    @property
    def storage(self) -> KVCacheStorage:
        if self._storage is None:
            raise RuntimeError("Shared KV storage has not been initialized")
        return self._storage

    def patch_model(self, model: Any) -> int:
        cache = self._require_initialized("patch_model")
        return self._patch_model(model, cache)

    def _patch_model(self, model: Any, latent_cache: MLAPagedLatentCache) -> int:
        wrappers: list[MLAPagedAttentionWrapper] = []

        def wrap_layer(layer_idx: int, attn: Any) -> Any:
            if isinstance(attn, MLAPagedAttentionWrapper):
                # Already patched — refresh cache reference in place.
                attn.rebind_cache(latent_cache, cache_idx=layer_idx)
                wrapper = attn
            else:
                wrapper = MLAPagedAttentionWrapper(attn, layer_idx, latent_cache)
            wrappers.append(wrapper)
            return wrapper

        patched = walk_and_wrap(model, wrap_layer)
        self._wrappers = wrappers
        return patched

    def warm_up(self) -> None:
        super().warm_up()
        self._log_decode_path()

    def _log_decode_path(self) -> None:
        """Say at startup whether decode-only batches take the single-pass kernel."""
        mismatches = [w.decode_kernel_mismatch() for w in self._wrappers]
        if not mismatches:
            return
        on_sdpa = [m for m in mismatches if m is not None]
        if not on_sdpa:
            logger.info(
                "Metal: MLA decode-only batches take the single-pass Metal kernel"
            )
            return
        logger.info(
            "Metal: MLA decode-only batches take the MLX SDPA path on %d of %d "
            "layers (%s)",
            len(on_sdpa),
            len(mismatches),
            "; ".join(sorted(set(on_sdpa))),
        )
