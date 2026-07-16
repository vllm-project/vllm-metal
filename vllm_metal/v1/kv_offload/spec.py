# SPDX-License-Identifier: Apache-2.0
"""OffloadingSpecs for the Metal backend.

Both specs reuse their upstream parents' host-pool sizing
(``cpu_bytes_to_use`` / bytes-per-offloaded-block, computed from
``kv_cache_config.kv_cache_tensors``) and scheduler-side managers. Only the
worker-side construction differs: the OffloadingWorker is built from the live
``MetalPagedKVCache`` (per-layer MLX arrays) instead of torch
``CanonicalKVCaches``, which cannot represent vllm-metal's split K/V layout.

- ``MetalOffloadingSpec`` (CPU tier only): host pool is plain numpy memory.
- ``MetalTieringOffloadingSpec`` (CPU + secondary tiers, e.g. ``fs`` disk):
  host pool lives in a shared region so the scheduler-side secondary tiers
  read/write the same bytes.

Selected via ``kv_connector_extra_config["spec_name"]`` (+
``spec_module_path``), injected by ``MetalPlatform.check_and_update_config``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, override

import vllm.v1.kv_offload.tiering.spec as _tiering_spec_module
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingWorker,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

if TYPE_CHECKING:
    from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
    from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion
    from vllm_metal.v1.kv_offload.worker import MetalKVOffloadWorker

logger = init_logger(__name__)


def _validate_metal_support(kv_cache_config: KVCacheConfig) -> None:
    groups = kv_cache_config.kv_cache_groups
    if len(groups) != 1:
        raise NotImplementedError(
            "KV offloading on Metal currently supports a single KV cache "
            f"group; got {len(groups)} (hybrid/sliding-window models are "
            "not supported yet)"
        )
    if not isinstance(groups[0].kv_cache_spec, AttentionSpec):
        # Hybrid models (e.g. gemma-4's mixed sliding/full layer_types) reach
        # here as UniformTypeKVCacheSpecs rather than as multiple groups.
        raise NotImplementedError(
            "KV offloading on Metal currently supports uniform full-attention "
            f"KV caches only; got {type(groups[0].kv_cache_spec).__name__} "
            "(hybrid/sliding-window models are not supported yet)"
        )


@contextmanager
def _metal_tiering_classes() -> Iterator[None]:
    """Route upstream tiering machinery to the Metal classes.

    Upstream ``TieringOffloadingSpec.get_manager`` hardcodes its module-global
    ``SharedOffloadRegion`` (a /dev/shm + Linux-madvise mmap) and resolves the
    ``fs`` tier through ``SecondaryTierFactory``'s registry (buffered I/O,
    0o644 files). Rebinding both around the ``super().get_manager()`` call
    inherits the whole orchestration without copying it. Safe because the
    CUDA path never runs in this process.
    """
    from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory

    from vllm_metal.v1.kv_offload.fs_tier import MetalFileSystemTierManager
    from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion

    original_region = _tiering_spec_module.SharedOffloadRegion
    original_fs = SecondaryTierFactory._registry.get("fs")
    _tiering_spec_module.SharedOffloadRegion = MetalSharedOffloadRegion  # type: ignore[misc]
    SecondaryTierFactory._registry["fs"] = lambda: MetalFileSystemTierManager
    try:
        yield
    finally:
        _tiering_spec_module.SharedOffloadRegion = original_region  # type: ignore[misc]
        if original_fs is None:
            del SecondaryTierFactory._registry["fs"]
        else:
            SecondaryTierFactory._registry["fs"] = original_fs


class MetalOffloadingSpecBase:
    """Mixin giving a spec a Metal (MLX) worker-side OffloadingWorker.

    MetalOffloadingConnector.register_kv_caches recognizes specs by this
    base and calls get_metal_worker with the live MetalPagedKVCache.
    """

    _metal_worker: MetalKVOffloadWorker | None = None

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        raise NotImplementedError(
            "Metal offloading specs build their worker from the MLX KV cache "
            "via MetalOffloadingConnector.register_kv_caches, not from torch "
            "CanonicalKVCaches"
        )

    def _make_worker_region(self) -> MetalSharedOffloadRegion | None:
        """Shared region backing the host pool; None keeps plain numpy."""
        return None

    def get_metal_worker(self, kv_cache: MetalPagedKVCache) -> MetalKVOffloadWorker:
        if self._metal_worker is None:
            # Imported here so the scheduler-side spec (which only calls
            # get_manager) never imports mlx.
            from vllm_metal.v1.kv_offload.worker import MetalKVOffloadWorker

            self._metal_worker = MetalKVOffloadWorker(
                kv_cache,
                block_size_factor=self.block_size_factor,  # type: ignore[attr-defined]
                num_cpu_blocks=self.num_blocks,  # type: ignore[attr-defined]
                region=self._make_worker_region(),
                expected_block_bytes=self.kv_bytes_per_offloaded_block,  # type: ignore[attr-defined]
            )
        return self._metal_worker


def _warn_if_pool_undersized(
    spec: CPUOffloadingSpec, kv_cache_config: KVCacheConfig
) -> None:
    """A host pool smaller than the GPU cache silently defeats evict-then-
    reuse: the LRU drops blocks before the GPU cache would have re-requested
    them, and every offload lookup misses (observed live: 32B model, 4 MiB
    blocks, restore count stayed 0). Warn loudly at init."""
    pool_gpu_blocks = spec.num_blocks * spec.block_size_factor
    if 0 < pool_gpu_blocks < kv_cache_config.num_blocks:
        logger.warning(
            "KV offloading host pool (%d GPU-block equivalents) is SMALLER "
            "than the GPU KV cache (%d blocks). Prefixes evicted from the "
            "GPU will usually already be evicted from the pool too, so "
            "offload restores will rarely hit. Increase "
            "--kv-offloading-size (or add a disk tier via secondary_tiers).",
            pool_gpu_blocks,
            kv_cache_config.num_blocks,
        )


class MetalOffloadingSpec(MetalOffloadingSpecBase, CPUOffloadingSpec):
    """CPU-tier-only offloading spec for Metal."""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)
        _validate_metal_support(kv_cache_config)
        _warn_if_pool_undersized(self, kv_cache_config)


class MetalTieringOffloadingSpec(MetalOffloadingSpecBase, TieringOffloadingSpec):
    """Multi-tier offloading spec for Metal (CPU primary + e.g. ``fs`` disk).

    Inherits TieringOffloadingSpec wholesale — configuration surface
    (``secondary_tiers`` et al.) and the scheduler-side ``get_manager``
    orchestration — swapping only the two platform-bound pieces: the
    ``/dev/shm`` SharedOffloadRegion (→ MetalSharedOffloadRegion) and the
    CUDA worker (→ MetalKVOffloadWorker over the region).
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)
        _validate_metal_support(kv_cache_config)
        # Tiering: a small RAM pool is fine (disk holds the working set), so
        # no pool-size warning here — but persistent tiers key files by
        # PYTHONHASHSEED-dependent content hashes.
        if os.environ.get("PYTHONHASHSEED") is None:
            logger.warning(
                "Secondary KV tiers are configured but PYTHONHASHSEED is "
                "unset: block filenames are hash-seeded per process, so "
                "blocks written now will NOT be found after a server "
                "restart (or by other instances sharing the store). Set "
                "PYTHONHASHSEED=0 for cross-restart reuse."
            )

    def _make_region(self, rank: int | None) -> MetalSharedOffloadRegion:
        from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion

        return MetalSharedOffloadRegion(
            instance_id=self.vllm_config.instance_id,
            num_blocks=self.num_blocks,
            rank=rank,
            kv_bytes_per_block=self.kv_bytes_per_offloaded_block,
            cpu_page_size=self.cpu_page_size_per_worker,
        )

    @override
    def _make_worker_region(self) -> MetalSharedOffloadRegion:
        # Metal runs scheduler and worker in one process (the platform hook
        # enforces the "uni" executor for tiering) with a single worker rank.
        return self._make_region(rank=0)

    @override
    def get_manager(self) -> OffloadingManager:
        """Delegate to the upstream body with the Metal classes routed in
        (shared region and macOS-tuned ``fs`` tier; see
        ``_metal_tiering_classes``)."""
        with _metal_tiering_classes():
            return super().get_manager()

    @classmethod
    @override
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        """Resolve secondary-tier metric definitions against the METAL tier
        classes. The upstream body resolves each tier via
        ``SecondaryTierFactory.get_tier_class`` at stat-logger construction —
        outside ``get_manager``'s rebinding scope — so without this the ``fs``
        tier's definitions would come from the upstream class and any metric
        a Metal tier registers would KeyError at observation time."""
        with _metal_tiering_classes():
            return super().build_metric_definitions(extra_config)
