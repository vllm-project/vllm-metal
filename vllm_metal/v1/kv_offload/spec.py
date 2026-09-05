# SPDX-License-Identifier: Apache-2.0
"""OffloadingSpec for the Metal backend.

``MetalTieringOffloadingSpec`` is upstream's ``TieringOffloadingSpec`` with
the two platform-bound pieces swapped: the ``/dev/shm`` shared region and the
CUDA copy engine. Host-pool sizing (``cpu_bytes_to_use`` / bytes per
offloaded block) and the scheduler-side manager are inherited. Only the
worker-side construction differs: the OffloadingWorker is built from the live
``MetalPagedKVCache`` (per-layer MLX arrays) instead of torch
``CanonicalKVCaches``. Upstream's canonicalization does handle split K/V (it
carries a list of tensors per layer), but its copy engine is gated on
``is_cuda_alike`` and the MLX cache rebinds ``key_caches[L]`` to a fresh array
on every layer call, so a torch alias taken once at registration goes stale
and carries no lazy-graph provenance.

There is one spec, used with or without secondary tiers. On unified memory
the host pool draws on the same RAM as the wired cache, so a separate
CPU-only path would buy nothing the tiering path does not.

Selected via ``kv_connector_extra_config["spec_name"]`` (+
``spec_module_path``), injected by ``MetalPlatform.check_and_update_config``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, override

import vllm.v1.kv_offload.tiering.spec as _tiering_spec_module
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

if TYPE_CHECKING:
    from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
    from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion
    from vllm_metal.v1.kv_offload.worker import MetalKVOffloadWorker

logger = init_logger(__name__)


def validate_metal_support(kv_cache_config: KVCacheConfig) -> None:
    groups = kv_cache_config.kv_cache_groups
    if len(groups) != 1:
        raise NotImplementedError(
            "KV offloading on Metal currently supports a single KV cache "
            f"group; got {len(groups)} (hybrid/sliding-window models are "
            "not supported yet)"
        )
    if not isinstance(groups[0].kv_cache_spec, AttentionSpec):
        # Two different configs land here as UniformTypeKVCacheSpecs rather
        # than as multiple groups, and they need different advice:
        #  - hybrid models (e.g. gemma-4's mixed sliding/full layer_types)
        #  - draft-model spec decode, since #630 registers the draft KV as a
        #    scheduler-managed cache group. Draft and target dims differ in
        #    the usual case, so the merge into one AttentionSpec fails.
        draft = any(name.startswith("draft_layers.") for name in groups[0].layer_names)
        reason = (
            "speculative decoding registers the draft model's KV as part of "
            "this group; drop --speculative-config to use KV offloading"
            if draft
            else "hybrid/sliding-window models are not supported yet"
        )
        raise NotImplementedError(
            "KV offloading on Metal currently supports uniform full-attention "
            f"KV caches only; got {type(groups[0].kv_cache_spec).__name__} "
            f"({reason})"
        )


def route_fs_tiers_to_metal(extra_config: dict[str, Any]) -> None:
    """Point ``fs`` tier configs at the Metal subclass, in place.

    Uses upstream's documented out-of-tree hook: ``SecondaryTierFactory.
    get_tier_class`` imports ``type`` from ``module_path`` when one is given,
    so no registry patching is needed. Applied on the spec rather than in the
    platform hook, so a spec built any other way still gets the Metal tier
    instead of silently falling back to upstream's buffered I/O and 0o644
    block files.
    """
    for tier in extra_config.get("secondary_tiers") or []:
        if isinstance(tier, dict) and tier.get("type") == "fs":
            tier["type"] = "MetalFileSystemTierManager"
            tier["module_path"] = "vllm_metal.v1.kv_offload.fs_tier"


@contextmanager
def _metal_tiering_classes() -> Iterator[None]:
    """Swap in the Metal shared region for the duration of a call.

    Upstream ``TieringOffloadingSpec.get_manager`` constructs its
    module-global ``SharedOffloadRegion`` inline, and that one is /dev/shm
    plus Linux ``madvise``. Rebinding it around ``super().get_manager()``
    inherits the whole orchestration without copying it.

    The ``fs`` tier does NOT need this. It is selected through upstream's
    documented out-of-tree hook instead: ``route_fs_tiers_to_metal`` writes
    ``module_path`` into the tier config and ``SecondaryTierFactory.
    get_tier_class`` imports it.

    Not re-entrant and not thread safe. Metal runs one engine per process
    (the platform hook enforces the uni executor for offloading), so the
    window cannot overlap in practice.
    """
    from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion

    original_region = _tiering_spec_module.SharedOffloadRegion
    _tiering_spec_module.SharedOffloadRegion = MetalSharedOffloadRegion
    try:
        yield
    finally:
        _tiering_spec_module.SharedOffloadRegion = original_region


def gpu_blocks_per_offloaded_block(spec: CPUOffloadingSpec) -> int:
    """GPU blocks coalesced into one offloaded block.

    vLLM 0.25.1 exposed this as ``block_size_factor``, derived from
    ``extra_config["block_size"] // gpu_block_size``. That derivation moved
    into the offloading config builder rather than going away, and the result
    is ``blocks_per_chunk``. Do not recompute it from ``tokens_per_hash``:
    that is the prefix-hash granularity, which equals ``tokens_per_block`` for
    every single-group model, so any ratio built from it is pinned to 1.
    """
    return int(spec.blocks_per_chunk)


def warn_if_pool_undersized(
    spec: CPUOffloadingSpec, kv_cache_config: KVCacheConfig
) -> None:
    """A host pool smaller than the GPU cache silently defeats evict-then-
    reuse: the LRU drops blocks before the GPU cache would have re-requested
    them, and every offload lookup misses (observed live: 32B model, 4 MiB
    blocks, restore count stayed 0). Warn loudly at init."""
    pool_gpu_blocks = spec.num_blocks * gpu_blocks_per_offloaded_block(spec)
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


class MetalTieringOffloadingSpec(TieringOffloadingSpec):
    """Offloading spec for Metal: a host pool plus optional ``fs`` disk tiers.

    Inherits TieringOffloadingSpec wholesale, the configuration surface
    (``secondary_tiers`` et al.) and the scheduler-side ``get_manager``
    orchestration, swapping only the two platform-bound pieces: the
    ``/dev/shm`` SharedOffloadRegion (MetalSharedOffloadRegion) and the CUDA
    worker (MetalKVOffloadWorker over the region).

    Metal support validation and the pool-size warning need the
    KVCacheConfig, which the spec no longer receives; MetalWorker runs both.
    """

    _metal_worker: MetalKVOffloadWorker | None = None

    def __init__(self, config: OffloadingConfig):
        route_fs_tiers_to_metal(config.extra_config)
        super().__init__(config)
        # Persistent tiers key files by PYTHONHASHSEED-dependent content
        # hashes.
        if self.secondary_tier_configs and os.environ.get("PYTHONHASHSEED") is None:
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
            engine_id=self.config.engine_id,
            num_blocks=self.num_blocks,
            rank=rank,
            kv_bytes_per_block=self.kv_bytes_per_chunk,
            cpu_page_size=self.cpu_page_size_per_worker,
        )

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        raise NotImplementedError(
            "The Metal offloading spec builds its worker from the MLX KV cache "
            "via MetalOffloadingConnector.register_kv_caches, not from torch "
            "CanonicalKVCaches"
        )

    def get_metal_worker(self, kv_cache: MetalPagedKVCache) -> MetalKVOffloadWorker:
        """Worker over the live MLX cache. Called by
        MetalOffloadingConnector.register_kv_caches."""
        if self._metal_worker is None:
            # Imported here so the scheduler-side spec (which only calls
            # get_manager) never imports mlx.
            from vllm_metal.v1.kv_offload.worker import MetalKVOffloadWorker

            self._metal_worker = MetalKVOffloadWorker(
                kv_cache,
                block_size_factor=gpu_blocks_per_offloaded_block(self),
                num_cpu_blocks=self.num_blocks,
                # Scheduler and worker share one process (the platform hook
                # enforces the "uni" executor) with a single worker rank.
                region=self._make_region(rank=0),
                expected_block_bytes=self.kv_bytes_per_chunk,
            )
        return self._metal_worker

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
        route_fs_tiers_to_metal(extra_config)
        return super().build_metric_definitions(extra_config)
