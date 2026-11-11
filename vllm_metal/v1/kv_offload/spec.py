# SPDX-License-Identifier: Apache-2.0
"""TieringOffloadingSpec with the Metal shared region and worker.

Upstream's region is a ``/dev/shm`` file and its worker copies with a
CUDA-only engine. Everything else, including the scheduler-side manager and
host-pool sizing, is inherited.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, override

import vllm.v1.kv_offload.tiering.spec as _tiering_spec_module
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

from vllm_metal.v1.kv_offload.config import AUTO_POOL_KEY

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm_metal.attention.caches.storage import KVCacheStorage
    from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion
    from vllm_metal.v1.kv_offload.worker import MetalKVOffloadWorker


def route_fs_tiers_to_metal(extra_config: dict[str, Any]) -> None:
    """Point ``fs`` tier configs at the Metal subclass, in place.

    Uses upstream's ``module_path`` hook in ``SecondaryTierFactory``.
    """
    for tier in extra_config.get("secondary_tiers") or []:
        if isinstance(tier, dict) and tier.get("type") == "fs":
            tier["type"] = "MetalFileSystemTierManager"
            tier["module_path"] = "vllm_metal.v1.kv_offload.fs_tier"


@contextmanager
def _metal_shared_region() -> Iterator[None]:
    """Swap in MetalSharedOffloadRegion while upstream's get_manager runs.

    Not re-entrant. The platform requires the uni executor for offloading, so
    only one engine runs per process.
    """
    from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion

    original_region = _tiering_spec_module.SharedOffloadRegion
    _tiering_spec_module.SharedOffloadRegion = MetalSharedOffloadRegion
    try:
        yield
    finally:
        _tiering_spec_module.SharedOffloadRegion = original_region


class MetalTieringOffloadingSpec(TieringOffloadingSpec):
    """Host pool plus optional ``fs`` disk tiers, on Metal."""

    _metal_worker: MetalKVOffloadWorker | None = None

    def __init__(self, config: OffloadingConfig):
        route_fs_tiers_to_metal(config.extra_config)
        super().__init__(config)

    def _make_region(self, rank: int | None) -> MetalSharedOffloadRegion:
        from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion

        return MetalSharedOffloadRegion(
            engine_id=self.config.engine_id,
            num_chunks=self.num_chunks,
            rank=rank,
            kv_bytes_per_chunk=self.kv_bytes_per_chunk,
            cpu_page_size=self.cpu_page_size_per_worker,
        )

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        raise NotImplementedError(
            "Metal builds its worker from KVCacheStorage via "
            "MetalOffloadingConnector.register_kv_caches"
        )

    def get_metal_worker(self, storage: KVCacheStorage) -> MetalKVOffloadWorker:
        if self._metal_worker is None:
            # Imported here so the scheduler side never imports mlx.
            from vllm_metal.v1.kv_offload.worker import MetalKVOffloadWorker

            self._metal_worker = MetalKVOffloadWorker(
                storage,
                blocks_per_chunk=self.blocks_per_chunk,
                num_cpu_chunks=self.num_chunks,
                # One process, one worker rank (uni executor).
                region=self._make_region(rank=0),
                kv_bytes_per_chunk=self.kv_bytes_per_chunk,
            )
        elif self._metal_worker.storage is not storage:
            raise RuntimeError("KV offloading worker is bound to another KV storage")
        return self._metal_worker

    @override
    def get_manager(self) -> OffloadingManager:
        """Upstream body with the Metal shared region."""
        self._warn_if_pool_under_one_request()
        with _metal_shared_region():
            return super().get_manager()

    def _warn_if_pool_under_one_request(self) -> None:
        """Warn when the host pool holds less than one full-length request.

        vLLM stores a prefill step's chunks all or nothing, so a longer
        prompt may never be stored. With a disk tier, restoring a longer
        prompt stalled the request in testing. Related: vllm#62931.
        """
        if self.num_chunks <= 0:
            return
        tokens_per_chunk = self.tokens_per_block[0] * self.blocks_per_chunk
        request_chunks = self.config.model.max_model_len // tokens_per_chunk
        if self.num_chunks >= request_chunks:
            return
        # Hundredths of a GiB, rounded up.
        min_size = -(-request_chunks * self.kv_bytes_per_chunk * 100 // 2**30)
        size = f"--kv-offloading-size to {min_size // 100}.{min_size % 100:02d} GiB"
        if self.extra_config.get(AUTO_POOL_KEY) is True:
            fix = (
                "This is the default pool. Use a shorter --max-model-len, or "
                f"set {size} or more if memory allows."
            )
        else:
            fix = f"Set {size} or more, or use a shorter --max-model-len."
        effect = "Prompts longer than the pool may not be stored."
        if self.secondary_tier_configs:
            effect += " A request restoring a longer prompt from disk can stall."
        logger.warning(
            "KV offloading host pool holds %d chunks, one --max-model-len "
            "request needs %d. %s %s",
            self.num_chunks,
            request_chunks,
            effect,
            fix,
        )

    @classmethod
    @override
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        """Resolve tier metrics against the Metal fs tier.

        Upstream resolves tier classes here, outside ``get_manager``, so
        without this the Metal tier's metrics would not be registered.
        """
        route_fs_tiers_to_metal(extra_config)
        return super().build_metric_definitions(extra_config)
