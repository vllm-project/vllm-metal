# SPDX-License-Identifier: Apache-2.0
"""OffloadingConnector that registers Metal's ``KVCacheStorage``.

Upstream's worker copies through a CUDA-only engine, so only worker
registration changes; the scheduler side is inherited. No mlx import here:
the scheduler process loads this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    MLAAttentionSpec,
)

from vllm_metal.v1.kv_offload.spec import MetalTieringOffloadingSpec

if TYPE_CHECKING:
    from vllm_metal.v1.kv_offload.worker import MetalKVOffloadWorker


def validate_metal_support(kv_cache_config: KVCacheConfig) -> None:
    groups = kv_cache_config.kv_cache_groups
    if len(groups) != 1:
        got = f"{len(groups)} groups"
    elif any(name.startswith("draft_layers.") for name in groups[0].layer_names):
        # Draft-model spec decode registers its KV in the same group.
        got = "a draft model's KV; drop --speculative-config"
    elif isinstance(groups[0].kv_cache_spec, FullAttentionSpec) and not isinstance(
        groups[0].kv_cache_spec, MLAAttentionSpec
    ):
        return
    else:
        got = type(groups[0].kv_cache_spec).__name__
    raise NotImplementedError(
        "KV offloading on Metal supports a single full-attention KV cache "
        f"group; got {got}"
    )


class MetalOffloadingConnector(OffloadingConnector):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        # The worker builds the connector before it allocates the KV cache.
        if role == KVConnectorRole.WORKER:
            validate_metal_support(kv_cache_config)
        super().__init__(vllm_config, role, kv_cache_config)

    def register_kv_caches(self, kv_caches) -> None:
        from vllm_metal.attention.caches.storage import KVCacheStorage

        assert self.connector_worker is not None
        if not isinstance(kv_caches, KVCacheStorage):
            raise TypeError(
                "MetalOffloadingConnector.register_kv_caches expects the "
                f"runtime's KVCacheStorage, got {type(kv_caches).__name__}"
            )
        spec = self.connector_worker.spec
        if not isinstance(spec, MetalTieringOffloadingSpec):
            raise TypeError(
                "MetalOffloadingConnector requires MetalTieringOffloadingSpec "
                f"(got {type(spec).__name__}); check kv_connector_extra_config"
            )
        # Upstream's _init_worker only sets self.worker.
        self.connector_worker.worker = spec.get_metal_worker(kv_caches)

    def release_host_pool(self) -> None:
        """Free the host pool for sleep mode.

        vLLM resets the connector before it discards the KV cache. That
        empties the pool's index, so nothing reads the pool until wake.
        """
        if (worker := self._metal_worker()) is not None:
            worker.release_host_pool()

    def restore_host_pool(self) -> None:
        if (worker := self._metal_worker()) is not None:
            worker.restore_host_pool()

    def _metal_worker(self) -> MetalKVOffloadWorker | None:
        # Worker side only, so importing mlx here is fine.
        from vllm_metal.v1.kv_offload.worker import MetalKVOffloadWorker

        if self.connector_worker is None:
            return None
        worker = self.connector_worker.worker
        return worker if isinstance(worker, MetalKVOffloadWorker) else None
