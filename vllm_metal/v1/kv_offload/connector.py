# SPDX-License-Identifier: Apache-2.0
"""OffloadingConnector that registers Metal's ``KVCacheStorage``.

Upstream's worker copies through a CUDA-only engine, so the Metal spec
supplies the worker that upstream's registration builds; the scheduler side
is inherited. No mlx import here: the scheduler process loads this module.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, ClassVar

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
    elif isinstance(groups[0].kv_cache_spec, MLAAttentionSpec) and (
        type(groups[0].kv_cache_spec) is not MLAAttentionSpec
    ):
        # Subclasses such as HiddenStateCacheSpec do not hold attention KV.
        got = type(groups[0].kv_cache_spec).__name__
    elif isinstance(groups[0].kv_cache_spec, FullAttentionSpec):
        # MLAAttentionSpec is a FullAttentionSpec: one latent row per token.
        return
    else:
        got = type(groups[0].kv_cache_spec).__name__
    raise NotImplementedError(
        "KV offloading on Metal supports a single full-attention KV cache "
        f"group; got {got}"
    )


class MetalOffloadingConnector(OffloadingConnector):
    # The uni executor runs these in the worker's process, one per engine id.
    _scheduler_connectors: ClassVar[
        weakref.WeakValueDictionary[str, MetalOffloadingConnector]
    ] = weakref.WeakValueDictionary()

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
        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        assert kv_transfer_config.engine_id is not None
        self._engine_id = kv_transfer_config.engine_id
        if role == KVConnectorRole.SCHEDULER:
            self._scheduler_connectors[self._engine_id] = self

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
        spec.bind_storage(kv_caches)
        self.connector_worker.register_kv_caches(kv_caches.tensors)

    def release_host_pool(self) -> None:
        """Free the host pool for sleep mode.

        vLLM resets the connector first only with ``reset_connector=True``.
        Reset it here either way, so no offload hit outlives the freed pages.
        This departs from upstream, which keeps the offloaded blocks with
        ``False``: on unified memory the pool shares the RAM that sleep frees.
        Disk tiers keep their blocks either way.
        """
        if (worker := self._metal_worker()) is not None:
            self._scheduler_connectors[self._engine_id].reset_cache()
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
