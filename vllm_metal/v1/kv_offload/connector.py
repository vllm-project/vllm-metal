# SPDX-License-Identifier: Apache-2.0
"""OffloadingConnector that registers Metal's ``KVCacheStorage``.

Upstream's worker copies through a CUDA-only engine, so the Metal spec
supplies the worker that upstream's registration builds; the scheduler side
is inherited. No mlx import here: the scheduler process loads this module.
"""

from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    MLAAttentionSpec,
)

from vllm_metal.v1.kv_offload.spec import MetalTieringOffloadingSpec

if TYPE_CHECKING:
    from vllm_metal.v1.kv_offload.worker import MetalKVOffloadWorker


def validate_metal_support(kv_cache_config: KVCacheConfig) -> None:
    groups = kv_cache_config.kv_cache_groups
    specs = [group.kv_cache_spec for group in groups]
    if any(
        name.startswith("draft_layers.")
        for group in groups
        for name in group.layer_names
    ):
        # Draft-model spec decode registers its KV in the same group.
        got = "a draft model's KV; drop --speculative-config"
    elif (
        len(specs) == 1
        and isinstance(specs[0], FullAttentionSpec)
        and (
            not isinstance(specs[0], MLAAttentionSpec)
            or type(specs[0]) is MLAAttentionSpec
        )
    ):
        return
    elif (
        len(specs) > 1
        and any(type(spec) is FullAttentionSpec for spec in specs)
        and any(isinstance(spec, MambaSpec) for spec in specs)
        and len(kv_cache_config.prefix_cacheable_group_ids) == len(groups)
        and all(
            type(spec) is FullAttentionSpec
            or (
                isinstance(spec, MambaSpec)
                and spec.mamba_type == MambaAttentionBackendEnum.GDN_ATTN
                and spec.mamba_cache_mode == "align"
            )
            for spec in specs
        )
    ):
        return
    else:
        got = ", ".join(type(spec).__name__ for spec in specs)
        if len(specs) != 1:
            got = f"{len(specs)} groups ({got})"
    raise NotImplementedError(
        "KV offloading on Metal supports a single full-attention KV cache group "
        f"or full attention with aligned GDN state; got {got}"
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
        else:
            recurrent = tuple(
                group.kv_cache_spec
                for group in kv_cache_config.kv_cache_groups
                if isinstance(group.kv_cache_spec, MambaSpec)
            )
            if recurrent:
                # FileMapper knows the attention dtype, but not the state dtypes
                # or the padded layout of the recurrent group's pages.
                layout = sha256(
                    repr((kv_cache_config.kv_cache_layout, recurrent)).encode()
                ).hexdigest()[:16]
                assert vllm_config.kv_transfer_config is not None
                extra = vllm_config.kv_transfer_config.kv_connector_extra_config
                for tier in extra.get("secondary_tiers") or ():
                    tier["hybrid_layout"] = f"gdn-{layout}"
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
        spec.bind_storage(kv_caches)
        self.connector_worker.register_kv_caches(kv_caches.tensors)

    def release_host_pool(self) -> None:
        """Free the host pool for sleep mode.

        With ``reset_connector=True`` (the default), vLLM resets the connector
        before it discards the KV cache. That empties the pool's index, so
        nothing reads the pool until wake. With ``False`` the index survives
        and can point at the freed pages.
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
