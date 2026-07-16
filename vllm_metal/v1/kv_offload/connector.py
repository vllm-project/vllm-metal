# SPDX-License-Identifier: Apache-2.0
"""OffloadingConnector subclass for the Metal backend.

The stock worker half canonicalizes torch KV tensors into int8
``(num_blocks, page_size_bytes)`` storage views — impossible for vllm-metal's
per-layer split K/V MLX arrays. This subclass overrides only the registration
entry point; the scheduler half, job bookkeeping, deferred stores, and
completion reporting are all inherited unchanged.

``MetalModelRunner`` owns the ``register_kv_caches`` call site and passes the
live ``MetalPagedKVCache`` instead of a torch tensor dict.

``MetalPlatform.check_and_update_config`` routes here by setting
``kv_transfer_config.kv_connector = "MetalOffloadingConnector"`` +
``kv_connector_module_path`` (it performs the ``--kv-offloading-size``
translation itself, before vLLM's own translation would run).

No mlx import at module scope: this module is also imported by the
scheduler/engine-core process to resolve the SCHEDULER-role connector class,
which must not initialize MLX.
"""

from __future__ import annotations

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)

from vllm_metal.v1.kv_offload.spec import MetalOffloadingSpecBase


class MetalOffloadingConnector(OffloadingConnector):
    def register_kv_caches(self, kv_caches) -> None:  # type: ignore[override]
        from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache

        assert self.connector_worker is not None
        if not isinstance(kv_caches, MetalPagedKVCache):
            raise TypeError(
                "MetalOffloadingConnector.register_kv_caches expects the "
                f"MetalPagedKVCache, got {type(kv_caches).__name__}"
            )
        spec = self.connector_worker.spec
        if not isinstance(spec, MetalOffloadingSpecBase):
            raise TypeError(
                "MetalOffloadingConnector requires a Metal offloading spec "
                f"(got {type(spec).__name__}); check kv_connector_extra_config"
            )
        # Mirrors OffloadingConnectorWorker.register_kv_caches minus the
        # torch canonicalization: _init_worker's only job is populating
        # self.worker from the spec.
        self.connector_worker.worker = spec.get_metal_worker(kv_caches)
