# SPDX-License-Identifier: Apache-2.0
"""Expose Metal's live MLX KV cache to vLLM's generic KVConnector contract.

This is the *only* Metal-specific KV-connector code. It is connector-agnostic:
it knows that *a* generic vLLM ``KVConnector`` may be configured, but never
which one. It contains no KV-cache library name, no cache format/layout, no
connector metadata parsing, and no per-request block tracking. Its entire job
is to present Metal's native per-layer MLX key/value arrays to vLLM as ordinary
CPU ``torch.Tensor`` views, and to drive vLLM's generic connector step lifecycle
around the Metal forward.

Why this is intrinsic to Metal (and cannot live in vLLM core or in a connector):
Metal's paged KV cache lives in MLX unified memory as two separate contiguous
``mx.array`` per layer (key and value). Only Metal can turn those MLX arrays
into torch views (via the MLX<->torch unified-memory bridge). vLLM core consumes
``torch.Tensor`` and must not import MLX; a connector consumes whatever tensors
it is handed and must not import MLX either. The MLX->torch aliasing therefore
belongs here, in the backend that owns the MLX buffers.

Aliasing: ``mlx_to_torch(arr, device="cpu")`` returns a torch tensor that shares
the MLX array's unified-memory buffer (verified: write-through in both
directions, K and V never alias each other, buffer stable for the array's
lifetime). Because the export is a live alias, a connector's writes on RETRIEVE
land directly in the live MLX cache and its reads on STORE observe the live KV
-- with no fused ``torch.stack`` staging buffer, no MLX->staging store copy, and
no staging->MLX load writeback.

Memory ordering and pointer stability: ``mlx_to_torch`` calls ``mx.eval`` on the
array, which materializes the lazy attention writes into the buffer before the
connector reads it. Metal's fused ``reshape_and_cache`` kernel writes IN PLACE,
so the K/V buffer addresses are stable for the lifetime of the connector
registration. The bridge therefore registers the live aliases EXACTLY ONCE, and
per step only (a) materializes pending MLX work (the "make prior writes visible"
barrier) and (b) verifies the live buffer pointers still match the ones
registered. If a pointer moved (e.g. a hypothetical out-of-place cache
implementation), the bridge FAILS FAST -- it does not silently re-register a
second context (register_kv_caches is not defined as a replace/update
operation) and does not silently continue with stale pointers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput

    from vllm_metal.v1.model_runner import MetalModelRunner

logger = init_logger(__name__)


# The generic vLLM lifecycle helpers this integration requires. Presence is
# checked (capability detection) only when a KV connector is actually
# configured, so importing vllm-metal and running normal inference against an
# older vLLM that lacks them is unaffected.
_REQUIRED_LIFECYCLE_METHODS = (
    "begin_kv_connector_step",
    "finish_kv_connector_step",
    "abort_kv_connector_step",
)


class MetalKVConnectorCapabilityError(RuntimeError):
    """Raised when a KV connector is configured but the installed vLLM lacks the
    generic imperative KVConnector step lifecycle this integration requires."""


class MetalKVConnectorBridge:
    """Bridge Metal's live MLX KV cache to vLLM's generic KVConnector lifecycle.

    Owns no connector knowledge. All connector calls go through vLLM's generic
    ``KVConnectorModelRunnerMixin`` step helpers and the public
    ``get_kv_transfer_group()`` SPI. Holds a back-reference to the runner only to
    read the current live MLX buffers and their vLLM layer names.
    """

    def __init__(self, runner: MetalModelRunner) -> None:
        self._runner = runner
        self._kv_cache_config: KVCacheConfig | None = None
        # A step handle kept between execute_model (submit) and sample_tokens
        # (complete) for the async paged path. None when no step is in flight.
        self._pending_step = None
        # The registered K/V buffer identities (data_ptr) and shapes, recorded
        # once at registration. Used to verify pointer stability before each
        # external KV access (fail fast if the storage moved).
        self._registered_ptrs: tuple[int, ...] | None = None
        self._registered_shapes: tuple[tuple[int, ...], ...] | None = None

    # -- registration -------------------------------------------------------

    def on_initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Register Metal's live split K/V views with the configured connector.

        No-op without a connector. When a connector IS configured, first verify
        the installed vLLM provides the required generic lifecycle (capability
        detection) and fail early with a clear version error otherwise. The
        registration payload is ``{layer_name: (key_view, value_view)}`` -- two
        independent live CPU torch aliases of the layer's MLX key and value
        arrays. No fused tensor is allocated. vLLM core forwards this mapping
        verbatim to the connector, which is responsible for interpreting the
        split representation. Registration happens EXACTLY ONCE; the K/V buffer
        identities and shapes are recorded for later pointer-stability checks.
        """
        self._kv_cache_config = kv_cache_config
        if not has_kv_transfer_group():
            return
        self._require_lifecycle_capability()
        views = self._export_live_split_kv()
        if views is None:
            return
        get_kv_transfer_group().register_kv_caches(views)
        self._registered_ptrs = self._live_buffer_ptrs()
        self._registered_shapes = self._live_buffer_shapes()

    # -- lifecycle (generic; no connector knowledge) ------------------------

    def has_connector(self) -> bool:
        return has_kv_transfer_group()

    def no_forward(self, scheduler_output: SchedulerOutput):
        """Drive the connector on a zero-token step (async load completion)."""
        from vllm.v1.worker.kv_connector_model_runner_mixin import (
            KVConnectorModelRunnerMixin,
        )

        return KVConnectorModelRunnerMixin.kv_connector_no_forward(
            scheduler_output, self._runner.vllm_config
        )

    def begin_step(self, scheduler_output: SchedulerOutput) -> None:
        """Begin this step's connector work: bind metadata + start loads.

        Uses vLLM's generic ``begin_kv_connector_step``. Metal's fused KV-write
        kernel writes IN PLACE, so the aliases registered once in
        ``on_initialize_kv_cache`` stay valid for the connector's lifetime. The
        only per-step obligations are (1) memory ordering -- ``mx.eval`` prior
        lazy KV writes so the connector reads current data on a store -- and (2)
        pointer stability -- verify the live buffers still match the registered
        pointers, failing fast otherwise. No re-export, no re-registration, no
        data copy.
        """
        from vllm.v1.worker.kv_connector_model_runner_mixin import (
            KVConnectorModelRunnerMixin,
        )

        self._materialize_live_kv()
        self._verify_pointer_stability()
        self._pending_step = KVConnectorModelRunnerMixin.begin_kv_connector_step(
            scheduler_output
        )

    def finish_step(self) -> KVConnectorOutput | None:
        """Complete this step's connector work and return its output.

        Uses vLLM's generic ``finish_kv_connector_step`` (wait_for_save,
        get_finished, invalid blocks, stats/events/meta, clear metadata). No
        staging sync is performed: the connector already read/wrote the live
        aliases directly.
        """
        from vllm.v1.worker.kv_connector_model_runner_mixin import (
            KVConnectorModelRunnerMixin,
        )

        if self._pending_step is None:
            return None
        step = self._pending_step
        self._pending_step = None
        return KVConnectorModelRunnerMixin.finish_kv_connector_step(step)

    def abort_step(self) -> None:
        """Abort the in-flight connector step after a failed forward.

        Uses vLLM's generic ``abort_kv_connector_step`` to clear connector
        metadata exactly once so the next scheduler step does not inherit stale
        metadata. No-op when no step is in flight. Does not swallow the caller's
        exception.
        """
        from vllm.v1.worker.kv_connector_model_runner_mixin import (
            KVConnectorModelRunnerMixin,
        )

        if self._pending_step is None:
            return
        step = self._pending_step
        self._pending_step = None
        KVConnectorModelRunnerMixin.abort_kv_connector_step(step)

    # -- MLX -> torch live split-K/V export (the intrinsic Metal part) ------

    def _layer_names(self) -> list[str]:
        cfg = self._kv_cache_config
        if cfg is None:
            return []
        names: list[str] = []
        for group in cfg.kv_cache_groups:
            names.extend(group.layer_names)
        return names

    def _live_kv_arrays(self):
        """Return ``(key_caches, value_caches)`` MLX arrays, or ``None``."""
        runtime = self._runner._paged_attention_runtime
        if runtime is None:
            return None
        kv_cache = getattr(runtime, "kv_cache", None)
        key_caches = getattr(kv_cache, "key_caches", None)
        value_caches = getattr(kv_cache, "value_caches", None)
        if not key_caches or not value_caches:
            return None
        return key_caches, value_caches

    def _export_live_split_kv(
        self,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]] | None:
        """Build ``{layer_name: (key_view, value_view)}`` over live MLX buffers.

        Each view is a zero-copy CPU torch alias of the layer's MLX array. No
        fused tensor is allocated. Raises if the vLLM layer-name mapping does
        not match the Metal cache -- layer identity is a correctness property
        and is never silently invented.
        """
        arrays = self._live_kv_arrays()
        if arrays is None:
            return None
        key_caches, value_caches = arrays
        from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch

        names = self._layer_names()
        n = len(key_caches)
        if len(names) != n:
            raise ValueError(
                "Metal KV cache layer count does not match the vLLM "
                f"KVCacheConfig layer mapping: configured {len(names)} layer "
                f"name(s) {names!r} but the Metal paged cache has {n} layer(s). "
                "Layer identity is required for correct KV transfer; refusing "
                "to fall back to positional layer names."
            )
        views: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for i, name in enumerate(names):
            key_view = mlx_to_torch(key_caches[i], device="cpu")
            value_view = mlx_to_torch(value_caches[i], device="cpu")
            views[name] = (key_view, value_view)
        return views

    def _materialize_live_kv(self) -> None:
        """Force pending lazy MLX KV writes to complete (memory-ordering barrier).

        A prior step's KV writes are lazy in MLX's graph, so ``mx.eval`` them
        before the connector reads the CPU alias on a store. Cheap eval barrier,
        not a data copy and not a re-registration.
        """
        arrays = self._live_kv_arrays()
        if arrays is None:
            return
        key_caches, value_caches = arrays
        import mlx.core as mx

        mx.eval(*key_caches, *value_caches)

    def _live_buffer_ptrs(self) -> tuple[int, ...] | None:
        """Return the data_ptr of every live K/V buffer, or ``None``.

        Cheap and allocation-free: exports zero-copy torch aliases and reads
        their ``data_ptr`` (``mlx_to_torch`` shares the buffer; no copy).
        """
        arrays = self._live_kv_arrays()
        if arrays is None:
            return None
        key_caches, value_caches = arrays
        from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch

        ptrs: list[int] = []
        for arr in list(key_caches) + list(value_caches):
            ptrs.append(mlx_to_torch(arr, device="cpu").data_ptr())
        return tuple(ptrs)

    def _live_buffer_shapes(self) -> tuple[tuple[int, ...], ...] | None:
        """Return the shape of every live K/V buffer, or ``None``."""
        arrays = self._live_kv_arrays()
        if arrays is None:
            return None
        key_caches, value_caches = arrays
        return tuple(tuple(a.shape) for a in list(key_caches) + list(value_caches))

    def _verify_pointer_stability(self) -> None:
        """Fail fast if the live K/V storage moved after registration.

        ``register_kv_caches`` is not a replace/update operation, so this
        integration requires the Metal KV storage to be stable for the lifetime
        of the connector registration (guaranteed by the in-place fused
        ``reshape_and_cache`` kernel). If a live buffer pointer no longer matches
        the pointer registered at init, raise -- do not silently continue with
        stale pointers and do not silently register a second context.
        """
        if self._registered_ptrs is None:
            return
        current = self._live_buffer_ptrs()
        if current is not None and current != self._registered_ptrs:
            raise RuntimeError(
                "Metal KV storage moved after connector registration: the live "
                "key/value buffer addresses no longer match the ones registered "
                "at initialization. Dynamic re-registration is not part of the "
                "current KVConnector contract, so this integration requires a "
                "cache implementation whose buffers are stable for the lifetime "
                "of the registration (the fused in-place reshape_and_cache path). "
                "An out-of-place cache implementation is unsupported here."
            )

    def _require_lifecycle_capability(self) -> None:
        """Fail early if the installed vLLM lacks the required step lifecycle.

        Capability detection (not a private-module import): checks the generic
        ``KVConnectorModelRunnerMixin`` for the imperative lifecycle methods this
        integration drives. Only called when a KV connector is configured, so
        importing vllm-metal and running normal inference on an older vLLM is
        unaffected. The error is generic (no concrete-connector name).
        """
        from vllm.v1.worker.kv_connector_model_runner_mixin import (
            KVConnectorModelRunnerMixin,
        )

        missing = [
            m
            for m in _REQUIRED_LIFECYCLE_METHODS
            if not hasattr(KVConnectorModelRunnerMixin, m)
        ]
        if missing:
            raise MetalKVConnectorCapabilityError(
                "This vllm-metal KVConnector integration requires a vLLM version "
                "that provides the imperative KVConnector step lifecycle "
                f"(missing: {', '.join(missing)}). Upgrade vLLM to a version that "
                "includes the generic begin/finish/abort KVConnector step API."
            )
