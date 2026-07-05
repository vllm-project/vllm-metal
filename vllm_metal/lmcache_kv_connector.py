# SPDX-License-Identifier: Apache-2.0
"""vLLM v1 KVConnector that routes the Metal (MLX) paged KV cache through LMCache.

This is the *correct* in-engine integration (as opposed to a worker-only prefix
skip, which desynchronizes the scheduler and produces wrong output). It plugs
into vLLM's scheduler-side KVConnector protocol so the scheduler and worker
agree on the externally-cached prefix:

  scheduler role:
    - ``get_num_new_matched_tokens``: ask LMCache how many (block-aligned)
      prefix tokens are cached beyond what vLLM already has locally; the
      scheduler adds these to ``num_computed_tokens`` and only schedules the
      suffix (this is what yields the TTFT win, safely).
    - ``build_connector_meta``: record which requests need a load (matched) or
      a store (new prompt), with their assigned block_ids.

  worker role (driven by MetalModelRunner):
    - ``start_load_kv``: before the forward, load matched-prefix KV from LMCache
      into the live MLX paged cache at the scheduler-assigned blocks.
    - ``wait_for_save``: after the forward, store freshly computed prompt KV to
      LMCache (whole-model; the MLX runner has no per-layer save callback).

Physical KV movement uses :class:`MetalLMCacheConnector` (MLX<->torch unified-
memory bridge + LMCache CPU connector + python_ops_fallback). No CUDA.

Enable via ``kv_transfer_config`` with ``kv_connector="MetalLMCacheKVConnector"``
and ``kv_connector_module_path="vllm_metal.lmcache_kv_connector"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Process-wide handle to the worker's live MLX LMCache bridge. In uniproc mode
# (VLLM_ENABLE_V1_MULTIPROCESSING=0, the Apple-Silicon default) the scheduler
# and worker share the interpreter, so the scheduler-role connector reuses the
# exact same MetalLMCacheConnector the worker bound — guaranteeing scheduler
# lookups see what the worker stored (same LMCache engine).
_WORKER_MLX_BRIDGE: Any = None
_WORKER_PAGED_CACHE: Any = None


def _align_down(n: int, block: int) -> int:
    return (n // block) * block


@dataclass
class _ReqLoad:
    req_id: str
    token_ids: list[int]
    block_ids: list[int]
    num_load: int  # block-aligned prefix token count to load from LMCache


@dataclass
class _ReqStore:
    req_id: str
    token_ids: list[int]
    block_ids: list[int]


@dataclass
class MetalLMCacheConnectorMetadata(KVConnectorMetadata):
    loads: list[_ReqLoad] = field(default_factory=list)
    stores: list[_ReqStore] = field(default_factory=list)


class MetalLMCacheKVConnector(KVConnectorBase_V1):
    """KVConnector bridging vLLM's scheduler to LMCache over the MLX cache."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Any = None,
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        self._block_size = vllm_config.cache_config.block_size
        # LMCache chunk granularity (matched prefixes must be chunk-aligned).
        import os

        self._chunk = int(os.environ.get("VLLM_METAL_LMCACHE_CHUNK_SIZE", "256"))
        # scheduler-role bookkeeping
        self._requests_need_load: dict[str, Request] = {}
        # worker-role: the live MLX cache bridge, set by the model runner.
        self._mlx: Any = None  # MetalLMCacheConnector
        self._paged_cache: Any = None  # MetalPagedKVCache

    # ---- called by MetalModelRunner (worker role) to wire the MLX cache ----
    def bind_metal_cache(self, mlx_connector, paged_cache) -> None:
        global _WORKER_MLX_BRIDGE, _WORKER_PAGED_CACHE
        self._mlx = mlx_connector
        self._paged_cache = paged_cache
        _WORKER_MLX_BRIDGE = mlx_connector
        _WORKER_PAGED_CACHE = paged_cache

    # ============================ scheduler role ============================
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        token_ids = list(request.prompt_token_ids or [])
        if not token_ids:
            return 0, False
        mlx = self._scheduler_mlx()
        if mlx is None:
            return 0, False
        try:
            matched = mlx.lookup(token_ids)
        except Exception as exc:  # pragma: no cover
            logger.warning("LMCache lookup failed: %s", exc)
            return 0, False
        # chunk- and block-aligned; must leave >=1 token to actually run forward
        gran = self._chunk if self._chunk % self._block_size == 0 else self._block_size
        matched = _align_down(min(matched, len(token_ids) - 1), gran)
        external = matched - num_computed_tokens
        if external <= 0:
            return 0, False
        logger.info(
            "LMCache external hit: +%d tokens (req %s)", external, request.request_id
        )
        return external, False  # synchronous load

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = (
                request,
                num_external_tokens,
            )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = MetalLMCacheConnectorMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = list(new_req.prompt_token_ids or [])
            rid = new_req.req_id
            block_ids = list(new_req.block_ids[0])
            if rid in self._requests_need_load:
                # The scheduler has ALREADY folded the external tokens into
                # ``new_req.num_computed_tokens`` (scheduler.py sets
                # request.num_computed_tokens = local + external). So the KV
                # prefix to load from LMCache is exactly num_computed_tokens —
                # NOT num_computed + ext (that would double-count). The worker
                # then computes only ``prompt_len - num_computed_tokens``.
                num_load = int(new_req.num_computed_tokens)
                if num_load > 0:
                    meta.loads.append(_ReqLoad(rid, token_ids, block_ids, num_load))
            else:
                # New prompt with no external hit: mark for store after forward.
                meta.stores.append(_ReqStore(rid, token_ids, block_ids))
        self._requests_need_load.clear()
        return meta

    def request_finished(self, request: Request, block_ids) -> tuple[bool, Any]:
        return False, None

    # ============================ worker role ============================
    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        meta = self._get_connector_metadata()
        if not isinstance(meta, MetalLMCacheConnectorMetadata):
            return
        if self._mlx is None or self._paged_cache is None:
            if meta.loads:
                logger.warning("LMCache load requested but MLX cache not bound")
            return
        for ld in meta.loads:
            try:
                hits = self._mlx.retrieve(
                    self._paged_cache,
                    ld.token_ids[: ld.num_load],
                    ld.block_ids[: -(-ld.num_load // self._block_size)],
                )
                logger.info(
                    "LMCache loaded %d/%d prefix tokens (req %s)",
                    hits,
                    ld.num_load,
                    ld.req_id,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("LMCache load failed (req %s): %s", ld.req_id, exc)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self, layer_name: str, kv_layer, attn_metadata, **kwargs: Any
    ) -> None:
        # MLX runner has no per-layer callback; whole-model store in wait_for_save.
        return

    def wait_for_save(self) -> None:
        meta = self._get_connector_metadata()
        if not isinstance(meta, MetalLMCacheConnectorMetadata):
            return
        if self._mlx is None or self._paged_cache is None:
            return
        for st in meta.stores:
            try:
                self._mlx.store(self._paged_cache, st.token_ids, st.block_ids)
            except Exception as exc:  # pragma: no cover
                logger.warning("LMCache store failed (req %s): %s", st.req_id, exc)

    def get_finished(self, finished_req_ids: set) -> tuple[set, set]:
        return set(), set()

    # ---- helpers ----
    def _scheduler_mlx(self):
        """Scheduler role needs LMCache lookup too. In uniproc mode it reuses
        the worker's bound MetalLMCacheConnector (same LMCache engine), so
        lookups observe the worker's stores. Returns None until the worker has
        run at least once and bound the cache (first prompt is always a miss,
        which is correct — nothing is cached yet)."""
        if self._mlx is not None:
            return self._mlx
        if _WORKER_MLX_BRIDGE is not None:
            self._mlx = _WORKER_MLX_BRIDGE
            self._paged_cache = _WORKER_PAGED_CACHE
            return self._mlx
        return None
