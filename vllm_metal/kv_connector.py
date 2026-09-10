# SPDX-License-Identifier: Apache-2.0
"""File-based KV transfer connector for prefill/decode disaggregation.

Implements the vLLM v1 ``KVConnectorBase`` contract with a shared directory
as the transfer medium (local filesystem, NFS, or sshfs mount), moving
block-granular K/V between a prefill instance and a decode instance on
Apple Silicon.

The design mirrors vLLM's ``ExampleConnector`` semantics (directory
existence = cache hit, block-aligned prompt hash as the key) but swaps
torch GPU tensors for ``mx.array`` slices of ``MetalPagedKVCache``'s
per-layer block pools, serialized with ``mx.save_safetensors``.

Layout on disk::

    <shared_storage_path>/<sha256(prompt tokens + mm hashes)>/
        layer_0000.safetensors   # {"key": [n, bs, h, d], "value": ...}
        layer_0001.safetensors
        ...
        done                     # marker written after all layers

Single KV-cache-group models only (dense GQA/MHA). Hybrid models with
linear-attention state or multiple block-size groups are rejected at
worker attach time.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

_DONE_MARKER = "done"


def _align_to_block_size(num_tokens: int, block_size: int) -> int:
    return (num_tokens - 1) // block_size * block_size


@dataclass
class MetalKVBlockRegistry:
    """Worker-side handle to the per-layer Metal paged KV block pools."""

    key_caches: list[mx.array]
    value_caches: list[mx.array]
    block_size: int
    num_blocks: int

    @property
    def num_layers(self) -> int:
        return len(self.key_caches)


@dataclass
class MetalReqMeta:
    """Per-request transfer plan carried from scheduler to worker."""

    token_ids: list[int]
    block_ids: list[int]
    is_store: bool
    mm_hashes: list[str]

    @property
    def num_blocks_to_transfer(self) -> int:
        return len(self.block_ids)


@dataclass
class MetalFileConnectorMetadata(KVConnectorMetadata):
    requests: list[MetalReqMeta] = field(default_factory=list)


class MetalFileConnector(KVConnectorBase_V1):
    """Synchronous file-backed KV connector for Metal disaggregation.

    One class serves both engine roles, like ``ExampleConnector``: the
    scheduler-role instance decides store/load per request (directory
    existence), the worker-role instance moves blocks between the paged
    pools and safetensors files. Both instances (and both Macs) must
    point ``shared_storage_path`` at the same directory.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        # Same prompt under a different cache_salt (or a LoRA adapter, not
        # yet folded in) must never hit another engine's store.
        self._cache_salt: str = (
            getattr(vllm_config.cache_config, "cache_salt", None) or ""
        )
        self._requests_need_load: dict[str, Request] = {}
        self._storage_path: str = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp/vllm-metal-kvshare"
        )
        # 0 disables cleanup; otherwise oldest stores (by done-marker mtime)
        # are pruned past this many entries.
        self._max_stored_prefixes: int = int(
            self._kv_transfer_config.get_from_extra_config("max_stored_prefixes", 0)
        )
        os.makedirs(self._storage_path, exist_ok=True)
        self._registry: MetalKVBlockRegistry | None = None
        # Only kv_producer engines write stores: a decode-side engine
        # computing a prompt locally must never publish its KV back into
        # the shared directory (one-way transfer plane).
        self._can_store = self._kv_transfer_config.kv_role in (
            "kv_producer",
            "kv_both",
        )
        logger.info(
            "MetalFileConnector role=%s storage=%s block_size=%d",
            role,
            self._storage_path,
            self._block_size,
        )

    # ------------------------------------------------------------------
    # Worker-side plumbing (called by MetalModelRunner, not by vLLM core)
    # ------------------------------------------------------------------

    def set_block_registry(self, registry: MetalKVBlockRegistry) -> None:
        """Attach the Metal paged KV pools owned by the model runner."""
        if len(registry.key_caches) != len(registry.value_caches):
            raise ValueError("key/value cache layer count mismatch")
        self._registry = registry

    def _require_registry(self) -> MetalKVBlockRegistry:
        if self._registry is None:
            raise RuntimeError(
                "MetalFileConnector worker role used before "
                "set_block_registry(); the model runner must attach the "
                "paged KV pools."
            )
        return self._registry

    # ------------------------------------------------------------------
    # KVConnectorBase worker hooks
    # ------------------------------------------------------------------

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        """Load externally-produced KV blocks into this engine's paged pools.

        The Metal path has no per-layer attention-op hookpoints, so loads
        (like saves) are bulk operations: all layers for all planned
        requests are materialized before the forward pass starts.
        ``forward_context`` is unused — the pools come from the registry.
        """
        del forward_context, kwargs
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MetalFileConnectorMetadata)
        registry = self._require_registry()

        for request in metadata.requests:
            if request.is_store:
                continue
            folder = self._folder_for(request.token_ids, request.mm_hashes)
            if not os.path.exists(os.path.join(folder, _DONE_MARKER)):
                raise FileNotFoundError(
                    f"KV for request vanished before load: {folder} has no "
                    f"{_DONE_MARKER} marker"
                )
            self._validate_manifest(folder, len(request.block_ids))
            for layer_idx in range(registry.num_layers):
                payload = mx.load(
                    os.path.join(folder, f"layer_{layer_idx:04d}.safetensors")
                )
                # The consumer allocates blocks for the whole prompt while
                # the payload covers only the block-aligned external prefix
                # — scatter into the leading blocks.
                num_blocks = payload["key"].shape[0]
                block_index = mx.array(request.block_ids[:num_blocks], dtype=mx.uint32)
                key_cache = registry.key_caches[layer_idx]
                value_cache = registry.value_caches[layer_idx]
                key_cache[block_index] = payload["key"]
                value_cache[block_index] = payload["value"]
            mx.eval(*registry.key_caches, *registry.value_caches)
            logger.info(
                "Loaded %d KV blocks x %d layers from %s",
                num_blocks,
                registry.num_layers,
                folder,
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Bulk loads complete inside ``start_load_kv``; nothing to wait for."""

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        """Per-layer saves are not used on the Metal path.

        ``save_finished_requests`` performs one bulk save after the async
        forward pass has materialized, which keeps the connector out of
        the model's attention wrappers entirely.
        """

    def wait_for_save(self) -> None:
        """Saves are synchronous inside ``save_finished_requests``."""

    def save_finished_requests(self) -> None:
        """Dump store-planned KV blocks to the shared directory.

        Called by the model runner after the forward pass is evaluated.

        Ordering guarantee (save-vs-free): the engine core is step
        synchronous on the Metal path — ``sample_tokens`` (which invokes
        this bulk save) returns before the scheduler runs its next
        ``schedule()`` and can free or reallocate any of this step's
        blocks. The gather below therefore always reads blocks that this
        step still owns; no delay_free_blocks handshake is required.

        Publication is atomic: files land in a hidden staging directory
        renamed into place as a whole, so a consumer never observes a
        partially-written prefix and a crashed store never wedges the
        prefix key (a re-store simply stages and renames again).
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MetalFileConnectorMetadata)
        registry = self._require_registry()

        for request in metadata.requests:
            if not request.is_store:
                continue
            if not request.block_ids:
                # Prompt shorter than block_size aligns down to zero
                # blocks; nothing to transfer for this request.
                logger.info("Skipping KV store for short prompt (0 aligned blocks)")
                continue
            folder = self._folder_for(request.token_ids, request.mm_hashes)
            staging = f"{folder}.staging-{os.getpid()}"
            shutil.rmtree(staging, ignore_errors=True)
            os.makedirs(staging, exist_ok=True)
            block_index = mx.array(request.block_ids, dtype=mx.uint32)
            for layer_idx in range(registry.num_layers):
                key_blocks = registry.key_caches[layer_idx][block_index]
                value_blocks = registry.value_caches[layer_idx][block_index]
                mx.eval(key_blocks, value_blocks)
                mx.save_safetensors(
                    os.path.join(staging, f"layer_{layer_idx:04d}.safetensors"),
                    {"key": key_blocks, "value": value_blocks},
                )
            # Layout manifest: the consumer refuses to load a prefix whose
            # geometry does not match its own pools (fail-fast instead of
            # silently scattering mismatched blocks into the paged cache).
            self._write_manifest(staging, registry, len(request.block_ids))
            with open(os.path.join(staging, _DONE_MARKER), "wb") as marker:
                marker.write(b"ok")
            shutil.rmtree(folder, ignore_errors=True)
            os.rename(staging, folder)
            self._fsync_dir(self._storage_path)
            self._maybe_prune_stores()
            logger.info(
                "Stored %d KV blocks x %d layers to %s",
                len(request.block_ids),
                registry.num_layers,
                folder,
            )

    @staticmethod
    def _fsync_dir(path: str) -> None:
        """Best-effort directory fsync so the rename survives a crash."""
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass

    def _write_manifest(
        self, folder: str, registry: MetalKVBlockRegistry, num_blocks: int
    ) -> None:
        first = registry.key_caches[0]
        _, block_size, kv_heads, head_dim = first.shape
        manifest = {
            "version": 1,
            "num_layers": registry.num_layers,
            "num_blocks": num_blocks,
            "block_size": int(block_size),
            "kv_heads": int(kv_heads),
            "head_dim": int(head_dim),
            "dtype": str(first.dtype),
        }
        with open(os.path.join(folder, "manifest.json"), "w") as fh:
            json.dump(manifest, fh)

    def _validate_manifest(self, folder: str, num_blocks_expected: int) -> None:
        """Refuse loads whose stored geometry disagrees with this engine."""
        registry = self._require_registry()
        path = os.path.join(folder, "manifest.json")
        if not os.path.exists(path):
            raise ValueError(
                f"KV store {folder} has no manifest.json; refusing to load "
                "(store and consumer must run the same connector version)"
            )
        with open(path) as fh:
            manifest = json.load(fh)
        first = registry.key_caches[0]
        _, block_size, kv_heads, head_dim = first.shape
        mismatches = []
        if manifest["num_layers"] != registry.num_layers:
            mismatches.append(
                f"num_layers {manifest['num_layers']} != {registry.num_layers}"
            )
        if manifest["block_size"] != int(block_size):
            mismatches.append(
                f"block_size {manifest['block_size']} != {int(block_size)}"
            )
        if manifest["kv_heads"] != int(kv_heads):
            mismatches.append(f"kv_heads {manifest['kv_heads']} != {int(kv_heads)}")
        if manifest["head_dim"] != int(head_dim):
            mismatches.append(f"head_dim {manifest['head_dim']} != {int(head_dim)}")
        if manifest["num_blocks"] > num_blocks_expected:
            mismatches.append(
                f"store holds {manifest['num_blocks']} blocks but only "
                f"{num_blocks_expected} were allocated for this request"
            )
        if mismatches:
            raise ValueError(
                "KV store geometry mismatch for "
                f"{folder}: {'; '.join(mismatches)}. The producer and "
                "consumer must serve the same model, revision, and "
                "cache configuration."
            )

    def _maybe_prune_stores(self) -> None:
        if self._max_stored_prefixes <= 0:
            return
        entries = []
        for name in os.listdir(self._storage_path):
            if ".staging-" in name:
                continue
            done = os.path.join(self._storage_path, name, _DONE_MARKER)
            if os.path.exists(done):
                entries.append((os.path.getmtime(done), name))
        entries.sort()
        excess = len(entries) - self._max_stored_prefixes
        for _, name in entries[: max(excess, 0)]:
            shutil.rmtree(os.path.join(self._storage_path, name), ignore_errors=True)

    # ------------------------------------------------------------------
    # KVConnectorBase scheduler hooks
    # ------------------------------------------------------------------

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Report externally-transferable tokens (directory hit = hit)."""
        if not self._found_match_for_request(request):
            return 0, False
        logger.info("External KV cache hit for request %s", request.request_id)
        token_ids = request.prompt_token_ids or []
        num_tokens_to_check = _align_to_block_size(len(token_ids) - 1, self._block_size)
        # KVConnector contract (RFC #44223): never report fewer tokens than
        # the engine already computed — a negative count would corrupt the
        # scheduler's num_computed_tokens accounting.
        return max(0, num_tokens_to_check - num_computed_tokens), False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Freeze this step's store/load plan (resets pending state)."""
        meta = MetalFileConnectorMetadata()

        total_need_load = 0
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            mm_hashes = [f.identifier for f in new_req.mm_features]
            if new_req.req_id in self._requests_need_load:
                # Same aligned-prefix key as the store side (see below).
                valid = _align_to_block_size(len(token_ids) - 1, self._block_size)
                meta.requests.append(
                    MetalReqMeta(
                        token_ids=token_ids[:valid],
                        block_ids=list(new_req.block_ids[0]),
                        is_store=False,
                        mm_hashes=mm_hashes,
                    )
                )
                total_need_load += 1
            elif self._can_store and not self._found_match_for_prompt(
                token_ids, mm_hashes
            ):
                valid = _align_to_block_size(len(token_ids) - 1, self._block_size)
                num_blocks = valid // self._block_size
                if num_blocks == 0:
                    continue
                meta.requests.append(
                    MetalReqMeta(
                        # Key must be the block-aligned prefix so the
                        # consumer's lookup (also aligned) hashes the
                        # exact same bytes.
                        token_ids=token_ids[:valid],
                        block_ids=list(new_req.block_ids[0])[:num_blocks],
                        is_store=True,
                        mm_hashes=mm_hashes,
                    )
                )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            resumed_from_preemption = req_id in cached_reqs.resumed_req_ids
            if not resumed_from_preemption or req_id not in self._requests_need_load:
                continue

            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            new_block_ids = cached_reqs.new_block_ids[i]

            request = self._requests_need_load[req_id]
            total_tokens = num_computed_tokens + num_new_tokens
            token_ids = request.all_token_ids[:total_tokens]
            assert new_block_ids is not None
            meta.requests.append(
                MetalReqMeta(
                    token_ids=token_ids,
                    block_ids=list(new_block_ids[0]),
                    is_store=False,
                    mm_hashes=[f.identifier for f in request.mm_features],
                )
            )
            total_need_load += 1

        # A request cancelled between update_state_after_alloc and here
        # would silently drop out of scheduled_new_reqs; drain instead of
        # asserting so cancellation cannot crash EngineCore.
        if total_need_load != len(self._requests_need_load):
            logger.warning(
                "Dropping %d load-planned request(s) not scheduled this step "
                "(cancelled or preempted); they will re-plan on retry",
                len(self._requests_need_load) - total_need_load,
            )
        self._requests_need_load.clear()
        return meta

    def request_finished(self, request: Request, block_ids: list[list[int]]) -> None:
        """Drop any pending load plan for a finished/cancelled request."""
        self._requests_need_load.pop(request.request_id, None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _found_match_for_request(self, request: Request) -> bool:
        return self._found_match_for_prompt(
            list(request.prompt_token_ids or []),
            [f.identifier for f in request.mm_features],
        )

    def _found_match_for_prompt(
        self,
        prompt_token_ids: list[int],
        mm_hashes: list[str],
    ) -> bool:
        num_tokens_to_check = _align_to_block_size(
            len(prompt_token_ids) - 1, self._block_size
        )
        folder = self._folder_for(prompt_token_ids[:num_tokens_to_check], mm_hashes)
        return os.path.exists(os.path.join(folder, _DONE_MARKER))

    def _folder_for(
        self,
        token_ids: list[int],
        mm_hashes: list[str],
        create: bool = False,
    ) -> str:
        # Deterministic across machines: comma-joined decimal token ids,
        # so producer and consumer never need matching tensor dtypes.
        hasher = hashlib.sha256(
            b",".join(str(t).encode() for t in token_ids) if token_ids else b""
        )
        hasher.update(self._cache_salt.encode())
        for mm_hash in mm_hashes:
            hasher.update(mm_hash.encode("utf-8"))
        folder = os.path.join(self._storage_path, hasher.hexdigest())
        if create:
            os.makedirs(folder, exist_ok=True)
        return folder
