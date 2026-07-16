# SPDX-License-Identifier: Apache-2.0
"""OffloadingWorker moving KV blocks between the MLX cache and host memory.

The wired MLX KV pool is the fast tier; the host pool is plain (pageable)
memory that macOS may reclaim under pressure. On unified memory a "transfer"
is an on-package copy, so transfers execute synchronously inside
``submit_store``/``submit_load`` and are reported complete on the next
``get_finished`` poll — the OffloadingWorker contract stays async-shaped for
a later move to ``mx.async_eval`` if profiling justifies it.

Errors propagate out of ``submit_store``/``submit_load``: the upstream
OffloadingConnectorWorker asserts a successful submission (job failure is
unsupported upstream), so raising here surfaces the root cause directly
instead of tripping a bare ``assert success`` with no context.

Ordering rules (see docs/offload-design.md and kv_cache.py): the cache write
kernels mutate Metal buffers in place but rebind ``kv_cache.<name>[layer]`` to
fresh array objects carrying graph provenance. Stores therefore always gather
through the *current* list entry, and loads index-assign then rebind, exactly
like the kernels do.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion

logger = init_logger(__name__)

# Every MLX array list on MetalPagedKVCache that holds per-block state and
# must therefore be offloaded together. The TurboQuant side arrays are empty
# lists on the fp16/bf16 path.
CACHE_ATTRS = (
    "key_caches",
    "value_caches",
    "key_scale_caches",
    "value_scale_caches",
    "key_zero_caches",
)

# MLX dtype -> numpy dtype for the host pool. bfloat16 has no numpy dtype and
# is round-tripped through a uint16 reinterpret view (mx.view).
_MX_TO_NP = {
    mx.float16: np.float16,
    mx.float32: np.float32,
    mx.bfloat16: np.uint16,
    mx.int8: np.int8,
    mx.uint8: np.uint8,
    mx.uint16: np.uint16,
    mx.uint32: np.uint32,
    mx.int16: np.int16,
    mx.int32: np.int32,
}


@dataclass
class _CacheArrayRef:
    """One MLX cache array (layer x kind) and its host-pool mirror."""

    arrays: list[mx.array]  # the live list on MetalPagedKVCache
    layer: int
    is_bf16: bool
    # (num_cpu_blocks, block_size_factor, block_size, heads, last_dim)
    cpu: np.ndarray

    @property
    def gpu(self) -> mx.array:
        # Read through the list entry: the write kernels rebind it to fresh
        # array objects whose provenance orders reads after pending writes.
        return self.arrays[self.layer]

    @gpu.setter
    def gpu(self, arr: mx.array) -> None:
        self.arrays[self.layer] = arr


class MetalKVOffloadWorker(OffloadingWorker):
    """Bidirectional GPU<->CPU block mover for the MLX paged KV cache.

    One worker serves both directions: ``submit_store`` (GPU -> host pool)
    and ``submit_load`` (host pool -> GPU) share the ``_transfer`` core.
    """

    def __init__(
        self,
        kv_cache: MetalPagedKVCache,
        *,
        block_size_factor: int,
        num_cpu_blocks: int,
        region: MetalSharedOffloadRegion | None = None,
        expected_block_bytes: int | None = None,
    ) -> None:
        self.kv_cache = kv_cache
        self.block_size_factor = block_size_factor
        self.num_cpu_blocks = num_cpu_blocks
        self.region = region
        self._finished: list[TransferResult] = []
        self._sub_slots = np.arange(block_size_factor)
        self._num_gpu_blocks = int(kv_cache.num_blocks)

        # Completeness guard: CACHE_ATTRS is the offload inventory. A
        # per-block array list on the cache that is NOT in the inventory
        # would be silently omitted from offload, and restored blocks would
        # be rebuilt missing part of their state — the exact "new model
        # family" silent-corruption mode. Fail fast at registration instead.
        for name, value in vars(kv_cache).items():
            if name in CACHE_ATTRS or not isinstance(value, list) or not value:
                continue
            if all(isinstance(a, mx.array) and a.ndim > 0 for a in value) and any(
                a.shape[0] == self._num_gpu_blocks for a in value
            ):
                raise NotImplementedError(
                    f"KV offloading: cache array list {name!r} on "
                    f"{type(kv_cache).__name__} looks per-block "
                    f"(leading dim == num_blocks) but is not in the offload "
                    "inventory (CACHE_ATTRS); offloading this model would "
                    "silently drop that state. Add it to CACHE_ATTRS with a "
                    "round-trip test."
                )

        if region is not None:
            # Tiering: the host pool lives in the shared region so the
            # scheduler-side secondary tiers (disk, ...) read/write the same
            # bytes through region.create_kv_memoryview(). Each cache array
            # claims its slice of every block row via the region's own
            # cursor (create_next_view), keeping the row layout arithmetic
            # in one place.
            assert region.num_blocks == num_cpu_blocks

        self._refs: list[_CacheArrayRef] = []
        host_bytes = 0
        for attr in CACHE_ATTRS:
            arrays: list[mx.array] = getattr(kv_cache, attr)
            for layer, arr in enumerate(arrays):
                np_dtype = _MX_TO_NP.get(arr.dtype)
                if np_dtype is None:
                    raise NotImplementedError(
                        f"KV offloading: unsupported cache dtype {arr.dtype} "
                        f"({attr}[{layer}])"
                    )
                shape = (num_cpu_blocks, block_size_factor, *arr.shape[1:])
                if region is None:
                    cpu = np.empty(shape, dtype=np_dtype)
                else:
                    itemsize = np.dtype(np_dtype).itemsize
                    tensor_page = (
                        int(np.prod(arr.shape[1:])) * itemsize * block_size_factor
                    )
                    view = region.create_next_view(tensor_page)
                    cpu = view.numpy().view(np_dtype).reshape(shape)
                    # The disk tier reads these bytes through the region's
                    # memoryview; a silent numpy copy here would corrupt it.
                    assert cpu.ctypes.data == view.data_ptr(), (
                        "host-pool view is not zero-copy over the shared "
                        f"region ({attr}[{layer}])"
                    )
                self._refs.append(
                    _CacheArrayRef(
                        arrays=arrays,
                        layer=layer,
                        is_bf16=arr.dtype == mx.bfloat16,
                        cpu=cpu,
                    )
                )
                host_bytes += cpu.nbytes
        # Two-sided carve check: the scheduler sizes the pool from vLLM's
        # torch-side kv_cache_config (spec.kv_bytes_per_offloaded_block);
        # the handler carves from the live MLX arrays. The spec value may
        # exceed the carve by alignment padding (upstream tiering rounds
        # blocks up to BLOCK_SIZE_ALIGNMENT — the "maybe-pad" tail; fp16
        # blocks are often exactly aligned, TurboQuant's rarely are). Padding
        # is safe: rows are addressed by the padded stride on both sides.
        # A carve LARGER than the spec sizing would overrun rows — fatal.
        if expected_block_bytes is not None and num_cpu_blocks > 0:
            actual = host_bytes // num_cpu_blocks
            if actual > expected_block_bytes:
                raise ValueError(
                    f"KV offloading: handler carves {actual} bytes per "
                    f"offloaded block from the MLX cache but the spec sized "
                    f"the pool at {expected_block_bytes}; rows would overrun"
                )
            if actual < expected_block_bytes:
                logger.debug(
                    "KV offloading: %d bytes/block alignment padding "
                    "(spec %d vs carve %d)",
                    expected_block_bytes - actual,
                    expected_block_bytes,
                    actual,
                )
        logger.info(
            "KV offloading host pool: %.1f MB (%d CPU blocks x %d GPU "
            "blocks/CPU block, %d cache arrays, %s)",
            host_bytes / 1e6,
            num_cpu_blocks,
            block_size_factor,
            len(self._refs),
            "shared region" if region is not None else "plain numpy",
        )

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        if not isinstance(src_spec, GPULoadStoreSpec) or not isinstance(
            dst_spec, CPULoadStoreSpec
        ):
            raise ValueError(
                f"KV offloading: unexpected store spec types "
                f"{type(src_spec).__name__} -> {type(dst_spec).__name__}"
            )
        start = time.perf_counter()
        num_bytes = self._transfer(src_spec, dst_spec, gpu_to_cpu=True)
        self._record_finished(job_id, num_bytes, start)
        return True

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        if not isinstance(src_spec, CPULoadStoreSpec) or not isinstance(
            dst_spec, GPULoadStoreSpec
        ):
            raise ValueError(
                f"KV offloading: unexpected load spec types "
                f"{type(src_spec).__name__} -> {type(dst_spec).__name__}"
            )
        start = time.perf_counter()
        num_bytes = self._transfer(dst_spec, src_spec, gpu_to_cpu=False)
        self._record_finished(job_id, num_bytes, start)
        return True

    def _record_finished(self, job_id: int, num_bytes: int, start: float) -> None:
        self._finished.append(
            TransferResult(
                job_id=job_id,
                success=True,
                transfer_size=num_bytes,
                transfer_time=time.perf_counter() - start,
            )
        )

    def _transfer(
        self,
        gpu_spec: GPULoadStoreSpec,
        cpu_spec: CPULoadStoreSpec,
        *,
        gpu_to_cpu: bool,
    ) -> int:
        """Copy blocks between the MLX cache and the host pool.

        One offloaded (CPU) block holds ``block_size_factor`` GPU-block-sized
        sub-blocks. The first CPU block of a transfer may be entered
        mid-block: ``block_indices[0] % block_size_factor`` sub-slots are
        skipped, mirroring the CUDA handler
        (vllm/v1/kv_offload/cpu/gpu_worker.py).
        """
        # MetalOffloadingSpecBase enforces a single KV cache group.
        # Real raises, not asserts: these guard silent corruption (OOB
        # mx.take returns garbage rows; OOB MLX scatter drops writes;
        # negative numpy ids wrap) and must survive `python -O`. Raising
        # fails the submission loudly (job failure is unsupported upstream).
        if len(gpu_spec.group_sizes) != 1:
            raise ValueError(
                f"KV offloading: expected a single KV cache group, got "
                f"group_sizes={gpu_spec.group_sizes!r}"
            )
        group_size = gpu_spec.group_sizes[0]
        if group_size == 0:
            return 0

        gpu_ids = gpu_spec.block_ids
        if len(gpu_ids) != group_size:
            raise ValueError(
                f"KV offloading: {len(gpu_ids)} GPU block ids != group size "
                f"{group_size}"
            )
        if int(gpu_ids.min()) < 0 or int(gpu_ids.max()) >= self._num_gpu_blocks:
            raise ValueError(
                f"KV offloading: GPU block ids out of range "
                f"[{int(gpu_ids.min())}, {int(gpu_ids.max())}] vs "
                f"{self._num_gpu_blocks} cache blocks"
            )
        factor = self.block_size_factor
        skip = gpu_spec.block_indices[0] % factor

        cpu_ids = cpu_spec.block_ids
        if int(cpu_ids.min()) < 0 or int(cpu_ids.max()) >= self.num_cpu_blocks:
            raise ValueError(
                f"KV offloading: CPU block ids out of range "
                f"[{int(cpu_ids.min())}, {int(cpu_ids.max())}] vs pool of "
                f"{self.num_cpu_blocks} blocks"
            )
        if len(cpu_ids) * factor < skip + group_size:
            raise ValueError(
                f"CPU blocks too few: {len(cpu_ids)} x {factor} < {skip} + {group_size}"
            )
        # Expand CPU block ids into flat (block, sub-slot) coordinates,
        # skipping the unaligned head of the first block.
        cpu_blocks = np.repeat(cpu_ids, factor)[skip : skip + group_size]
        cpu_subs = np.tile(self._sub_slots, len(cpu_ids))[skip : skip + group_size]

        mx_gpu_ids = mx.array(gpu_ids)
        num_bytes = 0
        _dbg = os.environ.get("VLLM_METAL_KV_OFFLOAD_DEBUG") == "1"
        if gpu_to_cpu:
            # Gather all arrays lazily, evaluate once, then copy out. The
            # evaluated gathers expose the buffer protocol, so the host copy
            # reads them zero-copy through frombuffer.
            gathered = []
            for ref in self._refs:
                g = mx.take(ref.gpu, mx_gpu_ids, axis=0)
                if ref.is_bf16:
                    g = mx.view(g, mx.uint16)
                gathered.append(g)
            mx.eval(*gathered)
            for ref, g in zip(self._refs, gathered, strict=True):
                host = np.frombuffer(memoryview(g), dtype=ref.cpu.dtype).reshape(
                    g.shape
                )
                ref.cpu[cpu_blocks, cpu_subs] = host
                num_bytes += host.nbytes
        else:
            written = []
            for ref in self._refs:
                # Advanced indexing yields a fresh contiguous copy, so the
                # host pool row can be safely reused right after this call.
                host = ref.cpu[cpu_blocks, cpu_subs]
                vals = mx.array(host)
                if ref.is_bf16:
                    vals = mx.view(vals, mx.bfloat16)
                arr = ref.gpu
                arr[mx_gpu_ids] = vals
                # Rebind, mirroring the write-then-rebind idiom of the cache
                # kernels so later readers see the scatter's provenance.
                ref.gpu = arr
                written.append(arr)
                num_bytes += host.nbytes
            mx.eval(*written)

        if _dbg:
            # Debug (env-gated): CRC the moved bytes from BOTH sides of the
            # copy so store-vs-load divergence and host-vs-GPU divergence can
            # be separated when hunting live corruption.
            import zlib

            # Per-CPU-row CRCs so a row can be compared between the store
            # that wrote it and the (possibly differently-shaped) load that
            # read it. crc_gpu covers the same rows from the GPU side.
            # Blind spot: on the store path both CRCs derive from gathers
            # through the same (rebound) array, so their agreement cannot by
            # itself exclude a stale first gather — cross-check against a
            # no-offload control when hunting divergence (2026-07-15 audit).
            n = len(cpu_blocks)
            row_host = [0] * n
            row_gpu = [0] * n
            for ref in self._refs:
                g = mx.take(ref.gpu, mx_gpu_ids, axis=0)
                if ref.is_bf16:
                    g = mx.view(g, mx.uint16)
                mx.eval(g)
                g_np = np.array(g)
                for i in range(n):
                    row = np.ascontiguousarray(ref.cpu[cpu_blocks[i], cpu_subs[i]])
                    row_host[i] = zlib.crc32(row.tobytes(), row_host[i])
                    row_gpu[i] = zlib.crc32(g_np[i].tobytes(), row_gpu[i])
            for i in range(n):
                logger.info(
                    "KVDBG dir=%s row=(%d,%d) gpu=%d crc_host=%08x crc_gpu=%08x",
                    "store" if gpu_to_cpu else "load",
                    cpu_blocks[i],
                    cpu_subs[i],
                    int(gpu_ids[i]),
                    row_host[i],
                    row_gpu[i],
                )
        return num_bytes

    def get_finished(self) -> list[TransferResult]:
        finished = self._finished
        self._finished = []
        return finished

    def wait(self, job_ids: set[int]) -> None:
        # Transfers complete synchronously in submit_store/submit_load;
        # nothing is ever in flight by the time wait() can be called.
        return

    def shutdown(self) -> None:
        self._refs.clear()
        if self.region is not None:
            self.region.cleanup()
            self.region = None
