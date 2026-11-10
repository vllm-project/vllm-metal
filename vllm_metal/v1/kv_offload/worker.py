# SPDX-License-Identifier: Apache-2.0
"""OffloadingWorker moving KV chunks between ``KVCacheStorage`` and the host pool.

Transfers run synchronously in ``submit_store``/``submit_load``. They move
whole pages of ``storage.pages``, the regions of vLLM's KV allocation. Loads
write with the native scatter, as ``KVCacheStorage.copy_blocks`` does: an MLX
index-assign on a page view never reaches the buffer.
"""

from __future__ import annotations

import time

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

from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion

logger = init_logger(__name__)

# Caps the transient one gather or scatter holds.
_SLICE_BYTES = 256 << 20


class MetalKVOffloadWorker(OffloadingWorker):
    """Moves KV chunks between the MLX-backed KV storage and the host pool."""

    def __init__(
        self,
        storage: KVCacheStorage,
        *,
        blocks_per_chunk: int,
        num_cpu_chunks: int,
        region: MetalSharedOffloadRegion,
        kv_bytes_per_chunk: int,
    ) -> None:
        if region.num_chunks != num_cpu_chunks:
            raise ValueError(
                f"region has {region.num_chunks} chunks, expected {num_cpu_chunks}"
            )
        self.storage = storage
        self.blocks_per_chunk = blocks_per_chunk
        self.num_cpu_chunks = num_cpu_chunks
        self.region: MetalSharedOffloadRegion | None = region
        self._finished: list[TransferResult] = []
        self._sub_slots = np.arange(blocks_per_chunk)
        self._num_gpu_blocks = int(storage.config.num_blocks)

        # The pool lives in the shared region so the disk tier reads the same
        # bytes. Each page takes its slice of every row, in page order.
        self._host: list[np.ndarray] = []
        for page in storage.pages:
            page_bytes = int(page.shape[1])
            view = region.create_next_worker_view(page_bytes * blocks_per_chunk)
            host = (
                view.numpy()
                .view(np.uint8)
                .reshape(num_cpu_chunks, blocks_per_chunk, page_bytes)
            )
            # A silent numpy copy would hide stores from the disk tier.
            if host.ctypes.data != view.data_ptr():
                raise RuntimeError("host pool view is not zero-copy")
            self._host.append(host)
        block_bytes = sum(host.shape[2] for host in self._host)
        self._blocks_per_slice = max(1, _SLICE_BYTES // max(block_bytes, 1))

        # The spec rounds each chunk up to BLOCK_SIZE_ALIGNMENT. Anything
        # beyond that padding means the two sides disagree on the layout.
        carved = block_bytes * blocks_per_chunk
        padding = kv_bytes_per_chunk - carved
        if not 0 <= padding < MetalSharedOffloadRegion.BLOCK_SIZE_ALIGNMENT:
            raise ValueError(
                f"host pool carves {carved} bytes per chunk but the spec sized "
                f"{kv_bytes_per_chunk}"
            )
        logger.info(
            "KV offloading host pool: %.1f MB (%d chunks x %d blocks, %d pages)",
            carved * num_cpu_chunks / 1e6,
            num_cpu_chunks,
            blocks_per_chunk,
            len(self._host),
        )

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        if not isinstance(src_spec, GPULoadStoreSpec) or not isinstance(
            dst_spec, CPULoadStoreSpec
        ):
            raise ValueError(
                f"unexpected store spec types {type(src_spec).__name__} -> "
                f"{type(dst_spec).__name__}"
            )
        self._submit(job_id, src_spec, dst_spec, store=True)
        return True

    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
        expedited: bool = False,
    ) -> bool:
        # Transfers finish inside submission, so an expedited load never queues.
        if not isinstance(src_spec, CPULoadStoreSpec) or not isinstance(
            dst_spec, GPULoadStoreSpec
        ):
            raise ValueError(
                f"unexpected load spec types {type(src_spec).__name__} -> "
                f"{type(dst_spec).__name__}"
            )
        self._submit(job_id, dst_spec, src_spec, store=False)
        return True

    def _submit(
        self,
        job_id: int,
        gpu_spec: GPULoadStoreSpec,
        cpu_spec: CPULoadStoreSpec,
        *,
        store: bool,
    ) -> None:
        start = time.perf_counter()
        num_bytes = 0
        for gpu_ids, chunks, subs in self._slices(gpu_spec, cpu_spec, store=store):
            if store:
                num_bytes += self._store(gpu_ids, chunks, subs)
            else:
                num_bytes += self._load(gpu_ids, chunks, subs)
        self._finished.append(
            TransferResult(
                job_id=job_id,
                success=True,
                transfer_size=num_bytes,
                transfer_time=time.perf_counter() - start,
            )
        )

    def _slices(self, gpu_spec, cpu_spec, *, store: bool):
        """Validate a transfer and yield it in slices of (gpu, chunk, sub-slot).

        The first chunk may be entered mid-chunk: ``block_indices[0] %
        blocks_per_chunk`` sub-slots are skipped, as in upstream's CUDA worker.
        """
        # Raise, not assert: bad ids corrupt silently and must fail under -O.
        if self.region is None:
            raise RuntimeError("KV offloading worker is shut down")
        if len(gpu_spec.group_sizes) != 1:
            raise ValueError(
                f"expected a single KV cache group, got {gpu_spec.group_sizes!r}"
            )
        # Upstream asserts this in the spec's __init__; asserts vanish under -O.
        if len(gpu_spec.block_indices) != len(gpu_spec.group_sizes):
            raise ValueError(
                f"block_indices must have one entry per KV cache group, got "
                f"{gpu_spec.block_indices!r} for {gpu_spec.group_sizes!r}"
            )
        if gpu_spec.block_indices[0] < 0:
            raise ValueError(
                f"block index must be non-negative, got {gpu_spec.block_indices[0]}"
            )
        group_size = gpu_spec.group_sizes[0]
        if group_size == 0:
            return
        gpu_ids = gpu_spec.block_ids
        if len(gpu_ids) != group_size:
            raise ValueError(f"{len(gpu_ids)} GPU block ids != group size {group_size}")
        if int(gpu_ids.min()) < 0 or int(gpu_ids.max()) >= self._num_gpu_blocks:
            raise ValueError(
                f"GPU block ids out of range [{int(gpu_ids.min())}, "
                f"{int(gpu_ids.max())}] for {self._num_gpu_blocks} blocks"
            )
        # The scatter writes destinations in parallel; repeats would race.
        if not store and len(np.unique(gpu_ids)) != len(gpu_ids):
            raise ValueError("duplicate GPU block ids in a load")
        factor = self.blocks_per_chunk
        skip = gpu_spec.block_indices[0] % factor
        cpu_ids = cpu_spec.block_ids
        if int(cpu_ids.min()) < 0 or int(cpu_ids.max()) >= self.num_cpu_chunks:
            raise ValueError(
                f"CPU chunk ids out of range [{int(cpu_ids.min())}, "
                f"{int(cpu_ids.max())}] for {self.num_cpu_chunks} chunks"
            )
        if len(cpu_ids) * factor < skip + group_size:
            raise ValueError(
                f"CPU chunks too few: {len(cpu_ids)} x {factor} < {skip} + {group_size}"
            )
        chunks = np.repeat(cpu_ids, factor)[skip : skip + group_size]
        subs = np.tile(self._sub_slots, len(cpu_ids))[skip : skip + group_size]
        step = self._blocks_per_slice
        for lo in range(0, group_size, step):
            yield gpu_ids[lo : lo + step], chunks[lo : lo + step], subs[lo : lo + step]

    def _store(self, gpu_ids, chunks, subs) -> int:
        src = mx.array(gpu_ids, dtype=mx.int32)
        rows = [page[src] for page in self.storage.pages]
        mx.eval(*rows)
        moved = 0
        for host, page_rows in zip(self._host, rows, strict=True):
            data = np.frombuffer(memoryview(page_rows), dtype=np.uint8)
            host[chunks, subs] = data.reshape(page_rows.shape)
            moved += data.nbytes
        return moved

    def _load(self, gpu_ids, chunks, subs) -> int:
        """Scatter host rows into the pages.

        The scatter reads the current page view, so it runs after any pending
        native write to these blocks, including stale zeroing.
        """
        moved = 0
        rows = []
        for host in self._host:
            # Fancy indexing copies, so the pool row is free once this returns.
            host_rows = host[chunks, subs]
            rows.append(mx.array(host_rows))
            moved += host_rows.nbytes
        self.storage.scatter_rows(rows, gpu_ids)
        mx.eval(*self.storage.pages)
        return moved

    def get_finished(self) -> list[TransferResult]:
        finished = self._finished
        self._finished = []
        return finished

    def wait(self, job_ids: set[int]) -> None:
        pass  # transfers complete in submit_*

    def release_host_pool(self) -> None:
        if self.region is not None:
            self.region.release_pages()

    def restore_host_pool(self) -> None:
        if self.region is not None:
            self.region.reuse_pages()

    def shutdown(self) -> None:
        self._host.clear()
        if self.region is not None:
            self.region.cleanup()
            self.region = None
