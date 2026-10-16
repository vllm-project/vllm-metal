# SPDX-License-Identifier: Apache-2.0
"""SharedOffloadRegion over anonymous RAM, shared within one process.

macOS has no /dev/shm, and a file-backed mmap would page KV out to SSD.
Offloading requires the uni executor, so the scheduler and the worker share a
process and find the same buffer through a registry keyed by engine id.
"""

from __future__ import annotations

import mmap
import threading
from dataclasses import dataclass

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

logger = init_logger(__name__)


@dataclass
class _RegionState:
    mapping: mmap.mmap
    buffer: np.ndarray  # 1-D uint8 view of mapping
    num_chunks: int
    row_stride: int
    refs: int


_registry: dict[str, _RegionState] = {}
_registry_lock = threading.Lock()


class MetalSharedOffloadRegion(SharedOffloadRegion):
    """SharedOffloadRegion over anonymous RAM, shared within one process."""

    def __init__(
        self,
        engine_id: str,
        num_chunks: int,
        rank: int | None,
        kv_bytes_per_chunk: int,
        cpu_page_size: int,
    ) -> None:
        # Never call super().__init__: it opens /dev/shm and pre-faults every
        # page, which would commit the whole pool at startup.
        self.page_size = mmap.PAGESIZE
        if num_chunks <= 0:
            raise ValueError(
                "KV offloading host pool holds no chunks; increase --kv-offloading-size"
            )
        if kv_bytes_per_chunk % self.page_size:
            raise ValueError(
                f"chunk size {kv_bytes_per_chunk} is not page aligned "
                f"({self.page_size})"
            )
        self.num_chunks = num_chunks
        self._row_stride = kv_bytes_per_chunk
        self.total_size_bytes = num_chunks * kv_bytes_per_chunk

        self.rank = rank
        if rank is not None:
            self._worker_offset = rank * cpu_page_size
            self._worker_area_end = (rank + 1) * cpu_page_size

        self._engine_id = engine_id
        with _registry_lock:
            state = _registry.get(engine_id)
            if state is None:
                # Anonymous memory pages in lazily, so the pool costs no RSS
                # until chunks are stored.
                mapping = mmap.mmap(
                    -1, self.total_size_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANON
                )
                state = _RegionState(
                    mapping=mapping,
                    buffer=np.frombuffer(mapping, dtype=np.uint8),
                    num_chunks=num_chunks,
                    row_stride=kv_bytes_per_chunk,
                    refs=1,
                )
                _registry[engine_id] = state
                logger.info(
                    "Created in-process offload region %s (%.2f GB)",
                    engine_id,
                    self.total_size_bytes / 1e9,
                )
            elif (state.num_chunks, state.row_stride) != (
                num_chunks,
                kv_bytes_per_chunk,
            ):
                raise ValueError(
                    f"offload region {engine_id} attached with mismatched "
                    f"geometry: {num_chunks}x{kv_bytes_per_chunk} vs existing "
                    f"{state.num_chunks}x{state.row_stride}"
                )
            else:
                state.refs += 1
        self._state: _RegionState | None = state

        self._base = torch.frombuffer(memoryview(state.buffer), dtype=torch.int8)
        self._views: list[torch.Tensor] = []
        # Attributes the inherited methods read. No file, no pinning, and
        # Metal never enables canonical_layout.
        self.is_pinned = False
        self._canonical_offset = 0
        self.pinned_addresses: list[int] = []
        self.mmap_obj = None
        self.fd = None
        self.mmap_path = None
        self._creator = False

    def release_pages(self) -> None:
        """Return the pool's pages to the OS. Their contents are lost.

        Only safe once nothing will read the pool. vLLM resets the connector,
        which empties the pool's index, before it frees the KV cache. The
        views stay valid. Call ``reuse_pages`` before writing again, or the
        kernel can drop the writes. Best effort: a failure keeps the pages,
        so sleep still completes.
        """
        if self._state is None:
            return
        try:
            self._state.mapping.madvise(mmap.MADV_FREE_REUSABLE)
        except OSError:
            logger.warning("Could not free the KV offloading host pool", exc_info=True)
            return
        logger.info(
            "Freed the KV offloading host pool (up to %.2f GiB)",
            self.total_size_bytes / (1 << 30),
        )

    def reuse_pages(self) -> None:
        """Take back the pages freed by ``release_pages``.

        Raises on failure, unlike ``release_pages``: the kernel can drop
        writes to pages not taken back, so wake must not go on.
        """
        if self._state is not None:
            self._state.mapping.madvise(mmap.MADV_FREE_REUSE)

    def cleanup(self) -> None:
        super().cleanup()
        state = self._state
        self._state = None
        if state is None:
            return
        with _registry_lock:
            state.refs -= 1
            if state.refs <= 0 and _registry.get(self._engine_id) is state:
                del _registry[self._engine_id]
                logger.info("Released in-process offload region %s", self._engine_id)
