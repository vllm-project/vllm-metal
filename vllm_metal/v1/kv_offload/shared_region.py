# SPDX-License-Identifier: Apache-2.0
"""macOS replacement for vLLM's SharedOffloadRegion.

The upstream region (vllm/v1/kv_offload/cpu/shared_offload_region.py) backs
the CPU offload tier with a /dev/shm mmap so the scheduler-side tiering
manager and the worker-side handlers see the same bytes across processes.
macOS has no tmpfs: a file-backed mmap would sit on APFS, where the pager
writes dirty KV pages back to SSD — turning the "RAM tier" into a second
disk tier (and leaking multi-GB files on crash).

On Metal the scheduler and worker always share one process: KV offloading
enforces the "uni" executor at config time (MetalPlatform), so the region
can simply be anonymous RAM shared through a per-process registry keyed by
instance id. The scheduler-role spec and the worker-role spec each construct
a MetalSharedOffloadRegion with the same engine_id and get views over the
same numpy buffer. Layout, ``create_next_worker_view``, ``create_kv_memoryview``,
and cleanup bookkeeping are inherited from the upstream class unchanged.

If a cross-process executor is ever supported, this class must move to real
shared memory (e.g. POSIX shm) — a registry miss in another process would
silently create a disjoint buffer, which is why the executor guard exists.
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


def _check_fits_in_ram(total_bytes: int) -> None:
    """Refuse a host pool the machine cannot hold.

    Upstream's SharedOffloadRegion calls check_shm_free_space before
    ftruncate. There is no /dev/shm here, and np.zeros pages in lazily, so
    an oversized pool starts cleanly and only thrashes once blocks are
    actually written. On a fixed-RAM Mac that is a wedged machine rather
    than a failed request, so check up front.

    Compared against total RAM, not available: the pool is long-lived and
    sizing it off whatever happens to be free at startup is not stable.
    """
    import psutil

    total_ram = psutil.virtual_memory().total
    # The wired KV cache and the model weights also live in this budget.
    if total_bytes > total_ram // 2:
        raise ValueError(
            f"KV offloading host pool is {total_bytes / (1 << 30):.1f} GiB on "
            f"a {total_ram / (1 << 30):.0f} GiB machine. The pool is pageable "
            "and shares memory with the model weights and the wired KV cache, "
            "so this would swap rather than fail cleanly. Lower "
            "--kv-offloading-size."
        )


@dataclass
class _RegionState:
    buffer: np.ndarray  # 1-D uint8 of num_blocks * row_stride bytes
    num_blocks: int
    row_stride: int
    refs: int


_registry: dict[str, _RegionState] = {}
_registry_lock = threading.Lock()


class MetalSharedOffloadRegion(SharedOffloadRegion):
    """SharedOffloadRegion over anonymous RAM, shared within one process."""

    def __init__(
        self,
        engine_id: str,
        num_blocks: int,
        rank: int | None,
        kv_bytes_per_block: int,
        cpu_page_size: int,
    ) -> None:
        # Full override of the upstream constructor (it is not parameterized
        # over its /dev/shm + madvise mechanism); every attribute consumed by
        # the inherited methods is established here.
        self.page_size = mmap.PAGESIZE
        assert kv_bytes_per_block % self.page_size == 0

        self.num_blocks = num_blocks
        self._row_stride = kv_bytes_per_block
        self.total_size_bytes = self.num_blocks * self._row_stride

        self.rank = rank
        if rank is not None:
            self._worker_offset = rank * cpu_page_size
            self._worker_area_end = (rank + 1) * cpu_page_size

        self._engine_id = engine_id
        _check_fits_in_ram(self.total_size_bytes)
        with _registry_lock:
            state = _registry.get(engine_id)
            if state is None:
                # np.zeros pages fault in lazily on first touch (calloc), so
                # a large pool costs no RSS until blocks are actually stored.
                state = _RegionState(
                    buffer=np.zeros(self.total_size_bytes, dtype=np.uint8),
                    num_blocks=num_blocks,
                    row_stride=self._row_stride,
                    refs=1,
                )
                _registry[engine_id] = state
                logger.info(
                    "Created in-process offload region %s (%.2f GB)",
                    engine_id,
                    self.total_size_bytes / 1e9,
                )
            else:
                assert (
                    state.num_blocks == num_blocks
                    and state.row_stride == self._row_stride
                ), (
                    f"offload region {engine_id} attached with mismatched "
                    f"geometry: {num_blocks}x{self._row_stride} vs existing "
                    f"{state.num_blocks}x{state.row_stride}"
                )
                state.refs += 1
        self._state: _RegionState | None = state

        self._base = torch.frombuffer(memoryview(state.buffer), dtype=torch.int8)
        self._views: list[torch.Tensor] = []
        self.is_pinned = False
        # Inherited cleanup() checks these; there is no file behind this
        # region, so they are inert.
        self.mmap_obj = None
        self.fd = None
        self.mmap_path = None
        self._creator = False

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
