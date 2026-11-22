# SPDX-License-Identifier: Apache-2.0
"""Execute vLLM's physical cache layout on MLX; Torch is used only at init."""

from __future__ import annotations

import ctypes
import resource
from bisect import bisect_right
from collections.abc import Sequence
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import torch
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheLayout,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import allocate_kv_cache

from vllm_metal import envs
from vllm_metal.tensor_bridge import TORCH_TO_MLX_DTYPE

_INT32_MAX = 2**31 - 1


def gather_cache_rows(cache: mx.array, ids: mx.array, block_stride: int) -> mx.array:
    """Read rows from ``(blocks, rows_per_block, ...)``; stride is in elements."""
    # MLX selects offset width from logical size, which misses wide pool strides
    # (#1081). Slice before flattening so each gather's offsets fit in 32 bits.
    blocks_per_take = max(_INT32_MAX // block_stride, 1)
    row_shape = cache.shape[2:]
    if cache.shape[0] <= blocks_per_take:
        return cache.reshape(-1, *row_shape)[ids]

    rows = None
    for first in range(0, cache.shape[0], blocks_per_take):
        part = cache[first : first + blocks_per_take].reshape(-1, *row_shape)
        local = ids - first * cache.shape[1]
        inside = (local >= 0) & (local < part.shape[0])
        taken = mx.take(part, mx.where(inside, local, 0), axis=0)
        mask = inside.reshape(-1, *([1] * len(row_shape)))
        rows = taken if rows is None else mx.where(mask, taken, rows)
    assert rows is not None
    return rows


def allocate_kv_cache_lazily(
    config: KVCacheConfig,
    layout: KVCacheLayout,
) -> dict[str, torch.Tensor]:
    """Allocate the KV backing store without committing its pages.

    ``vllm.v1.worker.utils.allocate_kv_cache`` fills the whole backing buffer
    with zeros (``torch.zeros``). Metal runs on unified memory, so that memset
    writes every page of a multi-GB pool and the entire pool becomes resident
    from the first step — even though a serving run only ever touches the
    blocks it actually allocates (a 16 GB Mac serving a 0.6B model pays ~5 GB
    of RSS for a pool that is ~90% unused).

    The zero fill is only load-bearing where vLLM says it is:
    ``KVCacheConfig.needs_kv_cache_zeroing`` marks Mamba state (read before it
    is written) and mixed-precision caches (a block reinterpreted under another
    precision would decode stale bytes as NaN/Inf). Uniform-precision attention
    caches skip per-block zeroing upstream, and their kernels mask every slot
    past the sequence length, so unwritten memory never reaches an output. Keep
    vLLM's zero-filled allocation verbatim in that case.

    Otherwise vLLM's ``allocate`` callback replaces the filled backing buffer
    with an uninitialized one, so the allocation and its layer views stay
    entirely upstream; ``tests/attention/test_lazy_kv_allocation.py`` pins that
    layout.
    """
    if config.needs_kv_cache_zeroing:
        return allocate_kv_cache(config, torch.device("cpu"), layout)
    return allocate_kv_cache(
        config,
        torch.device("cpu"),
        layout,
        allocate=lambda size: torch.empty(size, dtype=torch.int8, device="cpu"),
    )


_PROT_READ_WRITE = 0x3
_MAP_PRIVATE_FIXED_ANON = 0x2 | 0x10 | 0x1000


def _replace_pages(ptr: int, nbytes: int) -> int:
    """Map fresh zero pages over the whole pages inside [ptr, ptr + nbytes).

    The range keeps its address, so tensors over it stay valid, and its old
    pages go back to the OS at once; madvise(MADV_FREE_REUSABLE) only marks
    them reclaimable. Partial pages at either end are left alone, since they
    may belong to a neighbour. Returns the bytes replaced.
    """
    page = resource.getpagesize()
    first = -(-ptr // page) * page
    last = (ptr + nbytes) // page * page
    if last <= first:
        return 0
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    got = libc.mmap(
        first, last - first, _PROT_READ_WRITE, _MAP_PRIVATE_FIXED_ANON, -1, 0
    )
    if got != first:
        raise OSError(ctypes.get_errno(), "could not replace KV chunk pages")
    return last - first


class CacheViews(Sequence[mx.array]):
    """Layer views following the latest native write to their shared backing."""

    def __init__(self, storage: KVCacheStorage, tensors: Sequence[torch.Tensor]):
        self.storage = storage
        self.descriptors = []
        for t in tensors:
            byte_offset = t.storage_offset() * t.element_size()
            region = bisect_right(storage.region_offsets, byte_offset) - 1
            self.descriptors.append(
                (
                    region,
                    tuple(t.shape),
                    tuple(t.stride()),
                    (byte_offset - storage.region_offsets[region]) // t.element_size(),
                    TORCH_TO_MLX_DTYPE[t.dtype],
                )
            )

    def __len__(self) -> int:
        return len(self.descriptors)

    def __getitem__(self, index: int) -> mx.array:
        from vllm_metal.metal import get_ops

        region, shape, strides, offset, dtype = self.descriptors[index]
        if getattr(self.storage, "chunk_shift", None) is not None:
            # Chunk mode: the region's address table, offset to this view's K or V.
            return self.storage.buffers[region] + mx.array(
                offset * dtype.size, dtype=mx.uint64
            )
        return get_ops().as_strided(
            self.storage.buffers[region].view(dtype), shape, strides, offset
        )

    def __setitem__(self, index: int, value: mx.array) -> None:
        # Native writes alias their input. Track the write on its region so
        # sibling views, including aliases with a different dtype, see it.
        region = self.descriptors[index][0]
        self.storage.buffers[region] = mx.depends(self.storage.buffers[region], [value])

    def gather_rows(self, index: int, ids: mx.array) -> mx.array:
        """Read selected rows from a layer's shared-pool view."""
        _, _, strides, _, _ = self.descriptors[index]
        return gather_cache_rows(self[index][:, None], ids, strides[0])


class KVCacheStorage:
    """Consume vLLM's resolved cache layout and scheduler-owned block IDs."""

    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.specs = {
            name: (
                group.kv_cache_spec.kv_cache_specs[name]
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                else group.kv_cache_spec
            )
            for group in config.kv_cache_groups
            for name in group.layer_names
        }
        # Read once, so a sleep and wake cycle rebuilds the same layout.
        self.chunk_mib = envs.VLLM_METAL_KV_CHUNK_MIB
        self.chunk_shift: int | None = None
        if self.chunk_mib:
            if config.needs_kv_cache_zeroing:
                raise NotImplementedError("kv chunk mode: zero-filled pools")
            if any(hasattr(spec, "k_quant") for spec in self.specs.values()):
                raise NotImplementedError("kv chunk mode: TurboQuant caches")
        self._bind()

    def _bind(self) -> None:
        config = self.config
        self.tensors = allocate_kv_cache_lazily(
            config,
            KVCacheLayout[config.kv_cache_layout],
        )
        first = next(iter(self.tensors.values()))
        self._backing = first.untyped_storage()
        self._raw = torch.empty(0, dtype=torch.uint8).set_(self._backing)
        self.nbytes = self._backing.nbytes()
        # Metal advertises LBNHC: each distinct layer address starts a compact
        # region. Import views of those regions from vLLM's one allocation, so
        # the full cache need not fit in a single Metal buffer.
        self.region_offsets = sorted(
            {
                tensor.offset + i * tensor.layer_stride
                for tensor in config.kv_cache_tensors
                for i in range(len(tensor.layers))
            }
        )
        self._wrap_buffers()
        # CoW/zeroing cover physical bytes exactly once. Layer-outer layouts
        # place a logical block in disjoint regions of this one allocation.
        regions = {}
        for tensor in config.kv_cache_tensors:
            for i, name in enumerate(tensor.layers):
                offset = tensor.offset + i * tensor.layer_stride
                size = self.specs[name].page_size_bytes
                regions[(offset, tensor.block_stride, size)] = torch.as_strided(
                    self._raw,
                    (config.num_blocks, size),
                    (tensor.block_stride, 1),
                    offset,
                )
        # Chunked layers are address tables; nothing can view their bytes.
        self.pages = (
            CacheViews(self, list(regions.values())) if not self.chunk_mib else []
        )

    def _wrap_buffers(self) -> None:
        from vllm_metal.metal import get_ops

        if self.chunk_mib:
            self._wrap_chunks()
            return

        # Not torch_to_mlx: MLX wires the buffers it adopts while a wired limit
        # is set, which made the whole pool resident at startup (#1198). The
        # pages stay owned by self._backing.
        ends = [*self.region_offsets[1:], self.nbytes]
        regions = [
            self._raw[start:end].view(self.config.num_blocks, -1)
            for start, end in zip(self.region_offsets, ends, strict=True)
        ]
        self.buffers = [
            get_ops().wrap_host_memory(region.data_ptr(), *region.shape)
            for region in regions
        ]

    def _wrap_chunks(self) -> None:
        """Wrap each region as chunks of whole blocks; buffers[r] is r's address table.

        A dispatch commits every buffer it reaches, so one buffer per layer makes
        the whole pool resident on the first forward pass. Chunks are separate
        buffers the kernels reach through their GPU addresses, made resident only
        once a write lands in them (``wire_slots``).
        """
        from vllm_metal.metal import get_ops

        ops = get_ops()
        ends = [*self.region_offsets[1:], self.nbytes]
        num_blocks = self.config.num_blocks
        block_bytes = (ends[0] - self.region_offsets[0]) // num_blocks
        blocks_per_chunk = max(1, (self.chunk_mib << 20) // block_bytes)
        blocks_per_chunk = 1 << (blocks_per_chunk.bit_length() - 1)
        self.chunk_shift = blocks_per_chunk.bit_length() - 1
        self._blocks_per_chunk = blocks_per_chunk
        self._block_bytes = block_bytes
        self._region_starts = list(self.region_offsets)
        self._chunk_blocks = [
            [
                min(blocks_per_chunk, num_blocks - first)
                for first in range(0, num_blocks, blocks_per_chunk)
            ]
            for _ in self.region_offsets
        ]
        self._chunks: list[list[mx.array | None]] = [
            [self._new_chunk(r, c) for c in range(len(blocks))]
            for r, blocks in enumerate(self._chunk_blocks)
        ]
        self._addrs = [
            np.array([ops._buffer_gpu_address(c) for c in chunks], dtype=np.uint64)
            for chunks in self._chunks
        ]
        self.buffers = [mx.array(addrs) for addrs in self._addrs]
        self._wired_chunks: set[int] = set()
        self._idle_steps: dict[int, int] = {}
        self._wired_slots: mx.array | None = None
        # Strides the kernels use, in elements of the cache dtype. Kernels see a
        # layer as [block, token, head, C]; the upstream view is
        # [block, head, token, C], so read them off the transposed view.
        first = next(iter(self.tensors.values())).transpose(1, 2)
        strides = first.stride()
        self.chunk_dtype = TORCH_TO_MLX_DTYPE[first.dtype]
        self.chunk_layout = [
            strides[0],
            strides[1],
            strides[2],
            0,
            first.shape[-1] // 2,
        ]

    def _chunk_span(self, region: int, chunk: int) -> tuple[int, int]:
        """Address and length of a chunk inside the backing."""
        start = self._raw.data_ptr() + self._region_starts[region]
        start += chunk * self._blocks_per_chunk * self._block_bytes
        return start, self._chunk_blocks[region][chunk] * self._block_bytes

    def _new_chunk(self, region: int, chunk: int) -> mx.array:
        from vllm_metal.metal import get_ops

        ptr, _ = self._chunk_span(region, chunk)
        rows = self._chunk_blocks[region][chunk]
        return get_ops().wrap_host_memory(ptr, rows, self._block_bytes)

    def require_plain_pool(self, feature: str) -> None:
        """Refuse a consumer that reads or writes the pool bytes directly.

        In chunk mode a layer's cache array is an address table, so treating
        it as the KV data itself would corrupt the table.
        """
        if self.chunk_shift is not None:
            raise NotImplementedError(f"kv chunk mode: {feature}")

    def chunk_kwargs(self) -> dict:
        """Arguments telling paged_attention_primitive its K/V arrays are chunk
        address tables. Empty for a plain pool."""
        if self.chunk_shift is None:
            return {}
        return {
            "kv_chunk_shift": self.chunk_shift,
            "kv_chunk_dtype": self.chunk_dtype,
            "kv_chunk_strides": self.chunk_layout[:3],
        }

    def write_chunked(
        self,
        key: mx.array,
        value: mx.array,
        slot_mapping: mx.array,
        key_caches: CacheViews,
        layer: int,
        block_size: int,
    ) -> tuple[mx.array, mx.array]:
        """Scatter K/V rows into the chunks of ``key_caches[layer]``.

        Wires the chunks the slots fall in first, then reads the layer's table,
        since wiring may re-wrap a reclaimed chunk at a new GPU address. Returns
        the K and V tables the attention read must depend on; V is K shifted to
        the value half of a row.
        """
        from vllm_metal.metal import get_ops

        if self.chunk_shift is None:
            raise RuntimeError("write_chunked needs a chunked pool")
        self.wire_slots(slot_mapping, block_size)
        key_table = get_ops()._reshape_and_cache_chunked(
            key,
            value,
            slot_mapping,
            key_caches[layer],
            self.chunk_shift,
            block_size,
            self.chunk_layout,
            self.chunk_dtype,
        )
        value_offset = self.chunk_layout[4] * self.chunk_dtype.size
        return key_table, key_table + mx.array(value_offset, dtype=mx.uint64)

    def wire_slots(self, slot_mapping: mx.array, block_size: int) -> None:
        """Make the chunks ``slot_mapping`` writes into GPU-resident.

        Called once per forward step; every layer shares the block ids, so the
        chunk at an index is wired in every region together.
        """
        if slot_mapping is self._wired_slots:
            return
        # Held, not its id(): a freed array's id can come back next step.
        self._wired_slots = slot_mapping
        slots = np.asarray(slot_mapping)
        chunk_ids = np.unique((slots[slots >= 0] // block_size) >> self.chunk_shift)
        self._wire_chunk_ids({int(c) for c in chunk_ids})

    def _wire_chunk_ids(self, chunk_ids: set[int]) -> None:
        from vllm_metal.metal import get_ops

        ops = get_ops()
        new = sorted(chunk_ids - self._wired_chunks)
        wrappers = []
        for r, chunks in enumerate(self._chunks):
            rewrapped = False
            for c in new:
                if chunks[c] is None:
                    # Reclaimed earlier: a fresh wrapper, and the table learns it.
                    chunks[c] = self._new_chunk(r, c)
                    self._addrs[r][c] = ops._buffer_gpu_address(chunks[c])
                    rewrapped = True
                wrappers.append(chunks[c])
            if rewrapped:
                self.buffers[r] = mx.depends(
                    mx.array(self._addrs[r]), [self.buffers[r]]
                )
        ops._wire_buffers(wrappers)
        self._wired_chunks.update(new)
        for c in new:
            self._idle_steps.pop(c, None)

    def _chunk_block_ops(self, ids: np.ndarray, *, copy: bool) -> None:
        """Copy (ids [n, 2]) or zero (ids [n]) whole blocks in every region."""
        from vllm_metal.metal import get_ops

        ops = get_ops()
        self._wire_chunk_ids({int(c) for c in np.unique(ids >> self.chunk_shift)})
        ids_mx = mx.array(ids)
        for r in range(len(self.buffers)):
            self.buffers[r] = ops._chunk_block_ops(
                self.buffers[r], ids_mx, copy, self.chunk_shift, self._block_bytes
            )

    def reclaim_idle_chunks(
        self, live_chunk_ids: set[int], min_idle_steps: int = 4
    ) -> int:
        """Give back wired chunks no request's blocks have used for a while.

        Their contents are discarded, so the caller must not use this while freed
        blocks may still be read again (prefix caching). Returns bytes given back.
        """
        idle = []
        for c in sorted(self._wired_chunks):
            if c in live_chunk_ids:
                self._idle_steps.pop(c, None)
                continue
            self._idle_steps[c] = self._idle_steps.get(c, 0) + 1
            if self._idle_steps[c] >= min_idle_steps:
                idle.append(c)
        if not idle:
            return 0
        from vllm_metal.metal import get_ops

        # The GPU must be done with a chunk before its pages are replaced.
        mx.synchronize()
        get_ops()._wire_buffers(
            [
                chunks[c]
                for chunks in self._chunks
                for c in idle
                if chunks[c] is not None
            ],
            False,
        )
        freed = 0
        for c in idle:
            for r, chunks in enumerate(self._chunks):
                if chunks[c] is not None:
                    chunks[c] = None
                    freed += _replace_pages(*self._chunk_span(r, c))
            self._wired_chunks.discard(c)
            self._idle_steps.pop(c, None)
        return freed

    @property
    def released(self) -> bool:
        return not self.buffers

    def release(self) -> None:
        """Free the backing memory in place. Its contents are discarded.

        The tensors keep their identity, so views registered elsewhere (the KV
        offload connector's) see the memory ``reallocate`` brings back. No MLX
        array built from ``buffers`` may outlive this call: the pages go back
        to the OS even while such an array still points at them.
        """
        self.buffers = []
        mx.synchronize()
        if self.chunk_shift is not None:
            from vllm_metal.metal import get_ops

            get_ops()._wire_buffers(
                [w for chunks in self._chunks for w in chunks if w is not None], False
            )
            self._chunks, self._wired_chunks = [], set()
            self._wired_slots = None
        self._backing.resize_(0)

    def reallocate(self) -> None:
        """Bring the backing memory back with the layout vLLM scheduled against."""
        self._backing.resize_(self.nbytes)
        if self.config.needs_kv_cache_zeroing:
            self._raw.zero_()
        self._wrap_buffers()

    def depend(self, arrays: Sequence[mx.array]) -> None:
        if arrays:
            self.buffers = mx.depends(self.buffers, list(arrays))

    def views(self, tensors: Sequence[torch.Tensor]) -> CacheViews:
        return CacheViews(self, tensors)

    def state_views(self, names: Sequence[str]) -> list[CacheViews]:
        states = []
        for name in names:
            spec = self.specs[name]
            binding = SimpleNamespace(
                get_state_shape=lambda spec=spec: spec.shapes,
                get_state_dtype=lambda spec=spec: spec.dtypes,
                state_alignment=spec.state_alignment,
            )
            MambaBase.bind_kv_cache(binding, self.tensors[name])
            states.append(binding.kv_cache)
        return [self.views(component) for component in zip(*states, strict=True)]

    def copy_blocks(self, pairs: Sequence[tuple[int, int]]) -> None:
        if not pairs:
            return
        if self.chunk_shift is not None:
            self._chunk_block_ops(
                np.array(pairs, dtype=np.int32).reshape(-1, 2), copy=True
            )
            return
        from vllm_metal.metal import get_ops

        src, dst = zip(*pairs, strict=True)
        src_ids, dst_ids = mx.array(src, dtype=mx.int32), mx.array(dst, dtype=mx.int32)
        # Snapshot all sources before any destination writes, including cycles.
        sources = [page[src_ids] for page in self.pages]
        self.depend(sources)
        for i, rows in enumerate(sources):
            self.pages[i] = get_ops().gdn_state_scatter(self.pages[i], rows, dst_ids)

    def zero_blocks(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        if self.chunk_shift is not None:
            self._chunk_block_ops(
                np.array(sorted(set(ids)), dtype=np.int32), copy=False
            )
            return
        from vllm_metal.metal import get_ops

        ids = sorted(set(ids))
        indices = mx.array(ids, dtype=mx.int32)
        for i, page in enumerate(self.pages):
            self.pages[i] = get_ops().gdn_state_scatter(page, page, indices, zero=True)

    def scatter_rows(self, rows: Sequence[mx.array], block_ids: Sequence[int]) -> None:
        """Scatter ``rows[i]`` into page ``i`` at the ``block_ids`` slots.

        One entry in ``rows`` per page, as the KV offloading worker's loads
        provide. An MLX index-assign on a page view would never reach the
        backing buffer, so the write goes through the native scatter like
        ``copy_blocks`` does.
        """
        self.require_plain_pool("row scatters")
        if len(rows) != len(self.pages):
            raise ValueError(
                f"scatter_rows takes one row array per page: got {len(rows)} "
                f"for {len(self.pages)} pages"
            )
        if not rows:
            return
        from vllm_metal.metal import get_ops

        dst = mx.array(block_ids, dtype=mx.int32)
        for i, page_rows in enumerate(rows):
            self.pages[i] = get_ops().gdn_state_scatter(self.pages[i], page_rows, dst)
