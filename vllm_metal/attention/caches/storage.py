# SPDX-License-Identifier: Apache-2.0
"""Execute vLLM's physical cache layout on MLX; Torch is used only at init."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import torch
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheLayout,
    UniformTypeKVCacheSpecs,
    create_kv_cache_views,
)

from vllm_metal.tensor_bridge import (
    TORCH_TO_MLX_DTYPE,
    mlx_to_torch,
    torch_to_mlx,
)

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


def region_offsets_of(config: KVCacheConfig) -> list[int]:
    """Byte offsets where a layer's compact region starts, ascending.

    Metal advertises LBNHC, so each distinct layer address begins a region that
    runs to the next one. Cache groups overlay the same bytes, so a region can
    serve layers of more than one group.
    """
    return sorted(
        {
            tensor.offset + i * tensor.layer_stride
            for tensor in config.kv_cache_tensors
            for i in range(len(tensor.layers))
        }
    )


def _new_region(nbytes: int, *, zero: bool) -> torch.Tensor:
    """One Metal buffer of exactly ``nbytes`` as a flat uint8 MPS tensor.

    Not torch's allocator: it packs requests under 512 MiB into 1 GiB heaps.
    """
    from vllm_metal.metal import get_ops

    region = mlx_to_torch(get_ops().new_shared_buffer(nbytes), "mps").view(-1)
    if zero:
        region.zero_()
        torch.mps.synchronize()
    return region


def allocate_kv_cache_regions(
    config: KVCacheConfig,
    layout: KVCacheLayout,
    specs: Mapping[str, Any],
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    """Allocate one Metal buffer per region and view each layer inside its own.

    ``vllm.v1.worker.utils.allocate_kv_cache`` makes one allocation for the
    whole pool and zero-fills it. Metal runs on unified memory, so that memset
    writes every page of a multi-GB pool and the entire pool becomes resident
    from the first step — even though a serving run only ever touches the
    blocks it actually allocates (a 16 GB Mac serving a 0.6B model pays ~5 GB
    of RSS for a pool that is ~90% unused).

    The zero fill is only load-bearing where vLLM says it is:
    ``KVCacheConfig.needs_kv_cache_zeroing`` marks Mamba state (read before it
    is written) and mixed-precision caches (a block reinterpreted under another
    precision would decode stale bytes as NaN/Inf). Uniform-precision attention
    caches skip per-block zeroing upstream, and their kernels mask every slot
    past the sequence length, so unwritten memory never reaches an output.

    The regions are Metal buffers MLX refers to but does not own, so they are
    not wired and their pages are committed only when the GPU touches them
    (#1198). Torch writes to them are asynchronous; synchronize before MLX
    reads. The layer views come from vLLM's own :func:`create_kv_cache_views`,
    so the layout matches the single-allocation one;
    ``tests/attention/test_lazy_kv_allocation.py`` pins that.
    """
    sizes = {tensor.size for tensor in config.kv_cache_tensors}
    if len(sizes) != 1:
        raise ValueError("KV cache tensors must share one backing allocation.")
    nbytes = sizes.pop()
    offsets = region_offsets_of(config)
    ends = [*offsets[1:], nbytes]
    regions = [
        _new_region(end - start, zero=config.needs_kv_cache_zeroing)
        for start, end in zip(offsets, ends, strict=True)
    ]
    caches: dict[str, torch.Tensor] = {}
    for tensor in config.kv_cache_tensors:
        spec = specs[tensor.layers[0]]
        for i, name in enumerate(tensor.layers):
            start = tensor.offset + i * tensor.layer_stride
            region = regions[bisect_right(offsets, start) - 1]
            if not spec.has_layer_views:
                caches[name] = region
                continue
            (view,) = create_kv_cache_views(
                region,
                spec,
                config.num_blocks_of(tensor),
                layout,
                replace(tensor, size=region.nbytes, layers=[name], offset=0),
            )
            caches[name] = view
    return regions, caches


class CacheViews(Sequence[mx.array]):
    """Layer views following the latest native write to their shared backing."""

    def __init__(self, storage: KVCacheStorage, tensors: Sequence[torch.Tensor]):
        self.storage = storage
        self.descriptors = []
        for t in tensors:
            # Each region is its own allocation; the storage names the region.
            self.descriptors.append(
                (
                    storage.region_index[t.untyped_storage().data_ptr()],
                    tuple(t.shape),
                    tuple(t.stride()),
                    t.storage_offset(),
                    TORCH_TO_MLX_DTYPE[t.dtype],
                )
            )

    def __len__(self) -> int:
        return len(self.descriptors)

    def __getitem__(self, index: int) -> mx.array:
        from vllm_metal.metal import get_ops

        region, shape, strides, offset, dtype = self.descriptors[index]
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
        self._bind()

    def _bind(self) -> None:
        config = self.config
        regions, self.tensors = allocate_kv_cache_regions(
            config,
            KVCacheLayout[config.kv_cache_layout],
            self.specs,
        )
        self.nbytes = sum(region.nbytes for region in regions)
        self.region_offsets = region_offsets_of(config)
        self.region_index = {
            region.untyped_storage().data_ptr(): i for i, region in enumerate(regions)
        }
        self.buffers = [
            torch_to_mlx(region.view(config.num_blocks, -1)) for region in regions
        ]
        # CoW/zeroing cover physical bytes exactly once. Layer-outer layouts
        # place a logical block in disjoint regions.
        pages = {}
        for tensor in config.kv_cache_tensors:
            for i, name in enumerate(tensor.layers):
                offset = tensor.offset + i * tensor.layer_stride
                region = regions[bisect_right(self.region_offsets, offset) - 1]
                size = self.specs[name].page_size_bytes
                pages[(offset, tensor.block_stride, size)] = torch.as_strided(
                    region, (config.num_blocks, size), (tensor.block_stride, 1), 0
                )
        self.pages = CacheViews(self, list(pages.values()))

    @property
    def released(self) -> bool:
        return not self.buffers

    def release(self) -> None:
        """Free the backing memory. Its contents are discarded."""
        self.tensors = {}
        self.buffers = []

    def reallocate(self) -> None:
        """Re-create the backing memory with the layout vLLM scheduled against."""
        nbytes, region_offsets = self.nbytes, self.region_offsets
        self._bind()
        if (self.nbytes, self.region_offsets) != (nbytes, region_offsets):
            raise RuntimeError("Reallocated KV cache changed its layout")

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
            )
            MambaBase.bind_kv_cache(binding, self.tensors[name])
            states.append(binding.kv_cache)
        return [self.views(component) for component in zip(*states, strict=True)]

    def copy_blocks(self, pairs: Sequence[tuple[int, int]]) -> None:
        if not pairs:
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
