# SPDX-License-Identifier: Apache-2.0
"""Execute vLLM's physical cache layout on MLX; Torch is used only at init."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
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
from vllm.v1.worker.utils import allocate_kv_cache

from vllm_metal.tensor_bridge import (
    TORCH_TO_MLX_DTYPE,
    torch_to_mlx,
)


def allocate_kv_cache_lazily(
    config: KVCacheConfig,
    layout: KVCacheLayout,
    specs: Mapping[str, Any],
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
    vLLM's allocation verbatim in that case.

    The layer views are built by vLLM's own :func:`create_kv_cache_views`, so
    the layout (sizes, strides, offsets, dtype) is identical either way;
    ``tests/attention/test_lazy_kv_allocation.py`` pins that equality.
    """
    if config.needs_kv_cache_zeroing:
        return allocate_kv_cache(config, torch.device("cpu"), layout)

    sizes = {tensor.size for tensor in config.kv_cache_tensors}
    if len(sizes) != 1:
        raise ValueError("KV cache tensors must share one backing allocation.")
    backing = torch.empty(sizes.pop(), dtype=torch.int8, device="cpu")
    caches: dict[str, torch.Tensor] = {}
    for tensor in config.kv_cache_tensors:
        spec = specs[tensor.layers[0]]
        if not spec.has_layer_views:
            caches.update((name, backing) for name in tensor.layers)
            continue
        views = create_kv_cache_views(
            backing,
            spec,
            config.num_blocks_of(tensor),
            layout,
            tensor,
        )
        caches.update(zip(tensor.layers, views, strict=True))
    return caches


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
        return get_ops().as_strided(
            self.storage._get_buffer(region).view(dtype), shape, strides, offset
        )

    def __setitem__(self, index: int, value: mx.array) -> None:
        # Native writes alias their input. Track the write on its region so
        # sibling views, including aliases with a different dtype, see it.
        region = self.descriptors[index][0]
        current = self.storage._get_buffer(region)
        self.storage._set_buffer(region, mx.depends(current, [value]))


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
        self.tensors = allocate_kv_cache_lazily(
            config,
            KVCacheLayout[config.kv_cache_layout],
            self.specs,
        )
        first = next(iter(self.tensors.values()))
        backing = first.untyped_storage()
        self._raw_backing = torch.empty(0, dtype=torch.uint8).set_(backing)
        self.nbytes = backing.nbytes()
        # Metal advertises LBNHC: each distinct layer address starts a compact
        # region. Record the byte ranges now; create MLX buffers lazily below
        # so the pool's pages stay uncommitted until the first GPU access.
        self.region_offsets = sorted(
            {
                tensor.offset + i * tensor.layer_stride
                for tensor in config.kv_cache_tensors
                for i in range(len(tensor.layers))
            }
        )
        ends = [*self.region_offsets[1:], self.nbytes]
        self._region_ranges = list(
            zip(self.region_offsets, ends, strict=True)
        )
        # Lazy cache: populated on first access to each region.
        self._buffer_cache: list[mx.array | None] = [None] * len(self._region_ranges)
        # CoW/zeroing cover physical bytes exactly once. Layer-outer layouts
        # place a logical block in disjoint regions of this one allocation.
        regions = {}
        for tensor in config.kv_cache_tensors:
            for i, name in enumerate(tensor.layers):
                offset = tensor.offset + i * tensor.layer_stride
                size = self.specs[name].page_size_bytes
                regions[(offset, tensor.block_stride, size)] = torch.as_strided(
                    self._raw_backing,
                    (config.num_blocks, size),
                    (tensor.block_stride, 1),
                    offset,
                )
        self.pages = CacheViews(self, list(regions.values()))

    @property
    def buffers(self) -> list[mx.array]:
        """Materialize MLX Metal buffers lazily — pages commit only on first GPU use."""
        result = []
        for idx, (start, end) in enumerate(self._region_ranges):
            buf = self._buffer_cache[idx]
            if buf is None:
                buf = self._materialize_region(idx, start, end)
            result.append(buf)
        return result

    def _get_buffer(self, region: int) -> mx.array:
        """Return a region's MLX buffer, materializing it lazily if needed."""
        buf = self._buffer_cache[region]
        if buf is None:
            start, end = self._region_ranges[region]
            buf = self._materialize_region(region, start, end)
        return buf

    def _set_buffer(self, region: int, value: mx.array) -> None:
        """Update a region's buffer (used by mx.depends chains from CacheViews)."""
        self._buffer_cache[region] = value

    def _materialize_region(self, region: int, start: int, end: int) -> mx.array:
        buf = torch_to_mlx(
            self._raw_backing[start:end].view(self.config.num_blocks, -1)
        )
        self._buffer_cache[region] = buf
        return buf

    @buffers.setter
    def buffers(self, value: list[mx.array]) -> None:
        """Allow direct assignment (used by ``mx.depends`` chains and tests)."""
        self._buffer_cache = list(value)

    @property
    def released(self) -> bool:
        return not self.tensors

    def release(self) -> None:
        """Free the backing memory. Its contents are discarded."""
        self.tensors = {}
        self._raw_backing = None  # type: ignore[assignment]
        self._buffer_cache = []

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
