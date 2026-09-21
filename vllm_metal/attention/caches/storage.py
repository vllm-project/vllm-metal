# SPDX-License-Identifier: Apache-2.0
"""Execute vLLM's physical cache layout on MLX; Torch is used only at init."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from types import SimpleNamespace

import mlx.core as mx
import torch
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheLayout,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import allocate_kv_cache

from vllm_metal.pytorch_backend.tensor_bridge import (
    TORCH_TO_MLX_DTYPE,
    torch_to_mlx,
)


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
            self.storage.buffers[region].view(dtype), shape, strides, offset
        )

    def __setitem__(self, index: int, value: mx.array) -> None:
        # Native writes alias their input. Track the write on its region so
        # sibling views, including aliases with a different dtype, see it.
        region = self.descriptors[index][0]
        self.storage.buffers[region] = mx.depends(self.storage.buffers[region], [value])


class KVCacheStorage:
    """Consume vLLM's resolved cache layout and scheduler-owned block IDs."""

    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.tensors = allocate_kv_cache(
            config,
            torch.device("cpu"),
            KVCacheLayout[config.kv_cache_layout],
        )
        first = next(iter(self.tensors.values()))
        backing = first.untyped_storage()
        raw = torch.empty(0, dtype=torch.uint8).set_(backing)
        self.nbytes = backing.nbytes()
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
        ends = [*self.region_offsets[1:], self.nbytes]
        self.buffers = [
            torch_to_mlx(raw[start:end].view(config.num_blocks, -1))
            for start, end in zip(self.region_offsets, ends, strict=True)
        ]
        self.specs = {
            name: (
                group.kv_cache_spec.kv_cache_specs[name]
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                else group.kv_cache_spec
            )
            for group in config.kv_cache_groups
            for name in group.layer_names
        }
        # CoW/zeroing cover physical bytes exactly once. Layer-outer layouts
        # place a logical block in disjoint regions of this one allocation.
        regions = {}
        for tensor in config.kv_cache_tensors:
            for i, name in enumerate(tensor.layers):
                offset = tensor.offset + i * tensor.layer_stride
                size = self.specs[name].page_size_bytes
                regions[(offset, tensor.block_stride, size)] = torch.as_strided(
                    raw, (config.num_blocks, size), (tensor.block_stride, 1), offset
                )
        self.pages = CacheViews(self, list(regions.values()))

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
