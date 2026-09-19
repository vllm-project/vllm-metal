# SPDX-License-Identifier: Apache-2.0
"""Execute vLLM's physical cache layout on MLX; Torch is used only at init."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import mlx.core as mx
import torch
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheLayout,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import allocate_kv_cache

from vllm_metal.pytorch_backend.tensor_bridge import TORCH_TO_MLX_DTYPE


class CacheViews(Sequence[mx.array]):
    """Layer views following the latest native write to their shared backing.

    Ordering between native writes and later readers is carried entirely by
    the dependency chain, not by evaluation order: ``__setitem__`` records
    every write (native scatter, rebinding) into the storage's depends chain,
    and each ``__getitem__`` rebuilds its view from that chain's current head.
    A scheduler zero issued after a pending-state flush therefore reads a view
    that already depends on the flushed write — this construction is what
    keeps flush-then-zero, CoW copies, and decode reads ordered without any
    explicit synchronization.
    """

    def __init__(self, storage: KVCacheStorage, tensors: Sequence[torch.Tensor]):
        self.storage = storage
        self.descriptors = [
            (
                tuple(t.shape),
                tuple(t.stride()),
                t.storage_offset(),
                TORCH_TO_MLX_DTYPE[t.dtype],
            )
            for t in tensors
        ]

    def __len__(self) -> int:
        return len(self.descriptors)

    def __getitem__(self, index: int) -> mx.array:
        from vllm_metal.metal import get_ops

        shape, strides, offset, dtype = self.descriptors[index]
        return get_ops().cache_view(
            self.storage.buffer.view(dtype), shape, strides, offset
        )

    def __setitem__(self, index: int, value: mx.array) -> None:
        # Native writes alias their input. Preserve the dependency for every
        # layer and state component, including aliases with a different dtype.
        self.storage.depend([value])


class KVCacheStorage:
    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.tensors = allocate_kv_cache(
            config,
            torch.device("cpu"),
            KVCacheLayout[config.kv_cache_layout or "LBNHC"],
        )
        first = next(iter(self.tensors.values()))
        backing = first.untyped_storage()
        raw = torch.empty(0, dtype=torch.uint8).set_(backing)
        # Import a CPU NumPy view once, requiring shared storage. Every MLX
        # view shares this buffer, so Metal tracks the aliases as one allocation.
        # MLX dimensions are int32 even when the allocation exceeds 2 GiB.
        # The engine may lower num_blocks across workers after planning each
        # worker's tensor size. Shape the backing by its physical page stride.
        page_bytes = config.kv_cache_tensors[0].block_stride
        self.buffer = mx.asarray(raw.view(-1, page_bytes).numpy(), copy=False)
        self.nbytes = backing.nbytes()
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
            self.buffer = mx.depends(self.buffer, list(arrays))

    def views(self, tensors: Sequence[torch.Tensor]) -> CacheViews:
        return CacheViews(self, tensors)

    def state_views(self, names: Sequence[str]) -> list[CacheViews]:
        states = []
        for name in names:
            spec = self.specs[name]
            assert isinstance(spec, MambaSpec)
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
        self._check_ids((*src, *dst))
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
        self._check_ids(ids)
        indices = mx.array(ids, dtype=mx.int32)
        for i, page in enumerate(self.pages):
            self.pages[i] = get_ops().gdn_state_scatter(page, page, indices, zero=True)

    def _check_ids(self, ids: Sequence[int]) -> None:
        if any(i < 0 or i >= self.config.num_blocks for i in ids):
            raise ValueError("cache block ID outside the upstream allocation")
