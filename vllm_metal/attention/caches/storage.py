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
    compute_layer_kv_cache_shape_bytes,
    compute_layout_strides,
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

    Each descriptor names the region whose Metal buffer it aliases; region
    anchors re-wrap on every :meth:`KVCacheStorage.depend` call, so a view
    built after a write pulls the whole write history of its own region.
    """

    def __init__(
        self,
        storage: KVCacheStorage,
        tensors: Sequence[torch.Tensor],
        regions: Sequence[int] | None = None,
    ):
        self.storage = storage
        self.regions = list(regions) if regions is not None else None
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
        region = self.regions[index] if self.regions is not None else 0
        return get_ops().cache_view(
            self.storage.anchor(region).view(dtype), shape, strides, offset
        )

    def __setitem__(self, index: int, value: mx.array) -> None:
        # Native writes alias their input. Preserve the dependency for every
        # layer and state component, including aliases with a different dtype.
        self.storage.depend([value])


class KVCacheStorage:
    """Upstream cache layout on MLX, backed by one Metal buffer per region.

    Layer-compact layouts (LBNHC) give every layer a contiguous page span, so
    the backing splits into per-layer Metal buffers that each sit far below
    ``max_buffer_length`` — no single allocation caps the cache budget on
    high-memory hosts. Block-outermost layouts interleave groups inside one
    block and keep the whole-model single buffer, whose budget the planner
    caps. Either way layout, strides, and page semantics follow upstream's
    ``KVCacheTensor`` contract; only the backing granularity is ours.
    """

    def __init__(self, config: KVCacheConfig):
        self.config = config
        self._layout = KVCacheLayout[config.kv_cache_layout or "LBNHC"]
        self._region_by_address: dict[tuple[int, int, int], int] = {}
        self.specs = {
            name: (
                group.kv_cache_spec.kv_cache_specs[name]
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                else group.kv_cache_spec
            )
            for group in config.kv_cache_groups
            for name in group.layer_names
        }
        if self._layout.is_layer_compact:
            self.tensors = self._allocate_per_region()
        else:
            self.tensors = self._allocate_single_buffer()
        self.nbytes = sum(raw.numel() for raw in self._region_storages)
        self._init_anchors()
        self.pages = CacheViews(
            self, self._page_tensors, list(range(len(self._page_tensors)))
        )

    def _allocate_per_region(self) -> dict[str, torch.Tensor]:
        """Allocate one zeroed buffer per layer page span; views stay dense.

        A layer in a layer-compact layout owns ``num_blocks`` dense pages, so
        its region storage is exactly ``num_blocks * page_stride`` bytes and
        the per-layer logical view re-bases to offset zero. Regions are
        deduplicated globally by virtual address: layers that share pages —
        within one tensor (``layer_stride`` aliasing) or across overlaid
        group tensors (shared state pools) — share one region.
        """
        self._region_storages = []
        self._page_strides = []
        self._page_tensors = []
        self._region_by_name = {}
        tensors: dict[str, torch.Tensor] = {}
        num_blocks = self.config.num_blocks
        for tensor in self.config.kv_cache_tensors:
            spec = self.specs[tensor.layers[0]]
            shape = compute_layer_kv_cache_shape_bytes(spec, num_blocks)
            # 5-axis [L, B, H, N, C] strides for a single-layer layout; the
            # layer stride is dropped because each region holds one layer.
            strides = compute_layout_strides(spec, num_blocks, 1, self._layout)[1:]
            page_stride = strides[0]
            for i, name in enumerate(tensor.layers):
                offset = tensor.offset + i * tensor.layer_stride
                key = (offset, page_stride, spec.page_size_bytes)
                if key not in self._region_by_address:
                    raw = torch.zeros(page_stride * shape[0], dtype=torch.uint8)
                    self._region_by_address[key] = len(self._region_storages)
                    self._region_storages.append(raw)
                    self._page_strides.append(page_stride)
                    self._page_tensors.append(
                        torch.as_strided(
                            raw,
                            (num_blocks, spec.page_size_bytes),
                            (page_stride, 1),
                            0,
                        )
                    )
                idx = self._region_by_address[key]
                view = torch.as_strided(self._region_storages[idx], shape, strides, 0)
                dtype = getattr(spec, "dtype", None)
                if dtype is not None:
                    view = view.view(dtype)
                tensors[name] = view
                self._region_by_name[name] = idx
        return tensors

    def _allocate_single_buffer(self) -> dict[str, torch.Tensor]:
        """Allocate the whole backing as one buffer (block-outermost layouts)."""
        self._region_storages = []
        self._page_strides = []
        self._page_tensors = []
        self._region_by_name = {}
        allocated = allocate_kv_cache(
            self.config,
            torch.device("cpu"),
            self._layout,
        )
        first = next(iter(allocated.values()))
        backing = first.untyped_storage()
        raw = torch.empty(0, dtype=torch.uint8).set_(backing)
        # Import a CPU NumPy view once, requiring shared storage. Every MLX
        # view shares this buffer, so Metal tracks the aliases as one allocation.
        # MLX dimensions are int32 even when the allocation exceeds 2 GiB.
        # The engine may lower num_blocks across workers after planning each
        # worker's tensor size. Shape the backing by its physical page stride.
        page_bytes = self.config.kv_cache_tensors[0].block_stride
        self._region_storages.append(raw)
        self._page_strides.append(page_bytes)
        self._region_by_name = {name: 0 for name in allocated}
        # CoW/zeroing cover physical bytes exactly once. Layer-outermost
        # layouts place a logical block in disjoint regions of the allocation.
        regions = {}
        for tensor in self.config.kv_cache_tensors:
            for i, name in enumerate(tensor.layers):
                offset = tensor.offset + i * tensor.layer_stride
                size = self.specs[name].page_size_bytes
                regions[(offset, tensor.block_stride, size)] = torch.as_strided(
                    raw,
                    (self.config.num_blocks, size),
                    (tensor.block_stride, 1),
                    offset,
                )
        self._page_tensors = list(regions.values())
        return allocated

    def _init_anchors(self) -> None:
        """Build the per-region MLX arrays and the write-dependency chain."""
        self._base_anchors = [
            mx.asarray(raw.view(-1, page_stride).numpy(), copy=False)
            for raw, page_stride in zip(
                self._region_storages, self._page_strides, strict=True
            )
        ]
        self._chain_version = 0
        if self._layout.is_layer_compact:
            # The chain only orders writes; each region's anchor re-wraps its
            # own buffer plus the chain, rebuilt lazily in ``anchor()`` only
            # when the region is read after new writes land (one node per
            # stale region per read, not one per write per region).
            self._chain = mx.array([], mx.uint8)
            self._anchors = list(self._base_anchors)
            self._anchor_built_at = [0] * len(self._base_anchors)
        else:
            # Single buffer: the chain itself is the arena-backed anchor.
            self._chain = self._base_anchors[0]
            self._anchors = list(self._base_anchors)
            self._anchor_built_at = [0]

    @property
    def buffer(self) -> mx.array:
        """Ordering anchor for evaluation. Per-region mode it carries no data:
        evaluating it forces the recorded writes; reads pull their region's
        anchor instead."""
        return self._chain

    def anchor(self, region: int) -> mx.array:
        """Return the current depends-wrapped backing for one region.

        Rebuilt only when writes landed since the last read of *this* region,
        so steady-state decode adds one node per region read instead of one
        per write per region.
        """
        if self._layout.is_layer_compact:
            if self._anchor_built_at[region] != self._chain_version:
                self._anchors[region] = mx.depends(
                    self._base_anchors[region], [self._chain]
                )
                self._anchor_built_at[region] = self._chain_version
            return self._anchors[region]
        return self._chain

    def depend(self, arrays: Sequence[mx.array]) -> None:
        if not arrays:
            return
        self._chain = mx.depends(self._chain, list(arrays))
        self._chain_version += 1
        if not self._layout.is_layer_compact:
            self._anchors[0] = self._chain

    def views(self, tensors: Sequence[torch.Tensor], names: Sequence[str]) -> CacheViews:
        return CacheViews(
            self, tensors, [self._region_by_name[name] for name in names]
        )

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
        return [
            self.views(component, names)
            for component in zip(*states, strict=True)
        ]

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
