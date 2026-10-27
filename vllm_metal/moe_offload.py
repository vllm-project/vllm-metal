# SPDX-License-Identifier: Apache-2.0
"""Stream routed MoE experts from the checkpoint instead of keeping them resident.

Experts are the bulk of an MoE checkpoint, but each token uses only ``top_k``
of them. With offloading enabled, the expert tensors stay on disk and each
MoE layer reads the experts its tokens route to, through a byte-bounded LRU
cache. The compute path is unchanged: the touched experts are stacked and the
layer runs mlx-lm's own ``SwitchGLU`` forward on them, so outputs are identical
to the fully resident model.

Two checkpoint layouts are read in place: experts stacked under
``<block>.switch_mlp`` (mlx-community conversions) and one tensor per expert
under ``<block>.experts.<j>`` (the Hugging Face layout mlx-lm stacks at load
time). mlx-vlm's ``moe_offload`` needs a repacked copy of the checkpoint and
computes each expert separately, so its outputs differ from ``SwitchGLU``.
"""

from __future__ import annotations

import json
import os
import struct
from collections import OrderedDict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten
from mlx_lm.models.switch_layers import SwitchGLU
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class TensorLocation:
    path: Path
    offset: int
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class ExpertSlice:
    """Byte range of one expert's tensor inside a safetensors file."""

    location: TensorLocation
    offset: int
    nbytes: int
    shape: tuple[int, ...]


class ExpertStore:
    """Byte-bounded LRU of expert weights, read from the checkpoint on a miss.

    ``ExpertStore.install()`` is the entry point: it swaps every ``SwitchGLU``
    of a lazily loaded mlx-lm model for an ``OffloadedSwitchGLU`` reading
    through one shared store.
    """

    # safetensors dtype -> (numpy dtype that reads the raw bytes, MLX dtype).
    DTYPES: dict[str, tuple[type, mx.Dtype]] = {
        "U32": (np.uint32, mx.uint32),
        "U8": (np.uint8, mx.uint8),
        "F16": (np.float16, mx.float16),
        "F32": (np.float32, mx.float32),
        "BF16": (np.uint16, mx.bfloat16),
    }
    PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
    COMPONENTS = ("weight", "scales", "biases")
    # Parallel preads per batch of cache misses; SSD reads are latency bound.
    READ_THREADS = 8

    def __init__(
        self,
        index: dict[str, TensorLocation],
        cache_bytes: int,
        stack_bytes: int,
        dtypes: dict[tuple[str, str, str], mx.Dtype],
    ) -> None:
        self._index = index
        self._cache_bytes = cache_bytes
        self._dtypes = dtypes
        # A forward stacks the experts it touches on top of the cache, and a
        # long prefill touches nearly every expert of a layer.
        self.reserved_bytes = cache_bytes + stack_bytes
        self._entries: OrderedDict[tuple[str, int], dict[str, dict[str, mx.array]]] = (
            OrderedDict()
        )
        # Opened up front: reads run on pool threads.
        self._handles = {
            path: os.open(path, os.O_RDONLY)
            for path in {location.path for location in index.values()}
        }
        self._pool = ThreadPoolExecutor(max_workers=self.READ_THREADS)
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    @classmethod
    def install(
        cls,
        model: nn.Module,
        model_name: str,
        revision: str | None,
        cache_bytes: int,
    ) -> ExpertStore:
        """Stream the experts of a lazily loaded model and load everything else.

        The expert arrays are dropped without being read, then the remaining
        parameters are evaluated.
        """
        index = cls._index_checkpoint(cls._checkpoint_dir(model_name, revision))
        targets = cls._switch_layers(model)
        for path, module in targets:
            cls._validate_layer(index, path, module)
        # Reserve and reload the resident model's dtypes, which may differ
        # from those on disk after applying the serving dtype.
        stack_bytes = max(
            sum(
                array.nbytes
                for projection in cls.PROJECTIONS
                for component in cls.COMPONENTS
                if (array := module[projection].get(component)) is not None
            )
            for _, module in targets
        )
        dtypes = {
            (path, projection, component): array.dtype
            for path, module in targets
            for projection in cls.PROJECTIONS
            for component in cls.COMPONENTS
            if (array := module[projection].get(component)) is not None
        }
        store = cls(index, cache_bytes, stack_bytes, dtypes)
        model.update_modules(
            tree_unflatten(
                [
                    (path, OffloadedSwitchGLU(module, path, store))
                    for path, module in targets
                ]
            )
        )
        mx.eval(model.parameters())
        logger.info(
            "MoE expert offloading: %d layers stream experts from %s with a "
            "%.2f GiB cache.",
            len(targets),
            model_name,
            cache_bytes / 2**30,
        )
        return store

    def get(
        self, module_path: str, experts: Iterable[int]
    ) -> list[dict[str, dict[str, mx.array]]]:
        """Return ``{projection: {component: array}}`` per expert, in order."""
        experts = list(experts)
        missing = [e for e in experts if (module_path, e) not in self._entries]
        jobs = [
            (expert, projection, component, piece)
            for expert in missing
            for projection in self.PROJECTIONS
            for component in self.COMPONENTS
            if (
                piece := self._expert_slice(
                    self._index, module_path, projection, component, expert
                )
            )
            is not None
        ]
        # Reads run in parallel; arrays are created on the calling thread.
        raws = list(self._pool.map(lambda job: self._read(job[3]), jobs))
        for (expert, projection, component, piece), raw in zip(jobs, raws, strict=True):
            entry = self._entries.setdefault(
                (module_path, expert), {p: {} for p in self.PROJECTIONS}
            )
            array = self._to_array(piece, raw).astype(
                self._dtypes[(module_path, projection, component)]
            )
            entry[projection][component] = array
            self.bytes += array.nbytes
        self.misses += len(missing)
        self.hits += len(experts) - len(missing)
        for expert in experts:
            self._entries.move_to_end((module_path, expert))
        self._evict(keep=len(experts))
        return [self._entries[(module_path, expert)] for expert in experts]

    def clear(self) -> None:
        self._entries.clear()
        self.bytes = 0

    def _read(self, piece: ExpertSlice) -> bytes:
        return os.pread(self._handles[piece.location.path], piece.nbytes, piece.offset)

    def _evict(self, *, keep: int) -> None:
        # The experts of the current call stay cached even when they alone
        # exceed the budget; the layer needs all of them at once.
        while self.bytes > self._cache_bytes and len(self._entries) > keep:
            _, entry = self._entries.popitem(last=False)
            self.bytes -= sum(
                array.nbytes
                for components in entry.values()
                for array in components.values()
            )

    @staticmethod
    def _checkpoint_dir(model_name: str, revision: str | None) -> Path:
        """Local directory holding the safetensors shards mlx-lm just loaded."""
        path = Path(model_name)
        if path.is_dir():
            return path
        return Path(
            snapshot_download(
                model_name, revision=revision, allow_patterns=["*.safetensors"]
            )
        )

    @classmethod
    def _index_checkpoint(cls, model_dir: Path) -> dict[str, TensorLocation]:
        index: dict[str, TensorLocation] = {}
        for path in sorted(model_dir.glob("*.safetensors")):
            with path.open("rb") as handle:
                (header_len,) = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(header_len))
            data_start = 8 + header_len
            for name, meta in header.items():
                # An expert tensor in another dtype fails _validate_layer
                # with its name.
                if name == "__metadata__" or meta["dtype"] not in cls.DTYPES:
                    continue
                index[name] = TensorLocation(
                    path,
                    data_start + meta["data_offsets"][0],
                    tuple(meta["shape"]),
                    meta["dtype"],
                )
        return index

    @staticmethod
    def _switch_layers(model: nn.Module) -> list[tuple[str, SwitchGLU]]:
        targets = sorted(
            (path, module)
            for path, module in model.named_modules()
            if isinstance(module, SwitchGLU)
        )
        if not targets:
            raise ValueError(
                "MoE expert offloading requires a model with mlx-lm SwitchGLU experts."
            )
        return targets

    @classmethod
    def _validate_layer(
        cls, index: dict[str, TensorLocation], module_path: str, module: SwitchGLU
    ) -> None:
        last = module.gate_proj.num_experts - 1
        for projection in cls.PROJECTIONS:
            for component in cls.COMPONENTS:
                if module[projection].get(component) is None:
                    continue
                for expert in (0, last):
                    piece = cls._expert_slice(
                        index, module_path, projection, component, expert
                    )
                    if piece is None:
                        raise KeyError(
                            "MoE expert offloading found no checkpoint tensor for "
                            f"{module_path}.{projection}.{component} expert "
                            f"{expert}. Supported layouts: stacked "
                            "'<block>.switch_mlp.<proj>.<component>' and "
                            "per-expert '<block>.experts.<j>.<proj>.<component>'."
                        )

    @classmethod
    def _expert_slice(
        cls,
        index: dict[str, TensorLocation],
        module_path: str,
        projection: str,
        component: str,
        expert: int,
    ) -> ExpertSlice | None:
        stacked = index.get(f"{module_path}.{projection}.{component}")
        if stacked is not None:
            nbytes = cls._nbytes(stacked.shape[1:], stacked.dtype)
            return ExpertSlice(
                stacked, stacked.offset + expert * nbytes, nbytes, stacked.shape[1:]
            )
        block_path = module_path.rsplit(".", 1)[0]
        single = index.get(f"{block_path}.experts.{expert}.{projection}.{component}")
        if single is not None:
            return ExpertSlice(
                single,
                single.offset,
                cls._nbytes(single.shape, single.dtype),
                single.shape,
            )
        return None

    @classmethod
    def _nbytes(cls, shape: tuple[int, ...], dtype: str) -> int:
        return int(np.prod(shape)) * np.dtype(cls.DTYPES[dtype][0]).itemsize

    @classmethod
    def _to_array(cls, piece: ExpertSlice, raw: bytes) -> mx.array:
        np_dtype, mx_dtype = cls.DTYPES[piece.location.dtype]
        array = mx.array(np.frombuffer(raw, dtype=np_dtype).reshape(piece.shape))
        return array.view(mx_dtype) if array.dtype != mx_dtype else array


class OffloadedSwitchGLU(nn.Module):
    """``SwitchGLU`` whose expert weights are gathered per call from an ``ExpertStore``.

    The projection modules keep their quantization settings and are given
    zero-expert placeholders between calls, so the module tree stays valid for
    ``parameters()`` and ``mx.eval`` without holding expert weights.
    """

    def __init__(self, inner: SwitchGLU, module_path: str, store: ExpertStore) -> None:
        super().__init__()
        self.gate_proj = inner.gate_proj
        self.up_proj = inner.up_proj
        self.down_proj = inner.down_proj
        self.activation = inner.activation
        placeholders: dict[tuple[str, str], mx.array] = {}
        for projection in ExpertStore.PROJECTIONS:
            module = self[projection]
            for component in ExpertStore.COMPONENTS:
                array = module.get(component)
                if array is not None:
                    placeholders[(projection, component)] = mx.zeros(
                        (0, *array.shape[1:]), array.dtype
                    )
        object.__setattr__(self, "_module_path", module_path)
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_num_experts", inner.gate_proj.num_experts)
        object.__setattr__(self, "_placeholders", placeholders)
        self._reset()

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        # Routing must be known on the host to pick the experts to read.
        touched = np.unique(np.asarray(indices)).tolist()
        entries = self._store.get(self._module_path, touched)
        for projection, component in self._placeholders:
            self[projection][component] = mx.stack(
                [entry[projection][component] for entry in entries]
            )
        compact = mx.zeros((self._num_experts,), dtype=indices.dtype)
        compact[mx.array(touched)] = mx.arange(len(touched), dtype=indices.dtype)
        try:
            return SwitchGLU.__call__(self, x, compact[indices])
        finally:
            self._reset()

    def _reset(self) -> None:
        for (projection, component), placeholder in self._placeholders.items():
            self[projection][component] = placeholder
