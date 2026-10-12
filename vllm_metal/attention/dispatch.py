# SPDX-License-Identifier: Apache-2.0
"""Named selection of standalone paged-attention runtime families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

import vllm_metal.envs as envs
from vllm_metal.attention.caches.turboquant import FWHT_SUPPORTED_HEAD_DIMS
from vllm_metal.attention.runtime.protocol import PagedAttentionRuntime
from vllm_metal.metal.constants import KERNEL_BLOCK_SIZES

# Includes the per-token fallback specializations, not just tiled/NAX prefill.
PAGED_ATTENTION_HEAD_SIZES = (64, 80, 96, 112, 128, 192, 256, 512)


@dataclass(frozen=True)
class AttnCaps:
    is_mla: bool
    is_hybrid: bool
    kv_paged: bool
    dtype: mx.Dtype
    head_dims: tuple[int, ...]
    block_size: int
    turboquant: bool = False


class PagedAttentionRuntimeBuilder(Protocol):
    """Cache-policy factories retaining ownership of model/cache geometry."""

    def _build_sdpa_backend(self, block_size: int) -> PagedAttentionRuntime: ...
    def _build_mla_backend(self, block_size: int) -> PagedAttentionRuntime: ...


class MetalAttnBackend(Protocol):
    name: str

    @classmethod
    def validate(cls, caps: AttnCaps) -> list[str]: ...

    @classmethod
    def build_runtime(
        cls, builder: PagedAttentionRuntimeBuilder, *, block_size: int
    ) -> PagedAttentionRuntime: ...


_REGISTRY: dict[str, type[MetalAttnBackend]] = {}


def register_attention_backend[Backend: type[MetalAttnBackend]](
    backend: Backend,
) -> Backend:
    """Register a backend without replacing an existing implementation."""
    if not backend.name:
        raise ValueError("An attention backend must have a non-empty name")
    if backend.name in _REGISTRY:
        raise ValueError(f"Attention backend {backend.name!r} is already registered")
    _REGISTRY[backend.name] = backend
    return backend


def _paged_runtime_reasons(caps: AttnCaps) -> list[str]:
    reasons = []
    if not caps.kv_paged:
        reasons.append("requires paged KV cache")
    if caps.is_hybrid:
        reasons.append("does not support hybrid models")
    if caps.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        reasons.append(
            f"unsupported KV cache dtype {caps.dtype}; requires fp16/bf16/fp32"
        )
    if caps.block_size <= 0:
        reasons.append(f"block_size must be positive, got {caps.block_size}")
    return reasons


@register_attention_backend
class PagedMetalMHA:
    name = "paged_metal_mha"

    @classmethod
    def validate(cls, caps: AttnCaps) -> list[str]:
        reasons = _paged_runtime_reasons(caps)
        if caps.is_mla:
            reasons.append("requires a non-MLA model (MHA/GQA/MQA)")
        if not caps.head_dims:
            reasons.append("requires at least one attention head dimension")
        unsupported = sorted(set(caps.head_dims) - set(PAGED_ATTENTION_HEAD_SIZES))
        if unsupported:
            reasons.append(
                f"unsupported head dimensions {unsupported}; "
                f"supported: {PAGED_ATTENTION_HEAD_SIZES}"
            )
        if caps.block_size > 0 and not any(
            caps.block_size % size == 0 for size in KERNEL_BLOCK_SIZES
        ):
            reasons.append(
                f"block_size {caps.block_size} is not divisible by any "
                f"supported kernel block size {KERNEL_BLOCK_SIZES}"
            )
        if caps.turboquant and any(
            dim not in FWHT_SUPPORTED_HEAD_DIMS for dim in caps.head_dims
        ):
            reasons.append(
                f"TurboQuant requires head dimensions in {FWHT_SUPPORTED_HEAD_DIMS}"
            )
        return reasons

    @classmethod
    def build_runtime(
        cls, builder: PagedAttentionRuntimeBuilder, *, block_size: int
    ) -> PagedAttentionRuntime:
        return builder._build_sdpa_backend(block_size=block_size)


@register_attention_backend
class PagedMetalMLA:
    name = "paged_metal_mla"

    @classmethod
    def validate(cls, caps: AttnCaps) -> list[str]:
        reasons = _paged_runtime_reasons(caps)
        if not caps.is_mla:
            reasons.append("requires an MLA model")
        if caps.turboquant:
            reasons.append("TurboQuant is not supported for MLA models")
        return reasons

    @classmethod
    def build_runtime(
        cls, builder: PagedAttentionRuntimeBuilder, *, block_size: int
    ) -> PagedAttentionRuntime:
        return builder._build_mla_backend(block_size=block_size)


def candidate_backends(caps: AttnCaps) -> tuple[str, ...]:
    """Preference order for each standalone runtime family."""
    if caps.is_hybrid:
        return ()
    if caps.is_mla:
        return (PagedMetalMLA.name,)
    return (PagedMetalMHA.name,)


def select_attention_backend(caps: AttnCaps) -> type[MetalAttnBackend]:
    """Select the first compatible candidate, or validate a named override."""
    forced = envs.VLLM_METAL_ATTENTION_BACKEND
    if forced is not None:
        if forced not in _REGISTRY:
            raise ValueError(
                f"Unknown VLLM_METAL_ATTENTION_BACKEND={forced!r}. "
                f"Available backends: {', '.join(sorted(_REGISTRY))}"
            )
        backend = _REGISTRY[forced]
        reasons = backend.validate(caps)
        if reasons:
            raise ValueError(
                f"VLLM_METAL_ATTENTION_BACKEND={forced!r} is incompatible: "
                f"{'; '.join(reasons)}"
            )
        return backend

    rejected = []
    for name in candidate_backends(caps):
        backend = _REGISTRY[name]
        reasons = backend.validate(caps)
        if not reasons:
            return backend
        rejected.append(f"{name}: {'; '.join(reasons)}")
    detail = "; ".join(rejected) or "no registered candidates for this runtime family"
    raise ValueError(f"No compatible Metal attention backend: {detail}")
