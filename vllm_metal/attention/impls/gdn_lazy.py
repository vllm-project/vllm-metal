# SPDX-License-Identifier: Apache-2.0
"""Lazy Gated DeltaNet kernels for MLX/Metal execution."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, ClassVar

import mlx.core as mx

import vllm_metal.envs as envs
from vllm_metal.attention.caches.gdn_cache import GDNPagedStateCache
from vllm_metal.metal import _read_v2_metal_source

_GDN_CONV1D_V2_SOURCE = _read_v2_metal_source("gdn_conv1d_silu_decode.metal")
_GDN_CONV1D_PREFILL_V2_SOURCE = _read_v2_metal_source("gdn_conv1d_silu_prefill.metal")
_GDN_RECURRENT_DECODE_V2_SOURCE = _read_v2_metal_source("gdn_recurrent_decode.metal")
_GDN_RECURRENT_PREFILL_V2_SOURCE = _read_v2_metal_source("gdn_recurrent_prefill.metal")
_RECURRENT_SIMDGROUP_WIDTH = 32
_RECURRENT_MAX_KEY_HEAD_DIM = 256
_RECURRENT_PREFILL_THREADGROUP_DV = 4


@dataclass(frozen=True)
class GDNRecurrentRequest:
    """Common inputs for one lazy GDN recurrent kernel attempt."""

    q: mx.array
    k: mx.array
    v: mx.array
    g: mx.array
    beta: mx.array
    state_cache: GDNPagedStateCache
    cache_idx: int
    slot_ids: list[int]
    output_dtype: mx.Dtype

    @property
    def total_tokens(self) -> int:
        return self.q.shape[1]

    @property
    def num_key_heads(self) -> int:
        return self.q.shape[2]

    @property
    def num_value_heads(self) -> int:
        return self.v.shape[2]

    @property
    def key_head_dim(self) -> int:
        return self.q.shape[3]

    @property
    def value_head_dim(self) -> int:
        return self.v.shape[3]


@dataclass(frozen=True)
class GDNRecurrentDecodeRequest(GDNRecurrentRequest):
    """Inputs for one lazy GDN recurrent decode attempt."""

    threadgroup_dv: int = 4


@dataclass(frozen=True)
class GDNRecurrentPrefillRequest(GDNRecurrentRequest):
    """Inputs for one lazy GDN recurrent prefill-containing attempt."""

    cu_seqlens: list[int]
    compute_dtype: mx.Dtype | None = None
    defer_state_scatter: bool = False


class GDNLazyKernels:
    """Owner for lazy GDN fast-path policy and kernels.

    The fast path uses ``mx.fast.metal_kernel`` source snippets from
    ``kernels_v2``.  Unlike the legacy recurrent C++ path, these kernels
    materialize compact conv/recurrent state updates.  Conv decode scatters
    updates back into the stable cache pool; recurrent decode and eligible
    prefill defer compact updates until the next slot-order change, fallback,
    materialize, or release boundary.  Callers fall back to the legacy path
    when a request shape is ineligible.
    """

    _shared: ClassVar[GDNLazyKernels | None] = None
    _shared_enabled: ClassVar[bool | None] = None
    _shared_lock: ClassVar[Lock] = Lock()

    def __init__(
        self,
        *,
        enabled: bool,
        conv_kernel: Any | None = None,
        conv_prefill_kernel: Any | None = None,
        recurrent_decode_kernel: Any | None = None,
        recurrent_prefill_kernel: Any | None = None,
    ) -> None:
        self._enabled = enabled
        if not enabled:
            self._conv_kernel = None
            self._conv_prefill_kernel = None
            self._recurrent_decode_kernel = None
            self._recurrent_prefill_kernel = None
            return

        self._conv_kernel = (
            conv_kernel
            if conv_kernel is not None
            else self._make_kernel(
                "gdn_conv1d_silu_decode_v2",
                ["input", "conv_state_in", "weights", "slot_mapping", "num_requests"],
                ["output", "conv_state_out"],
                _GDN_CONV1D_V2_SOURCE,
            )
        )
        self._conv_prefill_kernel = (
            conv_prefill_kernel
            if conv_prefill_kernel is not None
            else self._make_kernel(
                "gdn_conv1d_silu_prefill_v2",
                [
                    "input",
                    "conv_state_in",
                    "weights",
                    "cu_seqlens",
                    "slot_mapping",
                    "num_requests",
                    "total_tokens",
                ],
                ["output", "conv_state_out"],
                _GDN_CONV1D_PREFILL_V2_SOURCE,
            )
        )
        self._recurrent_decode_kernel = (
            recurrent_decode_kernel
            if recurrent_decode_kernel is not None
            else self._make_kernel(
                "gdn_recurrent_decode_v2",
                [
                    "q",
                    "k",
                    "v",
                    "g",
                    "beta",
                    "state_in",
                    "slot_mapping",
                    "num_requests",
                ],
                ["y", "state_out"],
                _GDN_RECURRENT_DECODE_V2_SOURCE,
            )
        )
        self._recurrent_prefill_kernel = (
            recurrent_prefill_kernel
            if recurrent_prefill_kernel is not None
            else self._make_kernel(
                "gdn_recurrent_prefill_v2",
                [
                    "q",
                    "k",
                    "v",
                    "g",
                    "beta",
                    "state_in",
                    "cu_seqlens",
                    "slot_mapping",
                    "num_requests",
                ],
                ["y", "state_out"],
                _GDN_RECURRENT_PREFILL_V2_SOURCE,
            )
        )

    @classmethod
    def from_env(cls) -> GDNLazyKernels:
        return cls(enabled=envs.VLLM_METAL_GDN_LAZY_KERNELS)

    @property
    def enabled(self) -> bool:
        """Return whether lazy GDN kernels are enabled for this owner."""
        return self._enabled

    @classmethod
    def shared(cls) -> GDNLazyKernels:
        """Get the process-level lazy GDN kernel owner.

        The env kill switch is read when the shared owner is acquired;
        existing wrappers keep the owner they stored at construction time.
        """
        with cls._shared_lock:
            enabled = envs.VLLM_METAL_GDN_LAZY_KERNELS
            if cls._shared is None or cls._shared_enabled != enabled:
                cls._shared = cls(enabled=enabled)
                cls._shared_enabled = enabled
            return cls._shared

    @classmethod
    def reset_shared_for_tests(cls) -> None:
        """Reset the shared lazy GDN kernel owner for tests."""
        with cls._shared_lock:
            cls._shared = None
            cls._shared_enabled = None

    @staticmethod
    def _make_kernel(
        name: str,
        input_names: list[str],
        output_names: list[str],
        source: str,
    ) -> Any | None:
        if not mx.metal.is_available():
            return None
        kernel = mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=source,
            ensure_row_contiguous=False,
        )

        def run(*, inputs, **kwargs):
            inputs = [
                mx.contiguous(value)
                if isinstance(value, mx.array)
                and key not in ("state_in", "conv_state_in")
                else value
                for key, value in zip(input_names, inputs, strict=True)
            ]
            return kernel(inputs=inputs, **kwargs)

        return run

    def try_conv_decode(
        self,
        mixed_qkv: mx.array,
        inner: Any,
        state_cache: GDNPagedStateCache,
        cache_idx: int,
        slot_ids: list[int],
    ) -> mx.array | None:
        """Run the lazy GDN conv decode fast path, or return None."""
        num_requests = len(slot_ids)
        total_tokens = mixed_qkv.shape[1]
        if not (
            self._enabled
            and self._conv_kernel is not None
            and total_tokens == num_requests
        ):
            return None

        conv_dim = state_cache.conv_dim
        kernel_size = inner.conv_kernel_size
        state_view = state_cache.conv_state_for_decode(cache_idx, slot_ids)
        conv_state_in = state_view.state
        weight = inner.conv1d.weight
        output_dtype = mx.result_type(mixed_qkv, conv_state_in, weight)

        mixed_qkv_2d = mixed_qkv.reshape(num_requests, conv_dim)
        slot_ids_arr = state_view.cache_slot_ids
        state_slot_ids_arr = state_view.state_slot_ids

        grid_size = num_requests * conv_dim
        tg_size = min(256, grid_size)
        state_updates_shape = (num_requests, kernel_size - 1, conv_dim)

        kernel_inputs = [
            mixed_qkv_2d,
            conv_state_in,
            weight,
            state_slot_ids_arr,
            num_requests,
        ]
        template = [
            ("T", output_dtype),
            ("StT", conv_state_in.dtype),
            ("CONV_DIM", conv_dim),
            ("KERNEL_SIZE", kernel_size),
        ]
        conv_silu_out, conv_state_updates = self._conv_kernel(
            inputs=kernel_inputs,
            template=template,
            grid=(grid_size, 1, 1),
            threadgroup=(tg_size, 1, 1),
            output_shapes=[(num_requests, conv_dim), state_updates_shape],
            output_dtypes=[output_dtype, conv_state_in.dtype],
        )
        state_cache.write_conv_rows(cache_idx, conv_state_updates, slot_ids_arr)
        if state_view.uses_compact_state:
            state_cache.clear_pending_conv_state(cache_idx)
        return conv_silu_out.reshape(1, total_tokens, conv_dim)

    def try_conv_prefill(
        self,
        mixed_qkv: mx.array,
        inner: Any,
        state_cache: GDNPagedStateCache,
        cache_idx: int,
        slot_ids: list[int],
        cu_seqlens: list[int],
    ) -> mx.array | None:
        """Run the lazy GDN conv prefill-containing fast path, or return None."""
        num_requests = len(slot_ids)
        total_tokens = mixed_qkv.shape[1]
        prefill_batch = (
            total_tokens > num_requests
            and len(cu_seqlens) == num_requests + 1
            and cu_seqlens[0] == 0
            and cu_seqlens[-1] == total_tokens
            and all(cu_seqlens[i + 1] > cu_seqlens[i] for i in range(num_requests))
        )
        if not (
            self._enabled and self._conv_prefill_kernel is not None and prefill_batch
        ):
            return None

        conv_dim = state_cache.conv_dim
        kernel_size = inner.conv_kernel_size
        state_len = kernel_size - 1
        if state_cache.has_pending_conv_state(cache_idx):
            state_cache.apply_pending_conv_state(cache_idx)
        conv_state_in = state_cache.conv_states[cache_idx]
        output_dtype = mx.result_type(mixed_qkv, conv_state_in, inner.conv1d.weight)
        mixed_qkv_2d = mixed_qkv.reshape(total_tokens, conv_dim)
        slot_ids_arr = mx.array(slot_ids, dtype=mx.int32)
        cu_seqlens_arr = mx.array(cu_seqlens, dtype=mx.int32)

        state_updates_shape = (num_requests, state_len, conv_dim)
        grid_size = (total_tokens + num_requests * state_len) * conv_dim
        tg_size = min(256, grid_size)
        conv_out, conv_state_updates = self._conv_prefill_kernel(
            inputs=[
                mixed_qkv_2d,
                conv_state_in,
                inner.conv1d.weight,
                cu_seqlens_arr,
                slot_ids_arr,
                num_requests,
                total_tokens,
            ],
            template=[
                ("T", output_dtype),
                ("StT", conv_state_in.dtype),
                ("CONV_DIM", conv_dim),
                ("KERNEL_SIZE", kernel_size),
            ],
            grid=(grid_size, 1, 1),
            threadgroup=(tg_size, 1, 1),
            output_shapes=[(total_tokens, conv_dim), state_updates_shape],
            output_dtypes=[output_dtype, conv_state_in.dtype],
        )
        state_cache.set_pending_conv_state(cache_idx, slot_ids, conv_state_updates)
        return conv_out.reshape(1, total_tokens, conv_dim)

    def try_recurrent_decode(
        self,
        request: GDNRecurrentDecodeRequest,
    ) -> mx.array | None:
        """Run the lazy GDN recurrent decode fast path, or return None."""
        total_tokens = request.total_tokens
        n_hk = request.num_key_heads
        n_hv = request.num_value_heads
        d_k = request.key_head_dim
        d_v = request.value_head_dim
        num_requests = len(request.slot_ids)
        # The Metal source uses one SIMD group across Dk and a
        # fixed-size per-thread register array, matching the original C++
        # kernel's practical key-head-dim envelope. Fall back for unusual head
        # dimensions rather than silently dropping remainder channels.
        recurrent_shape_supported = (
            d_k % _RECURRENT_SIMDGROUP_WIDTH == 0 and d_k <= _RECURRENT_MAX_KEY_HEAD_DIM
        )
        if not (
            self._enabled
            and self._recurrent_decode_kernel is not None
            and total_tokens == num_requests
            and recurrent_shape_supported
        ):
            return None

        state_cache = request.state_cache
        state_view = state_cache.recurrent_state_for_decode(
            request.cache_idx, request.slot_ids
        )
        state_in = state_view.state
        state_slot_ids_arr = state_view.state_slot_ids

        kernel_inputs = [
            request.q.reshape(total_tokens, n_hk, d_k),
            request.k.reshape(total_tokens, n_hk, d_k),
            request.v.reshape(total_tokens, n_hv, d_v),
            request.g.reshape(total_tokens, n_hv),
            request.beta.reshape(total_tokens, n_hv),
            state_in,
            state_slot_ids_arr,
            num_requests,
        ]
        template = [
            ("T", request.output_dtype),
            ("StT", state_in.dtype),
            ("Dk", d_k),
            ("Dv", d_v),
            ("Hk", n_hk),
            ("Hv", n_hv),
        ]
        y_out, state_updates = self._recurrent_decode_kernel(
            inputs=kernel_inputs,
            template=template,
            grid=(_RECURRENT_SIMDGROUP_WIDTH, d_v, num_requests * n_hv),
            threadgroup=(_RECURRENT_SIMDGROUP_WIDTH, request.threadgroup_dv, 1),
            output_shapes=[(total_tokens, n_hv, d_v), (num_requests, n_hv, d_v, d_k)],
            output_dtypes=[request.output_dtype, state_in.dtype],
        )
        # Park the compact update instead of scattering it into the stable
        # pool.  A steady-state decode chain then reads and writes only
        # ``num_requests`` slabs per step; without this each step also copies
        # the update into the pool, doubling recurrent state traffic.
        #
        # ``recurrent_state_for_decode`` above already drained any pending
        # update whose slot order did not match, so the only pending state
        # that can still be live here is this layer's own previous step —
        # which ``state_updates`` supersedes.  Clear it rather than letting
        # ``set_pending_recurrent_state`` scatter that stale slab first.
        state_cache.clear_pending_recurrent_state(request.cache_idx)
        state_cache.set_pending_recurrent_state(
            request.cache_idx, request.slot_ids, state_updates
        )
        return y_out

    def try_recurrent_prefill(
        self,
        request: GDNRecurrentPrefillRequest,
    ) -> mx.array | None:
        """Run the lazy GDN recurrent prefill-containing fast path, or return None."""
        total_tokens = request.total_tokens
        n_hk = request.num_key_heads
        n_hv = request.num_value_heads
        d_k = request.key_head_dim
        d_v = request.value_head_dim
        num_requests = len(request.slot_ids)
        recurrent_shape_supported = (
            d_k % _RECURRENT_SIMDGROUP_WIDTH == 0 and d_k <= _RECURRENT_MAX_KEY_HEAD_DIM
        )
        prefill_batch = (
            total_tokens > num_requests
            and len(request.cu_seqlens) == num_requests + 1
            and request.cu_seqlens[0] == 0
            and request.cu_seqlens[-1] == total_tokens
            and all(
                request.cu_seqlens[i + 1] - request.cu_seqlens[i] >= 1
                for i in range(num_requests)
            )
        )
        if not (
            self._enabled
            and self._recurrent_prefill_kernel is not None
            and prefill_batch
            and recurrent_shape_supported
        ):
            return None

        state_cache = request.state_cache
        if state_cache.has_pending_recurrent_state(request.cache_idx):
            state_cache.apply_pending_recurrent_state(request.cache_idx)
        state_in = state_cache.recurrent_states[request.cache_idx]
        slot_ids_arr = mx.array(request.slot_ids, dtype=mx.int32)
        cu_seqlens_arr = mx.array(request.cu_seqlens, dtype=mx.int32)
        kernel_dtype = request.compute_dtype or request.output_dtype
        state_dtype = state_in.dtype

        kernel_inputs = [
            request.q.reshape(total_tokens, n_hk, d_k).astype(kernel_dtype),
            request.k.reshape(total_tokens, n_hk, d_k).astype(kernel_dtype),
            request.v.reshape(total_tokens, n_hv, d_v).astype(kernel_dtype),
            request.g.reshape(total_tokens, n_hv).astype(kernel_dtype),
            request.beta.reshape(total_tokens, n_hv).astype(kernel_dtype),
            state_in,
            cu_seqlens_arr,
            slot_ids_arr,
            num_requests,
        ]
        template = [
            ("T", kernel_dtype),
            ("StT", state_dtype),
            ("Dk", d_k),
            ("Dv", d_v),
            ("Hk", n_hk),
            ("Hv", n_hv),
        ]
        y_out, state_updates = self._recurrent_prefill_kernel(
            inputs=kernel_inputs,
            template=template,
            grid=(_RECURRENT_SIMDGROUP_WIDTH, d_v, num_requests * n_hv),
            threadgroup=(
                _RECURRENT_SIMDGROUP_WIDTH,
                _RECURRENT_PREFILL_THREADGROUP_DV,
                1,
            ),
            output_shapes=[(total_tokens, n_hv, d_v), (num_requests, n_hv, d_v, d_k)],
            output_dtypes=[kernel_dtype, state_dtype],
        )
        if request.defer_state_scatter:
            request.state_cache.set_pending_recurrent_state(
                request.cache_idx, request.slot_ids, state_updates
            )
        else:
            request.state_cache.write_recurrent_rows(
                request.cache_idx, state_updates, slot_ids_arr
            )
        y_out = y_out.astype(request.output_dtype)
        return y_out
