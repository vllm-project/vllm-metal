# SPDX-License-Identifier: Apache-2.0
"""Lazy Gated DeltaNet decode kernels for MLX/Metal execution."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, ClassVar

import mlx.core as mx

import vllm_metal.envs as envs
from vllm_metal.metal import _read_v2_metal_source
from vllm_metal.mlx_backend.gdn_cache import GDNPagedStateCache

_GDN_CONV1D_V2_SOURCE = _read_v2_metal_source("gdn_conv1d_silu_decode.metal")
_GDN_RECURRENT_V2_SOURCE = _read_v2_metal_source("gdn_recurrent_decode.metal")


@dataclass(frozen=True)
class GDNRecurrentDecodeRequest:
    """Inputs for one lazy GDN recurrent decode attempt."""

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


class GDNLazyDecodeKernels:
    """Owner for lazy GDN decode fast-path policy and kernels.

    The fast path uses ``mx.fast.metal_kernel`` source snippets from
    ``kernels_v2``.  Unlike the legacy recurrent C++ path, these kernels
    materialize compact conv/recurrent state updates.  Updates are scattered
    back into the stable cache pool so decode-only state work stays
    proportional to active request count;
    callers fall back to the legacy path when a request shape is ineligible.
    """

    _shared: ClassVar[GDNLazyDecodeKernels | None] = None
    _shared_enabled: ClassVar[bool | None] = None
    _shared_lock: ClassVar[Lock] = Lock()

    def __init__(
        self,
        *,
        enabled: bool,
        conv_kernel: Any | None = None,
        recurrent_kernel: Any | None = None,
    ) -> None:
        self._enabled = enabled
        if not enabled:
            self._conv_kernel = None
            self._recurrent_kernel = None
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
        self._recurrent_kernel = (
            recurrent_kernel
            if recurrent_kernel is not None
            else self._make_kernel(
                "gdn_recurrent_v2",
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
                _GDN_RECURRENT_V2_SOURCE,
            )
        )

    @classmethod
    def from_env(cls) -> GDNLazyDecodeKernels:
        return cls(enabled=envs.VLLM_METAL_GDN_LAZY_DECODE)

    @classmethod
    def shared(cls) -> GDNLazyDecodeKernels:
        """Get the process-level lazy GDN decode kernel owner.

        The env kill switch is read when the shared owner is acquired;
        existing wrappers keep the owner they stored at construction time.
        """
        with cls._shared_lock:
            enabled = envs.VLLM_METAL_GDN_LAZY_DECODE
            if cls._shared is None or cls._shared_enabled != enabled:
                cls._shared = cls(enabled=enabled)
                cls._shared_enabled = enabled
            return cls._shared

    @classmethod
    def reset_shared_for_tests(cls) -> None:
        """Reset the shared lazy GDN decode kernel owner for tests."""
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
        try:
            if not mx.metal.is_available():
                return None
        except AttributeError:
            return None
        return mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=source,
        )

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
        conv_state_in = state_cache.conv_states[cache_idx]
        weight = inner.conv1d.weight

        mixed_qkv_2d = mixed_qkv.reshape(num_requests, conv_dim)
        slot_ids_arr = mx.array(slot_ids, dtype=mx.int32)

        grid_size = num_requests * conv_dim
        tg_size = min(256, grid_size)
        state_updates_shape = (num_requests, kernel_size - 1, conv_dim)

        kernel_inputs = [
            mixed_qkv_2d,
            conv_state_in,
            weight,
            slot_ids_arr,
            num_requests,
        ]
        template = [
            ("T", mixed_qkv.dtype),
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
            output_dtypes=[mixed_qkv.dtype, conv_state_in.dtype],
        )
        conv_state_in[slot_ids_arr] = conv_state_updates
        state_cache.conv_states[cache_idx] = conv_state_in
        return conv_silu_out.reshape(1, total_tokens, conv_dim)

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
        # The Metal source uses one 32-lane SIMD group across Dk and a
        # fixed-size per-thread register array, matching the original C++
        # kernel's practical Dk <= 256 envelope. Fall back for unusual head
        # dimensions rather than silently dropping remainder channels.
        recurrent_shape_supported = d_k % 32 == 0 and d_k <= 256
        if not (
            self._enabled
            and self._recurrent_kernel is not None
            and total_tokens == num_requests
            and recurrent_shape_supported
        ):
            return None

        state_in = request.state_cache.recurrent_states[request.cache_idx]
        slot_ids_arr = mx.array(request.slot_ids, dtype=mx.int32)

        kernel_inputs = [
            request.q.reshape(total_tokens, n_hk, d_k),
            request.k.reshape(total_tokens, n_hk, d_k),
            request.v.reshape(total_tokens, n_hv, d_v),
            request.g.reshape(total_tokens, n_hv),
            request.beta.reshape(total_tokens, n_hv),
            state_in,
            slot_ids_arr,
            num_requests,
        ]
        template = [
            ("T", request.output_dtype),
            ("StT", mx.float32),
            ("Dk", d_k),
            ("Dv", d_v),
            ("Hk", n_hk),
            ("Hv", n_hv),
        ]
        y_out, state_updates = self._recurrent_kernel(
            inputs=kernel_inputs,
            template=template,
            grid=(32, d_v, num_requests * n_hv),
            threadgroup=(32, 4, 1),
            output_shapes=[(total_tokens, n_hv, d_v), (num_requests, n_hv, d_v, d_k)],
            output_dtypes=[request.output_dtype, mx.float32],
        )
        state_in[slot_ids_arr] = state_updates
        request.state_cache.recurrent_states[request.cache_idx] = state_in
        return y_out
