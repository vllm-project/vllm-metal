# SPDX-License-Identifier: Apache-2.0
"""Linear attention (Gated DeltaNet) with paged Metal state dispatch.

Decomposes the mlx_lm GDN module's forward pass and routes eligible decode or
prefill-containing batches through lazy Metal GDN kernels.  Unsupported shapes
or disabled lazy paths use the eager conv / C++ Metal recurrent fallback path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import compute_g

from vllm_metal.metal import get_ops
from vllm_metal.metal_kernel_backend.gdn_lazy import (
    GDNLazyKernels,
    GDNRecurrentDecodeRequest,
    GDNRecurrentPrefillRequest,
)
from vllm_metal.mlx_backend.gdn_cache import GDNPagedStateCache
from vllm_metal.paged_attention_common import PagedAttentionContext, get_context

_DEFAULT_RECURRENT_DECODE_THREADGROUP_DV = 4
_EXPANDED_RECURRENT_DECODE_THREADGROUP_DV = 8


def is_linear_attention(module: nn.Module) -> bool:
    """Return True if *module* is a linear attention layer (e.g. GatedDeltaNet).

    Checks for ``conv1d`` (present in all known GatedDeltaNet variants) and
    the absence of ``q_proj`` (which would indicate SDPA).
    """
    return hasattr(module, "conv1d") and not hasattr(module, "q_proj")


@dataclass(frozen=True)
class _GDNForwardState:
    x: mx.array
    cu_seqlens: list[int]
    num_requests: int
    total_tokens: int
    slot_ids: list[int]
    num_decode_requests: int


@dataclass(frozen=True)
class _GDNLazyPolicy:
    """Named lazy-kernel policy derived from the GDN module layout.

    The lazy kernels are gated by structural capabilities rather than by model
    name.  Keeping those decisions in one object makes the supported layouts
    explicit:

    - combined QKVZ projection: Qwen3-Next style, cheapest handoff path
    - separate projection with expanded value state: Qwen3.6 style, use the
      conservative fp32/materialized handoff policy
    - separate projection with compact value state: Qwen3.5 style, avoid the
      extra conv-prefill launch but keep lazy recurrent prefill
    """

    combined_qkvz_projection: bool
    expanded_recurrent_value_state: bool

    @classmethod
    def from_module(cls, inner: nn.Module) -> _GDNLazyPolicy:
        return cls(
            combined_qkvz_projection=hasattr(inner, "in_proj_qkvz"),
            expanded_recurrent_value_state=inner.num_v_heads > inner.num_k_heads,
        )

    def should_try_conv_prefill_lazy(self) -> bool:
        return self.combined_qkvz_projection or self.expanded_recurrent_value_state

    def recurrent_decode_threadgroup_dv(self) -> int:
        if self.uses_conservative_recurrent_prefill_policy():
            return _EXPANDED_RECURRENT_DECODE_THREADGROUP_DV
        return _DEFAULT_RECURRENT_DECODE_THREADGROUP_DV

    def should_materialize_prefill_state(self, num_requests: int) -> bool:
        return num_requests > 1 and not self.combined_qkvz_projection

    def recurrent_prefill_compute_dtype(self) -> mx.Dtype | None:
        if self.uses_conservative_recurrent_prefill_policy():
            return mx.float32
        return None

    def should_defer_recurrent_prefill_state(self, num_requests: int) -> bool:
        return (
            num_requests > 1 and not self.uses_conservative_recurrent_prefill_policy()
        )

    def uses_conservative_recurrent_prefill_policy(self) -> bool:
        return not self.combined_qkvz_projection and self.expanded_recurrent_value_state


class GDNPagedAttentionWrapper(nn.Module):
    """Wraps a GDN linear attention module for paged state dispatch.

    The forward pass decomposes the mlx_lm GDN module into:
    1. Projections (in_proj_qkv, z, a, b) — stateless, batched
    2. Conv1d with state management — per-request (stateful)
    3. Q/K/V split + RMS norm + gating — stateless, batched
    4. Recurrent update — lazy Metal decode/prefill fast paths, with fallback
    5. Output norm + projection — stateless, batched

    When no ``PagedAttentionContext`` is active, delegates to the original
    module unchanged.
    """

    def __init__(
        self,
        inner: nn.Module,
        layer_idx: int,
        cache_idx: int,
        state_cache: GDNPagedStateCache,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_gdn_layer_idx", layer_idx)
        object.__setattr__(self, "_gdn_cache_idx", cache_idx)
        object.__setattr__(self, "_gdn_state_cache", state_cache)
        object.__setattr__(self, "_gdn_lazy", GDNLazyKernels.shared())
        object.__setattr__(self, "_gdn_lazy_policy", _GDNLazyPolicy.from_module(inner))

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: nn.Module | None = None,
        position_ids: mx.array | None = None,
        **kwargs: Any,
    ) -> mx.array:
        ctx = get_context()
        if ctx is None:
            # GDN is recurrent — does not use position_ids; drop it.
            return self._inner(x, mask=mask, cache=cache)

        state = self._prepare_gdn_forward_state(x, ctx)
        mixed_qkv, z, a, b = self._project_inputs(state)
        conv_packed = self._run_conv(mixed_qkv, state)
        q, k, v = self._split_and_normalize(conv_packed, state)
        g, beta = self._compute_gates(a, b, state)
        y_flat = self._run_recurrent(q, k, v, g, beta, state)
        return self._project_output(y_flat, z, state)

    def _prepare_gdn_forward_state(
        self, x: mx.array, ctx: PagedAttentionContext
    ) -> _GDNForwardState:
        cu_seqlens = ctx.cu_seqlens
        if cu_seqlens is None or len(cu_seqlens) < 2:
            raise RuntimeError("GDN wrapper requires cu_seqlens in context")

        num_requests = len(cu_seqlens) - 1
        slot_ids = (
            ctx.gdn_slot_mapping
            if ctx.gdn_slot_mapping is not None
            else list(range(num_requests))
        )
        if len(slot_ids) != num_requests:
            raise RuntimeError("GDN wrapper requires one slot per request")
        if len(set(slot_ids)) != len(slot_ids):
            raise RuntimeError("GDN wrapper requires unique slots per request")
        if any(slot < 0 or slot >= self._gdn_state_cache.max_seqs for slot in slot_ids):
            raise RuntimeError("GDN wrapper received out-of-range slot mapping")

        return _GDNForwardState(
            x=x,
            cu_seqlens=cu_seqlens,
            num_requests=num_requests,
            total_tokens=x.shape[1],
            slot_ids=slot_ids,
            num_decode_requests=ctx.num_decode_requests,
        )

    def _project_inputs(
        self, state: _GDNForwardState
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        # === Step 1: Projections (stateless, on full packed input) ===
        inner = self._inner
        total_tokens = state.total_tokens
        x = state.x

        if hasattr(inner, "in_proj_qkvz"):
            # Qwen3-Next style: combined projections
            q_pre, k_pre, v_pre, z, b, a = inner.fix_query_key_value_ordering(
                inner.in_proj_qkvz(x), inner.in_proj_ba(x)
            )
            # z: [1, total_tokens, num_v_heads, head_v_dim]
            # b, a: [1, total_tokens, num_v_heads]
            mixed_qkv = mx.concatenate(
                [
                    q_pre.reshape(1, total_tokens, -1),
                    k_pre.reshape(1, total_tokens, -1),
                    v_pre.reshape(1, total_tokens, -1),
                ],
                axis=-1,
            )
        else:
            # Qwen3.5 style: separate projections
            mixed_qkv = inner.in_proj_qkv(x)  # [1, total_tokens, conv_dim]
            z = inner.in_proj_z(x)  # [1, total_tokens, Hv * Dv]
            z = z.reshape(1, total_tokens, -1, inner.head_v_dim)
            b = inner.in_proj_b(x)  # [1, total_tokens, Hv]
            a = inner.in_proj_a(x)  # [1, total_tokens, Hk]

        return mixed_qkv, z, a, b

    def _run_conv(self, mixed_qkv: mx.array, state: _GDNForwardState) -> mx.array:
        # === Step 2: Conv1d (per-request, needs conv_state) ===
        inner = self._inner
        state_cache = self._gdn_state_cache
        cache_idx = self._gdn_cache_idx
        slot_ids = state.slot_ids

        if state.num_decode_requests == state.num_requests:
            conv_packed = self._gdn_lazy.try_conv_decode(
                mixed_qkv, inner, state_cache, cache_idx, slot_ids
            )
            if conv_packed is not None:
                return conv_packed
        elif self._should_try_conv_prefill_containing_lazy(state):
            conv_packed = self._gdn_lazy.try_conv_prefill(
                mixed_qkv, inner, state_cache, cache_idx, slot_ids, state.cu_seqlens
            )
            if conv_packed is not None:
                if self._should_materialize_prefill_state(state):
                    mx.eval(state_cache.updated_conv_state_array(cache_idx))
                return conv_packed

        state_cache.apply_pending_conv_state(cache_idx)
        conv_outputs = []
        conv_updates = []
        defer_conv_state = self._should_defer_conv_prefill_containing_state(state)
        for req_idx in range(state.num_requests):
            slot = slot_ids[req_idx]
            start = state.cu_seqlens[req_idx]
            end = state.cu_seqlens[req_idx + 1]
            req_qkv = mixed_qkv[:, start:end, :]

            # Load conv state from stable slot.
            conv_state = state_cache.conv_states[cache_idx][slot : slot + 1]
            conv_input = mx.concatenate([conv_state, req_qkv], axis=1)

            # Prefill-containing batches keep compact updates so model_runner
            # avoids submitting the full max_num_seqs conv pool.
            new_conv = conv_input[:, -(inner.conv_kernel_size - 1) :]
            if defer_conv_state:
                conv_updates.append(new_conv)
            else:
                cs = state_cache.conv_states[cache_idx]
                cs[slot : slot + 1] = new_conv
                state_cache.conv_states[cache_idx] = cs

            conv_out = nn.silu(inner.conv1d(conv_input))
            # Take only the output tokens (not the conv state prefix)
            conv_outputs.append(conv_out[:, -(end - start) :, :])

        if defer_conv_state:
            state_cache.set_pending_conv_state(
                cache_idx, slot_ids, mx.concatenate(conv_updates, axis=0)
            )
            if self._should_materialize_prefill_state(state):
                mx.eval(state_cache.updated_conv_state_array(cache_idx))
        elif self._should_materialize_prefill_state(state):
            mx.eval(state_cache.conv_states[cache_idx])

        return mx.concatenate(conv_outputs, axis=1)

    def _split_and_normalize(
        self, conv_packed: mx.array, state: _GDNForwardState
    ) -> tuple[mx.array, mx.array, mx.array]:
        # === Step 3: Split Q/K/V + norm ===
        inner = self._inner
        q, k, v = [
            t.reshape(1, state.total_tokens, h, d)
            for t, h, d in zip(
                mx.split(
                    conv_packed,
                    [inner.key_dim, 2 * inner.key_dim],
                    axis=-1,
                ),
                [inner.num_k_heads, inner.num_k_heads, inner.num_v_heads],
                [inner.head_k_dim, inner.head_k_dim, inner.head_v_dim],
                strict=True,
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        return q, k, v

    def _compute_gates(
        self, a: mx.array, b: mx.array, state: _GDNForwardState
    ) -> tuple[mx.array, mx.array]:
        # === Step 4: Gating (stateless) ===
        # compute_g returns float32; cast to match kernel dispatch dtype.
        g = compute_g(self._inner.A_log, a, self._inner.dt_bias).astype(state.x.dtype)
        beta = mx.sigmoid(b).astype(state.x.dtype)
        return g, beta

    def _run_recurrent(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        g: mx.array,
        beta: mx.array,
        state: _GDNForwardState,
    ) -> mx.array:
        # === Step 5: Batched recurrent update ===
        if state.num_decode_requests == state.num_requests:
            request = GDNRecurrentDecodeRequest(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                state_cache=self._gdn_state_cache,
                cache_idx=self._gdn_cache_idx,
                slot_ids=state.slot_ids,
                output_dtype=state.x.dtype,
                threadgroup_dv=self._recurrent_decode_threadgroup_dv(),
            )
            y_flat = self._gdn_lazy.try_recurrent_decode(request)
            if y_flat is not None:
                return y_flat
        elif self._should_try_recurrent_prefill_containing_lazy(state):
            request = GDNRecurrentPrefillRequest(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                state_cache=self._gdn_state_cache,
                cache_idx=self._gdn_cache_idx,
                slot_ids=state.slot_ids,
                output_dtype=state.x.dtype,
                cu_seqlens=state.cu_seqlens,
                materialize_outputs=self._should_materialize_prefill_state(state),
                compute_dtype=self._recurrent_prefill_compute_dtype(),
                defer_state_scatter=self._should_defer_recurrent_prefill_state(state),
            )
            y_flat = self._gdn_lazy.try_recurrent_prefill(request)
            if y_flat is not None:
                return y_flat
        self._gdn_state_cache.apply_pending_recurrent_state(self._gdn_cache_idx)
        return self._run_recurrent_fallback(q, k, v, g, beta, state)

    def _should_try_recurrent_prefill_containing_lazy(
        self, state: _GDNForwardState
    ) -> bool:
        return (
            self._lazy_kernels_enabled()
            and state.num_decode_requests < state.num_requests
            and state.total_tokens > state.num_requests
        )

    def _lazy_kernels_enabled(self) -> bool:
        return self._gdn_lazy.enabled

    def _should_defer_conv_prefill_containing_state(
        self, state: _GDNForwardState
    ) -> bool:
        return (
            self._lazy_kernels_enabled()
            and state.num_decode_requests < state.num_requests
        )

    def _should_try_conv_prefill_containing_lazy(self, state: _GDNForwardState) -> bool:
        # The prefill conv kernel removes per-request Python dispatch, but
        # compact separate-projection variants are small enough that the extra
        # Metal launch can dominate long prompts.  Keep those on the existing
        # eager conv path while enabling the lazy conv path for combined-QKVZ
        # and expanded-value-state variants where measured handoff cost is
        # larger.
        return (
            self._lazy_kernels_enabled()
            and state.num_decode_requests < state.num_requests
            and self._gdn_lazy_policy.should_try_conv_prefill_lazy()
        )

    def _recurrent_decode_threadgroup_dv(self) -> int:
        # Expanded separate-projection variants have more value-state work per
        # decode step and benefit from grouping more Dv rows per threadgroup.
        # Compact separate-projection and combined-QKVZ variants stay on the
        # original launch shape, which benchmarks better for those layouts.
        return self._gdn_lazy_policy.recurrent_decode_threadgroup_dv()

    def _should_materialize_prefill_state(self, state: _GDNForwardState) -> bool:
        # Separate-projection GDN variants keep more independent state-producing
        # arrays alive across the lazy prefill graph.  Materializing conv and
        # recurrent state at each layer bounds the prefill→decode handoff
        # without penalizing Qwen3-Next's combined QKVZ projection path.
        return self._gdn_lazy_policy.should_materialize_prefill_state(
            state.num_requests
        )

    def _recurrent_prefill_compute_dtype(self) -> mx.Dtype | None:
        return self._gdn_lazy_policy.recurrent_prefill_compute_dtype()

    def _should_defer_recurrent_prefill_state(self, state: _GDNForwardState) -> bool:
        return (
            self._lazy_kernels_enabled()
            and self._gdn_lazy_policy.should_defer_recurrent_prefill_state(
                state.num_requests
            )
        )

    def _run_recurrent_fallback(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        g: mx.array,
        beta: mx.array,
        state: _GDNForwardState,
    ) -> mx.array:
        # C++ Metal fallback path.
        self._gdn_state_cache.apply_pending_recurrent_state(self._gdn_cache_idx)
        inner = self._inner
        total_tokens = state.total_tokens
        n_hk = inner.num_k_heads
        n_hv = inner.num_v_heads
        d_k = inner.head_k_dim
        d_v = inner.head_v_dim

        # Flatten for kernel: remove batch dim.
        # Use float32 for kernel dispatch to avoid float16 overflow in
        # recurrent state accumulation.  Output is cast back after.
        kernel_dtype = mx.float32
        q_flat = mx.contiguous(q.reshape(total_tokens, n_hk, d_k).astype(kernel_dtype))
        k_flat = mx.contiguous(k.reshape(total_tokens, n_hk, d_k).astype(kernel_dtype))
        v_flat = mx.contiguous(v.reshape(total_tokens, n_hv, d_v).astype(kernel_dtype))
        g_flat = mx.contiguous(g.reshape(total_tokens, n_hv).astype(kernel_dtype))
        beta_flat = mx.contiguous(beta.reshape(total_tokens, n_hv).astype(kernel_dtype))

        cu_seqlens_arr = mx.array(state.cu_seqlens, dtype=mx.int32)
        # Stable request → slot mapping from model_runner's allocator.
        slot_mapping = mx.array(state.slot_ids, dtype=mx.int32)

        y_flat = mx.zeros((total_tokens, n_hv, d_v), dtype=kernel_dtype)
        recurrent_pool = self._gdn_state_cache.recurrent_states[self._gdn_cache_idx]

        mx.eval(
            q_flat,
            k_flat,
            v_flat,
            g_flat,
            beta_flat,
            recurrent_pool,
            cu_seqlens_arr,
            slot_mapping,
            y_flat,
        )

        ops = get_ops()
        ops.gdn_linear_attention(
            q_flat,
            k_flat,
            v_flat,
            g_flat,
            beta_flat,
            recurrent_pool,
            cu_seqlens_arr,
            slot_mapping,
            y_flat,
            n_hk,
            n_hv,
            d_k,
            d_v,
        )
        mx.eval(y_flat, recurrent_pool)
        return y_flat.astype(state.x.dtype)

    def _project_output(
        self, y_flat: mx.array, z: mx.array, state: _GDNForwardState
    ) -> mx.array:
        # === Step 6: Output norm + projection ===
        inner = self._inner
        out = y_flat.reshape(1, state.total_tokens, inner.num_v_heads, inner.head_v_dim)
        out = inner.norm(out, z)
        return inner.out_proj(out.reshape(1, state.total_tokens, -1))
