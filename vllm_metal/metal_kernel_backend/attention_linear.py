# SPDX-License-Identifier: Apache-2.0
"""Linear attention (Gated DeltaNet) with C++ Metal kernel for paged state.

Decomposes the mlx_lm GDN module's forward pass and replaces the recurrent
update step with a C++ nanobind Metal kernel that reads/writes state in-place
from a managed pool via slot_mapping.

Conv1d remains per-request (stateful), but the expensive recurrent step is
dispatched as a single batched Metal kernel call across all requests.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import compute_g

from vllm_metal.metal import get_ops
from vllm_metal.mlx_backend.gdn_cache import GDNPagedStateCache
from vllm_metal.paged_attention_common import get_context


def is_linear_attention(module: nn.Module) -> bool:
    """Return True if *module* is a linear attention layer (e.g. GatedDeltaNet).

    Checks for ``conv1d`` (present in all known GatedDeltaNet variants) and
    the absence of ``q_proj`` (which would indicate SDPA).
    """
    return hasattr(module, "conv1d") and not hasattr(module, "q_proj")


class GDNPagedAttentionWrapper(nn.Module):
    """Wraps a GDN linear attention module with C++ Metal kernel dispatch.

    The forward pass decomposes the mlx_lm GDN module into:
    1. Projections (in_proj_qkv, z, a, b) — stateless, batched
    2. Conv1d with state management — per-request (stateful)
    3. Q/K/V split + RMS norm + gating — stateless, batched
    4. Recurrent update — C++ Metal kernel, batched, in-place state pool
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

        inner = self._inner
        cache_idx: int = self._gdn_cache_idx
        state_cache: GDNPagedStateCache = self._gdn_state_cache

        cu_seqlens = ctx.cu_seqlens
        if cu_seqlens is None or len(cu_seqlens) < 2:
            raise RuntimeError("GDN wrapper requires cu_seqlens in context")

        num_requests = len(cu_seqlens) - 1
        total_tokens = x.shape[1]

        # === Step 1: Projections (stateless, on full packed input) ===
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

        # === Step 2: Conv1d (per-request, needs conv_state) ===
        # Use stable slot mapping for state pool access.
        slot_ids = (
            ctx.gdn_slot_mapping
            if ctx.gdn_slot_mapping is not None
            else list(range(num_requests))
        )
        conv_outputs = []
        for req_idx in range(num_requests):
            slot = slot_ids[req_idx]
            start = cu_seqlens[req_idx]
            end = cu_seqlens[req_idx + 1]
            req_qkv = mixed_qkv[:, start:end, :]

            # Load conv state from stable slot
            conv_state = state_cache.conv_states[cache_idx][slot : slot + 1]
            conv_input = mx.concatenate([conv_state, req_qkv], axis=1)

            # Save updated conv state back to stable slot
            new_conv = conv_input[:, -(inner.conv_kernel_size - 1) :]
            cs = state_cache.conv_states[cache_idx]
            cs[slot : slot + 1] = new_conv
            state_cache.conv_states[cache_idx] = cs

            conv_out = nn.silu(inner.conv1d(conv_input))
            # Take only the output tokens (not the conv state prefix)
            conv_outputs.append(conv_out[:, -(end - start) :, :])

        conv_packed = mx.concatenate(conv_outputs, axis=1)

        # === Step 3: Split Q/K/V + norm ===
        q, k, v = [
            t.reshape(1, total_tokens, h, d)
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

        # === Step 4: Gating (stateless) ===
        # compute_g returns float32; cast to match kernel dispatch dtype.
        g = compute_g(inner.A_log, a, inner.dt_bias).astype(x.dtype)
        beta = mx.sigmoid(b).astype(x.dtype)

        # === Step 5: C++ Metal kernel — batched recurrent update ===
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

        cu_seqlens_arr = mx.array(cu_seqlens, dtype=mx.int32)
        # Stable request → slot mapping from model_runner's allocator.
        if ctx.gdn_slot_mapping is not None:
            slot_mapping = mx.array(ctx.gdn_slot_mapping, dtype=mx.int32)
        else:
            slot_mapping = mx.arange(num_requests, dtype=mx.int32)

        y_flat = mx.zeros((total_tokens, n_hv, d_v), dtype=kernel_dtype)
        recurrent_pool = state_cache.recurrent_states[cache_idx]

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
        y_flat = y_flat.astype(x.dtype)

        # === Step 6: Output norm + projection ===
        out = y_flat.reshape(1, total_tokens, n_hv, d_v)
        out = inner.norm(out, z)
        return inner.out_proj(out.reshape(1, total_tokens, -1))
