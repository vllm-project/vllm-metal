# SPDX-License-Identifier: Apache-2.0
"""Linear attention (Gated DeltaNet) with mx.fast.metal_kernel for paged state.

Decomposes the mlx_lm GDN module's forward pass and replaces the recurrent
update step with an mx.fast.metal_kernel that operates on gathered state
slices from a paged pool via slot_mapping.

The kernel participates in MLX's lazy evaluation graph — no explicit mx.eval
barriers are needed in the forward path.  State is gathered from the pool
before the kernel and scattered back afterward, both as lazy MLX ops.

Conv1d remains per-request (stateful), but the expensive recurrent step is
dispatched as a single batched Metal kernel call across all requests.
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import compute_g

from vllm_metal.metal import _read_v2_metal_source, get_ops
from vllm_metal.mlx_backend.gdn_cache import GDNPagedStateCache
from vllm_metal.paged_attention_common import get_context

# ---------------------------------------------------------------------------
# Phase 2: mx.fast.metal_kernel conv1d + SiLU decode (lazy graph integration)
# ---------------------------------------------------------------------------

_GDN_CONV1D_V2_SOURCE = _read_v2_metal_source("gdn_conv1d_silu_decode.metal")


def _make_conv1d_silu_decode_kernel():
    """Build mx.fast.metal_kernel for batched depthwise conv1d + SiLU decode."""
    try:
        if not mx.metal.is_available():
            return None
    except AttributeError:
        return None
    return mx.fast.metal_kernel(
        name="gdn_conv1d_silu_decode_v2",
        input_names=[
            "input",
            "conv_state_in",
            "weights",
            "slot_mapping",
            "num_requests",
        ],
        output_names=["output", "conv_state_out"],
        source=_GDN_CONV1D_V2_SOURCE,
    )


_conv1d_silu_decode_kernel = _make_conv1d_silu_decode_kernel()
_conv_kernel_mode = os.environ.get("VLLM_GDN_CONV_KERNEL", "2")

# ---------------------------------------------------------------------------
# Recurrent kernel modes: "1" = C++ eager, "2" = lazy (default)
# ---------------------------------------------------------------------------
_recurrent_kernel_mode = os.environ.get("VLLM_GDN_RECURRENT_KERNEL", "2")

# Phase 2.5: mx.fast.metal_kernel recurrent update (lazy graph integration)
# Ported from mlx_lm/models/gated_delta.py with slot_mapping for paged state.
_GDN_RECURRENT_V2_SOURCE = _read_v2_metal_source("gdn_recurrent_decode.metal")


def _make_recurrent_v2_kernel():
    """Build mx.fast.metal_kernel for GDN recurrent update (lazy graph)."""
    try:
        if not mx.metal.is_available():
            return None
    except AttributeError:
        return None
    return mx.fast.metal_kernel(
        name="gdn_recurrent_v2",
        input_names=[
            "q",
            "k",
            "v",
            "g",
            "beta",
            "state_in",
            "slot_mapping",
            "num_requests",
        ],
        output_names=["y", "state_out"],
        source=_GDN_RECURRENT_V2_SOURCE,
    )


_recurrent_v2_kernel = (
    _make_recurrent_v2_kernel() if _recurrent_kernel_mode == "2" else None
)


def is_linear_attention(module: nn.Module) -> bool:
    """Return True if *module* is a linear attention layer (e.g. GatedDeltaNet).

    Checks for ``conv1d`` (present in all known GatedDeltaNet variants) and
    the absence of ``q_proj`` (which would indicate SDPA).
    """
    return hasattr(module, "conv1d") and not hasattr(module, "q_proj")


class GDNPagedAttentionWrapper(nn.Module):
    """Wraps a GDN linear attention module with lazy Metal kernel dispatch.

    The forward pass decomposes the mlx_lm GDN module into:
    1. Projections (in_proj_qkv, z, a, b) — stateless, batched
    2. Conv1d with state management — per-request (stateful)
    3. Q/K/V split + RMS norm + gating — stateless, batched
    4. Recurrent update — lazy Metal kernel, batched functional state
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

        # === Step 2: Conv1d (batched for decode, per-request for prefill) ===
        # Use stable slot mapping for state pool access.
        slot_ids = (
            ctx.gdn_slot_mapping
            if ctx.gdn_slot_mapping is not None
            else list(range(num_requests))
        )

        is_decode = total_tokens == num_requests
        use_v2 = (
            _conv_kernel_mode == "2"
            and _conv1d_silu_decode_kernel is not None
            and is_decode
        )
        if use_v2:
            # --- Phase 2: mx.fast.metal_kernel (lazy, no sync) ---
            conv_dim = state_cache.conv_dim
            kernel_size = inner.conv_kernel_size
            max_seqs = state_cache.max_seqs
            conv_state_in = state_cache.conv_states[cache_idx]
            weight = inner.conv1d.weight

            mixed_qkv_2d = mixed_qkv.reshape(num_requests, conv_dim)
            slot_ids_arr = mx.array(slot_ids, dtype=mx.int32)

            grid_size = max_seqs * conv_dim
            tg_size = min(256, grid_size)
            state_shape = (max_seqs, kernel_size - 1, conv_dim)

            conv_silu_out, conv_state_out = _conv1d_silu_decode_kernel(
                inputs=[
                    mixed_qkv_2d,
                    conv_state_in,
                    weight,
                    slot_ids_arr,
                    num_requests,
                ],
                template=[
                    ("T", mixed_qkv.dtype),
                    ("CONV_DIM", conv_dim),
                    ("KERNEL_SIZE", kernel_size),
                    ("MAX_SEQS", max_seqs),
                ],
                grid=(grid_size, 1, 1),
                threadgroup=(tg_size, 1, 1),
                output_shapes=[(num_requests, conv_dim), state_shape],
                output_dtypes=[mixed_qkv.dtype, conv_state_in.dtype],
            )
            state_cache.conv_states[cache_idx] = conv_state_out
            conv_packed = conv_silu_out.reshape(1, total_tokens, conv_dim)

        else:
            # --- Fallback: batched for decode, per-request for prefill ---
            all_decode = all(
                cu_seqlens[i + 1] - cu_seqlens[i] == 1 for i in range(num_requests)
            )

            if all_decode and num_requests > 1:
                # Batched decode: gather states → concat → single conv1d → scatter
                slot_mapping_conv = mx.array(slot_ids, dtype=mx.int32)
                gathered_states = state_cache.conv_states[cache_idx][slot_mapping_conv]
                qkv_batch = mixed_qkv[0, :, :].reshape(num_requests, 1, -1)
                conv_input = mx.concatenate([gathered_states, qkv_batch], axis=1)
                new_states = conv_input[:, -(inner.conv_kernel_size - 1) :]
                pool = state_cache.conv_states[cache_idx]
                pool[slot_mapping_conv] = new_states
                state_cache.conv_states[cache_idx] = pool
                conv_out = nn.silu(inner.conv1d(conv_input))
                conv_packed = conv_out[:, -1:, :].reshape(1, num_requests, -1)
            else:
                # Per-request loop (prefill with variable lengths, or single request)
                conv_outputs = []
                for req_idx in range(num_requests):
                    slot = slot_ids[req_idx]
                    start = cu_seqlens[req_idx]
                    end = cu_seqlens[req_idx + 1]
                    req_qkv = mixed_qkv[:, start:end, :]

                    conv_state = state_cache.conv_states[cache_idx][slot : slot + 1]
                    conv_input = mx.concatenate([conv_state, req_qkv], axis=1)

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

        # === Step 5: Recurrent update — V2 lazy / noop / V1 eager ===
        n_hk = inner.num_k_heads
        n_hv = inner.num_v_heads
        d_k = inner.head_k_dim
        d_v = inner.head_v_dim

        is_decode = total_tokens == num_requests
        use_recurrent_v2 = (
            _recurrent_kernel_mode == "2"
            and _recurrent_v2_kernel is not None
            and is_decode
        )
        # Stable request → slot mapping from model_runner's allocator.
        if ctx.gdn_slot_mapping is not None:
            slot_ids = ctx.gdn_slot_mapping
        else:
            slot_ids = list(range(num_requests))

        if use_recurrent_v2:
            # --- Phase 2.5: mx.fast.metal_kernel lazy dispatch ---
            # No mx.eval — stays in lazy graph for downstream fusion.
            state_in = state_cache.recurrent_states[cache_idx]
            max_seqs = state_cache.max_seqs

            slot_ids_arr = mx.array(slot_ids, dtype=mx.int32)

            y_out, state_out = _recurrent_v2_kernel(
                inputs=[
                    q.reshape(total_tokens, n_hk, d_k),
                    k.reshape(total_tokens, n_hk, d_k),
                    v.reshape(total_tokens, n_hv, d_v),
                    g.reshape(total_tokens, n_hv),
                    beta.reshape(total_tokens, n_hv),
                    state_in,
                    slot_ids_arr,
                    num_requests,
                ],
                template=[
                    ("T", x.dtype),
                    ("StT", mx.float32),
                    ("Dk", d_k),
                    ("Dv", d_v),
                    ("Hk", n_hk),
                    ("Hv", n_hv),
                    ("MAX_SEQS", max_seqs),
                ],
                grid=(32, d_v, max_seqs * n_hv),
                threadgroup=(32, 4, 1),
                output_shapes=[
                    (total_tokens, n_hv, d_v),
                    state_in.shape,
                ],
                output_dtypes=[x.dtype, mx.float32],
            )
            state_cache.recurrent_states[cache_idx] = state_out
            y_flat = y_out

        else:
            # --- V1: C++ eager dispatch (original path) ---
            kernel_dtype = mx.float32
            q_flat = mx.contiguous(
                q.reshape(total_tokens, n_hk, d_k).astype(kernel_dtype)
            )
            k_flat = mx.contiguous(
                k.reshape(total_tokens, n_hk, d_k).astype(kernel_dtype)
            )
            v_flat = mx.contiguous(
                v.reshape(total_tokens, n_hv, d_v).astype(kernel_dtype)
            )
            g_flat = mx.contiguous(g.reshape(total_tokens, n_hv).astype(kernel_dtype))
            beta_flat = mx.contiguous(
                beta.reshape(total_tokens, n_hv).astype(kernel_dtype)
            )

            cu_seqlens_arr = mx.array(cu_seqlens, dtype=mx.int32)
            slot_mapping = mx.array(slot_ids, dtype=mx.int32)

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
