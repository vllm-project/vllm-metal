# SPDX-License-Identifier: Apache-2.0
"""Fused GDN linear attention kernel for Metal.

Fuses gating computation (A_log, a, b, dt_bias → g, beta) with the
recurrent state update, eliminating separate kernel dispatches.

Uses ``mx.fast.metal_kernel`` for rapid prototyping.  Once validated,
can be migrated to the nanobind C++ dispatch path (paged_ops.cpp) for
tighter integration with the paged attention pipeline.
"""

from __future__ import annotations

import mlx.core as mx


def _make_fused_gdn_kernel():
    """Build the fused GDN decode Metal kernel via mx.fast.metal_kernel."""
    if not mx.metal.is_available():
        return None

    source = """
        const int T = T_val;
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(i_state[s_idx]);
        }

        // Per-head constants
        float A_log_val = static_cast<float>(A_log[hv_idx]);
        float dt_bias_val = static_cast<float>(dt_bias[hv_idx]);

        // a, b: [B, T, Hv]
        auto a_ = a + b_idx * T * Hv;
        auto b_ = b + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
            // Fused gating: g = exp(-exp(A_log) * softplus(a + dt_bias))
            float a_val = static_cast<float>(a_[hv_idx]);
            float x = a_val + dt_bias_val;
            float sp = (x > 20.0f) ? x : log(1.0f + exp(x));
            float g_val = exp(-exp(A_log_val) * sp);

            // beta = sigmoid(b)
            float b_val = static_cast<float>(b_[hv_idx]);
            float beta_val = 1.0f / (1.0f + exp(-b_val));

            // Recurrence
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] * g_val;
                kv_mem += state[i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_val;

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] + k_[s_idx] * delta;
                out += state[i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
                y[dv_idx] = static_cast<InT>(out);
            }

            q_ += Hk * Dk;
            k_ += Hk * Dk;
            v_ += Hv * Dv;
            y += Hv * Dv;
            a_ += Hv;
            b_ += Hv;
        }

        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            o_state[s_idx] = static_cast<InT>(state[i]);
        }
    """

    return mx.fast.metal_kernel(
        name="fused_gdn_decode",
        input_names=["q", "k", "v", "a", "b", "A_log", "dt_bias", "state_in", "T_val"],
        output_names=["y", "state_out"],
        source=source,
    )


_fused_kernel = _make_fused_gdn_kernel()


def fused_gdn_decode(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,  # noqa: N803
    dt_bias: mx.array,
    state: mx.array,
) -> tuple[mx.array, mx.array]:
    """Fused GDN decode: gating + recurrence in one Metal dispatch.

    Args:
        q: [B, T, Hk, Dk] queries
        k: [B, T, Hk, Dk] keys
        v: [B, T, Hv, Dv] values
        a: [B, T, Hv] decay parameter
        b: [B, T, Hv] gating parameter
        A_log: [Hv] log-space decay base
        dt_bias: [Hv] bias for decay computation
        state: [B, Hv, Dv, Dk] recurrent state

    Returns:
        (output [B, T, Hv, Dv], new_state [B, Hv, Dv, Dk])
    """
    B, T, Hk, Dk = k.shape  # noqa: N806
    Hv, Dv = v.shape[2], v.shape[3]  # noqa: N806

    return _fused_kernel(
        inputs=[q, k, v, a, b, A_log, dt_bias, state, T],
        template=[
            ("InT", q.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, q.dtype],
    )
