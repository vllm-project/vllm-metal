# SPDX-License-Identifier: Apache-2.0
"""Deterministic golden tests for the fused GDN linear attention kernel.

Verifies that the vllm-metal fused kernel produces identical output to
mlx_lm's Metal kernel across decode (T=1) and prefill (T>1) configs.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.gated_delta import compute_g, gated_delta_kernel

from vllm_metal.metal.linear_attention import fused_gdn_decode

# Qwen3.5 shared dimensions
DK = 128
DV = 128
HK = 16

# Absolute tolerance for fp16 gating-order differences.
# Empirical max_abs is ~0.000031 for output and ~0.000061 for state.
# Set tolerance at 10x empirical to allow for hardware variance while
# still catching meaningful drift.
ATOL_Y = 0.001
ATOL_S = 0.001


def _make_inputs(B, T, Hv, dtype=mx.float16):  # noqa: N803
    mx.random.seed(42)
    sc = 0.1
    q = (mx.random.normal((B, T, HK, DK)) * sc).astype(dtype)
    k = (mx.random.normal((B, T, HK, DK)) * sc).astype(dtype)
    v = (mx.random.normal((B, T, Hv, DV)) * sc).astype(dtype)
    a = (mx.random.normal((B, T, Hv)) * sc).astype(dtype)
    b = (mx.random.normal((B, T, Hv)) * sc).astype(dtype)
    A_log = (mx.random.normal((Hv,)) * sc).astype(dtype)  # noqa: N806
    dt_bias = (mx.random.normal((Hv,)) * sc).astype(dtype)
    state = mx.zeros((B, Hv, DV, DK), dtype=dtype)
    mx.eval(q, k, v, a, b, A_log, dt_bias, state)
    return q, k, v, a, b, A_log, dt_bias, state


def _run_reference(q, k, v, a, b, A_log, dt_bias, state):  # noqa: N803
    """mlx_lm Metal kernel with pre-computed gating."""
    g = compute_g(A_log, a, dt_bias)
    beta = mx.sigmoid(b)
    mx.eval(g, beta)
    state_copy = mx.array(state)
    mx.eval(state_copy)
    y, s = gated_delta_kernel(q, k, v, g, beta, state_copy)
    mx.eval(y, s)
    return y, s


def _run_fused(q, k, v, a, b, A_log, dt_bias, state):  # noqa: N803
    """vllm-metal fused kernel."""
    y, s = fused_gdn_decode(q, k, v, a, b, A_log, dt_bias, state)
    mx.eval(y, s)
    return y, s


# --- Decode (T=1) ---


@pytest.mark.parametrize("B,Hv", [(1, 32), (1, 48), (4, 32), (8, 48)])
def test_decode_matches_reference(B, Hv):  # noqa: N803
    inputs = _make_inputs(B, T=1, Hv=Hv)
    y_ref, s_ref = _run_reference(*inputs)
    y_fused, s_fused = _run_fused(*inputs)

    y_diff = mx.abs(y_ref.astype(mx.float32) - y_fused.astype(mx.float32)).max().item()
    s_diff = mx.abs(s_ref.astype(mx.float32) - s_fused.astype(mx.float32)).max().item()

    assert y_diff < ATOL_Y, f"y max_abs_diff={y_diff}"
    assert s_diff < ATOL_S, f"state max_abs_diff={s_diff}"


# --- Prefill (T>1) ---


@pytest.mark.parametrize("T", [4, 16])
def test_prefill_matches_reference(T):  # noqa: N803
    inputs = _make_inputs(B=1, T=T, Hv=32)
    y_ref, s_ref = _run_reference(*inputs)
    y_fused, s_fused = _run_fused(*inputs)

    y_diff = mx.abs(y_ref.astype(mx.float32) - y_fused.astype(mx.float32)).max().item()
    s_diff = mx.abs(s_ref.astype(mx.float32) - s_fused.astype(mx.float32)).max().item()

    assert y_diff < ATOL_Y, f"y max_abs_diff={y_diff}"
    assert s_diff < ATOL_S, f"state max_abs_diff={s_diff}"


# --- Output shape ---


def test_output_shapes():
    B, T, Hv = 2, 8, 32  # noqa: N806
    inputs = _make_inputs(B, T, Hv)
    y, s = _run_fused(*inputs)

    assert y.shape == (B, T, Hv, DV)
    assert s.shape == (B, Hv, DV, DK)
