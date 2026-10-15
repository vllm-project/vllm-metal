# SPDX-License-Identifier: Apache-2.0
"""Preserve model precision across the paged GDN kernel boundaries."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.gated_delta import compute_g, normalize_qk

from tests.stub_runner import make_state_cache
from vllm_metal.attention.impls.gdn_lazy import (
    GDNLazyKernels,
    GDNRecurrentDecodeRequest,
    GDNRecurrentPrefillRequest,
)
from vllm_metal.attention.impls.linear import GDNPagedAttentionWrapper, _GDNForwardState


def _cache(dtype: mx.Dtype, channels: int = 64):
    return make_state_cache(
        num_layers=1,
        max_seqs=4,
        conv_kernel_dim=4,
        conv_dim=channels,
        num_v_heads=1,
        value_head_dim=4,
        key_head_dim=32,
        dtype=dtype,
        recurrent_dtype=mx.float32,
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("prefill", [False, True])
def test_lazy_conv_matches_native_activation_rounding(dtype, prefill):
    mx.random.seed(314)
    channels = 64
    cache = _cache(dtype, channels)
    conv = nn.Conv1d(channels, channels, 4, groups=channels, bias=False)
    conv.weight = mx.random.normal(conv.weight.shape).astype(dtype)
    inner = SimpleNamespace(conv_kernel_size=4, conv1d=conv)
    initial = mx.random.normal(cache.conv_states[0].shape).astype(dtype)
    cache.conv_states[0] = mx.array(initial)
    slots = [3, 1]
    cu = [0, 1, 38] if prefill else [0, 1, 2]
    x = mx.random.normal((1, cu[-1], channels)).astype(dtype)
    expected, states = [], []
    for i, slot in enumerate(slots):
        joined = mx.concatenate([initial[slot : slot + 1], x[:, cu[i] : cu[i + 1]]], 1)
        expected.append(nn.silu(conv(joined)))
        states.append(joined[:, -3:])
    reference = mx.concatenate(expected, 1)
    kernels = GDNLazyKernels(enabled=True)
    if prefill:
        actual = kernels.try_conv_prefill(x, inner, cache, 0, slots, cu)
        updated = cache.pending_conv_state(0, slots)
    else:
        actual = kernels.try_conv_decode(x, inner, cache, 0, slots)
        updated = cache.conv_states[0][slots]
    assert actual is not None and updated is not None
    assert actual.dtype == dtype
    np.testing.assert_array_equal(
        np.array(actual.astype(mx.float32)), np.array(reference.astype(mx.float32))
    )
    np.testing.assert_array_equal(
        np.array(updated.astype(mx.float32)),
        np.array(mx.concatenate(states, 0).astype(mx.float32)),
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_decay_gate_retains_fp32_state_precision(dtype):
    inner = SimpleNamespace(
        num_k_heads=1,
        num_v_heads=1,
        A_log=mx.zeros((1,), dtype=mx.float32),
        dt_bias=mx.zeros((1,), dtype=mx.float32),
    )
    wrapper = GDNPagedAttentionWrapper(inner, 0, 0, _cache(dtype))
    state = _GDNForwardState(mx.zeros((1, 1, 64), dtype=dtype), [0, 1], 1, 1, [0], 1)
    a = mx.full((1, 1, 1), -9.25, dtype=dtype)
    b = mx.zeros_like(a)
    g, beta = wrapper._compute_gates(a, b, state)
    expected = compute_g(inner.A_log, a, inner.dt_bias)
    assert g.dtype == mx.float32 and beta.dtype == dtype
    assert g.item() < 1.0
    assert g.astype(dtype).item() == 1.0  # The old cast erased this decay.
    np.testing.assert_array_equal(np.array(g), np.array(expected))


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("route", ["decode", "prefill", "prefill_fp32"])
def test_recurrent_decay_is_not_rounded_to_one(dtype, route):
    tokens = 1 if route == "decode" else 256
    cache = _cache(dtype)
    cache.recurrent_states[0] = mx.ones_like(cache.recurrent_states[0])
    q = mx.zeros((1, tokens, 1, 32), dtype=dtype)
    q[..., 0] = 1
    g = mx.full((1, tokens, 1), 0.9999, dtype=mx.float32)
    common = {
        "q": q,
        "k": mx.zeros_like(q),
        "v": mx.zeros((1, tokens, 1, 4), dtype=dtype),
        "g": g,
        "beta": mx.zeros((1, tokens, 1), dtype=dtype),
        "state_cache": cache,
        "cache_idx": 0,
        "slot_ids": [2],
        "output_dtype": dtype,
    }
    kernels = GDNLazyKernels(enabled=True)
    if route == "decode":
        output = kernels.try_recurrent_decode(GDNRecurrentDecodeRequest(**common))
        updated = cache.pending_recurrent_state(0, [2])
    else:
        output = kernels.try_recurrent_prefill(
            GDNRecurrentPrefillRequest(
                **common,
                cu_seqlens=[0, tokens],
                compute_dtype=mx.float32 if route == "prefill_fp32" else None,
            )
        )
        updated = cache.recurrent_states[0][2:3]
    assert output is not None and updated is not None
    assert output.dtype == dtype
    expected = np.float32(1)
    for _ in range(tokens):
        expected *= np.float32(0.9999)
    np.testing.assert_allclose(np.array(updated), expected, atol=1e-7, rtol=1e-6)
    assert float(mx.max(updated).item()) < 1.0


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_native_gdn_reference_uses_the_same_normalization(dtype):
    mx.random.seed(23)
    q = (mx.random.normal((1, 3, 1, 32)) * 1e-3).astype(dtype)
    k = (mx.random.normal(q.shape) * 1e-3).astype(dtype)
    expected_q, expected_k = normalize_qk(q, k, inv_scale=32**-0.5, eps=1e-6)
    inner = SimpleNamespace(
        num_k_heads=1, num_v_heads=1, key_dim=32, head_k_dim=32, head_v_dim=4
    )
    wrapper = GDNPagedAttentionWrapper(inner, 0, 0, _cache(dtype, 68))
    state = _GDNForwardState(mx.zeros((1, 3, 68), dtype=dtype), [0, 3], 1, 3, [0], 0)
    packed = mx.concatenate(
        [q.reshape(1, 3, 32), k.reshape(1, 3, 32), mx.zeros((1, 3, 4), dtype=dtype)],
        axis=-1,
    )
    actual_q, actual_k, _ = wrapper._split_and_normalize(packed, state)
    np.testing.assert_array_equal(
        np.array(actual_q.astype(mx.float32)), np.array(expected_q.astype(mx.float32))
    )
    np.testing.assert_array_equal(
        np.array(actual_k.astype(mx.float32)), np.array(expected_k.astype(mx.float32))
    )
