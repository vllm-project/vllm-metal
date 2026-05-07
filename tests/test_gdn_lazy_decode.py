# SPDX-License-Identifier: Apache-2.0
"""Focused regression tests for lazy GDN decode dispatch."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import vllm_metal.metal_kernel_backend.attention_linear as attention_linear
from vllm_metal.metal import get_ops
from vllm_metal.metal_kernel_backend.attention_linear import GDNPagedAttentionWrapper
from vllm_metal.metal_kernel_backend.gdn_lazy_decode import (
    GDNLazyDecodeKernels,
    GDNRecurrentDecodeRequest,
)
from vllm_metal.mlx_backend.gdn_cache import GDNPagedStateCache
from vllm_metal.paged_attention_common import (
    PagedAttentionContext,
    clear_context,
    set_context,
)


@pytest.fixture(autouse=True)
def _reset_lazy_decode_kernels() -> None:
    GDNLazyDecodeKernels.reset_shared_for_tests()
    yield
    GDNLazyDecodeKernels.reset_shared_for_tests()


class _DepthwiseConv1D:
    def __init__(self, weight: mx.array) -> None:
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        kernel_size = self.weight.shape[1]
        outputs = []
        for pos in range(x.shape[1] - kernel_size + 1):
            window = x[:, pos : pos + kernel_size, :]
            outputs.append(mx.sum(window * self.weight.T[None, :, :], axis=1))
        return mx.stack(outputs, axis=1)


class _LastTokenConv1D:
    def __init__(self, conv_dim: int) -> None:
        self.weight = mx.ones((conv_dim, 1), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x[:, -1:, :]


class _TinyGDNInner:
    num_k_heads = 1
    num_v_heads = 1
    head_k_dim = 32
    head_v_dim = 4
    key_dim = 32
    conv_dim = 68
    conv_kernel_size = 2

    def __init__(self) -> None:
        self.conv1d = _LastTokenConv1D(self.conv_dim)
        self.A_log = mx.zeros((self.num_v_heads,), dtype=mx.float32)
        self.dt_bias = mx.zeros((self.num_v_heads,), dtype=mx.float32)

    def in_proj_qkv(self, x: mx.array) -> mx.array:
        return x

    def in_proj_z(self, x: mx.array) -> mx.array:
        return mx.zeros(
            (1, x.shape[1], self.num_v_heads * self.head_v_dim), dtype=x.dtype
        )

    def in_proj_b(self, x: mx.array) -> mx.array:
        return mx.zeros((1, x.shape[1], self.num_v_heads), dtype=x.dtype)

    def in_proj_a(self, x: mx.array) -> mx.array:
        return mx.zeros((1, x.shape[1], self.num_k_heads), dtype=x.dtype)

    def norm(self, out: mx.array, z: mx.array) -> mx.array:
        return out

    def out_proj(self, out: mx.array) -> mx.array:
        return out


class _RaisingKernel:
    def __call__(self, **_: Any) -> tuple[mx.array, mx.array]:
        raise AssertionError("fallback path should not invoke lazy kernel")


def _require_metal() -> None:
    try:
        available = mx.metal.is_available()
    except AttributeError:
        available = False
    if not available:
        pytest.skip("MLX Metal is not available")


def _get_native_ops_or_skip() -> Any:
    try:
        return get_ops()
    except ImportError as exc:
        if "Symbol not found" in str(exc):
            pytest.skip(
                "cached native _paged_ops extension is incompatible with "
                "the active MLX; remove ~/.cache/vllm-metal/_paged_ops*.so "
                "to rebuild it"
            )
        raise


def _make_state_cache(
    *,
    max_seqs: int = 3,
    conv_kernel_dim: int = 3,
    conv_dim: int = 4,
    num_v_heads: int = 1,
    value_head_dim: int = 4,
    key_head_dim: int = 32,
) -> GDNPagedStateCache:
    return GDNPagedStateCache(
        num_layers=1,
        max_seqs=max_seqs,
        conv_kernel_dim=conv_kernel_dim,
        conv_dim=conv_dim,
        num_v_heads=num_v_heads,
        value_head_dim=value_head_dim,
        key_head_dim=key_head_dim,
        dtype=mx.float32,
    )


def _recurrent_request(
    *,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    cache: GDNPagedStateCache,
    slot_ids: list[int],
    output_dtype: mx.Dtype = mx.float32,
) -> GDNRecurrentDecodeRequest:
    return GDNRecurrentDecodeRequest(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        state_cache=cache,
        cache_idx=0,
        slot_ids=slot_ids,
        output_dtype=output_dtype,
    )


class TestLazyConvDecode:
    def test_matches_eager_per_request_conv(self) -> None:
        # Arrange
        _require_metal()
        mx.random.seed(0)
        conv_dim = 4
        kernel_size = 3
        cache = _make_state_cache(conv_kernel_dim=kernel_size, conv_dim=conv_dim)
        initial_state = mx.random.normal(cache.conv_states[0].shape).astype(mx.float32)
        cache.conv_states[0] = mx.array(initial_state)
        weight = mx.random.normal((conv_dim, kernel_size)).astype(mx.float32)
        inner = SimpleNamespace(
            conv_kernel_size=kernel_size, conv1d=_DepthwiseConv1D(weight)
        )
        mixed_qkv = mx.random.normal((1, 2, conv_dim)).astype(mx.float32)
        slot_ids = [1, 0]

        expected_state = mx.array(initial_state)
        expected_outputs = []
        for req_idx, slot in enumerate(slot_ids):
            conv_input = mx.concatenate(
                [
                    expected_state[slot : slot + 1],
                    mixed_qkv[:, req_idx : req_idx + 1],
                ],
                axis=1,
            )
            expected_state[slot : slot + 1] = conv_input[:, -(kernel_size - 1) :]
            expected_outputs.append(nn.silu(inner.conv1d(conv_input))[:, -1:])
        expected = mx.concatenate(expected_outputs, axis=1)

        # Act
        actual = GDNLazyDecodeKernels(enabled=True).try_conv_decode(
            mixed_qkv, inner, cache, 0, slot_ids
        )

        # Assert
        assert actual is not None
        mx.eval(actual, expected, cache.conv_states[0], expected_state)
        np.testing.assert_allclose(np.array(actual), np.array(expected), atol=1e-5)
        np.testing.assert_allclose(
            np.array(cache.conv_states[0]), np.array(expected_state), atol=1e-5
        )


class TestLazyRecurrentDecode:
    def test_matches_cpp_recurrent_path(self) -> None:
        # Arrange
        _require_metal()
        mx.random.seed(0)
        total_tokens = 2
        n_hk = 1
        n_hv = 1
        d_k = 32
        d_v = 4
        slot_ids = [1, 0]
        cache_lazy = _make_state_cache(value_head_dim=d_v, key_head_dim=d_k)
        cache_cpp = _make_state_cache(value_head_dim=d_v, key_head_dim=d_k)
        initial_state = mx.random.normal(cache_lazy.recurrent_states[0].shape).astype(
            mx.float32
        )
        cache_lazy.recurrent_states[0] = mx.array(initial_state)
        cache_cpp.recurrent_states[0] = mx.array(initial_state)

        q = mx.random.normal((1, total_tokens, n_hk, d_k)).astype(mx.float32)
        k = mx.random.normal((1, total_tokens, n_hk, d_k)).astype(mx.float32)
        v = mx.random.normal((1, total_tokens, n_hv, d_v)).astype(mx.float32)
        g = mx.random.normal((1, total_tokens, n_hv)).astype(mx.float32)
        beta = mx.random.normal((1, total_tokens, n_hv)).astype(mx.float32)
        request = _recurrent_request(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cache=cache_lazy,
            slot_ids=slot_ids,
        )

        # Act
        lazy_out = GDNLazyDecodeKernels(enabled=True).try_recurrent_decode(request)

        # Assert
        assert lazy_out is not None
        mx.eval(lazy_out, cache_lazy.recurrent_states[0])

        q_flat = mx.contiguous(q.reshape(total_tokens, n_hk, d_k))
        k_flat = mx.contiguous(k.reshape(total_tokens, n_hk, d_k))
        v_flat = mx.contiguous(v.reshape(total_tokens, n_hv, d_v))
        g_flat = mx.contiguous(g.reshape(total_tokens, n_hv))
        beta_flat = mx.contiguous(beta.reshape(total_tokens, n_hv))
        cpp_out = mx.zeros((total_tokens, n_hv, d_v), dtype=mx.float32)
        mx.eval(
            q_flat,
            k_flat,
            v_flat,
            g_flat,
            beta_flat,
            cache_cpp.recurrent_states[0],
            cpp_out,
        )
        _get_native_ops_or_skip().gdn_linear_attention(
            q_flat,
            k_flat,
            v_flat,
            g_flat,
            beta_flat,
            cache_cpp.recurrent_states[0],
            mx.array([0, 1, 2], dtype=mx.int32),
            mx.array(slot_ids, dtype=mx.int32),
            cpp_out,
            n_hk,
            n_hv,
            d_k,
            d_v,
        )
        mx.synchronize()
        mx.eval(
            lazy_out,
            cpp_out,
            cache_lazy.recurrent_states[0],
            cache_cpp.recurrent_states[0],
        )
        np.testing.assert_allclose(np.array(lazy_out), np.array(cpp_out), atol=1e-4)
        np.testing.assert_allclose(
            np.array(cache_lazy.recurrent_states[0]),
            np.array(cache_cpp.recurrent_states[0]),
            atol=1e-4,
        )


class TestLazyDecodeFallbacks:
    def test_falls_back_for_multi_token_requests(self) -> None:
        # Arrange
        cache = _make_state_cache()
        conv_state = mx.ones_like(cache.conv_states[0])
        recurrent_state = mx.ones_like(cache.recurrent_states[0])
        cache.conv_states[0] = conv_state
        cache.recurrent_states[0] = recurrent_state
        kernels = GDNLazyDecodeKernels(
            enabled=True,
            conv_kernel=_RaisingKernel(),
            recurrent_kernel=_RaisingKernel(),
        )
        recurrent_request = _recurrent_request(
            q=mx.zeros((1, 3, 1, 32), dtype=mx.float32),
            k=mx.zeros((1, 3, 1, 32), dtype=mx.float32),
            v=mx.zeros((1, 3, 1, 4), dtype=mx.float32),
            g=mx.zeros((1, 3, 1), dtype=mx.float32),
            beta=mx.zeros((1, 3, 1), dtype=mx.float32),
            cache=cache,
            slot_ids=[0, 1],
        )

        # Act
        conv_result = kernels.try_conv_decode(
            mx.zeros((1, 3, cache.conv_dim), dtype=mx.float32),
            SimpleNamespace(
                conv_kernel_size=cache.conv_kernel_dim, conv1d=_RaisingKernel()
            ),
            cache,
            0,
            [0, 1],
        )
        recurrent_result = kernels.try_recurrent_decode(recurrent_request)

        # Assert
        assert conv_result is None
        assert recurrent_result is None
        np.testing.assert_array_equal(
            np.array(cache.conv_states[0]), np.array(conv_state)
        )
        np.testing.assert_array_equal(
            np.array(cache.recurrent_states[0]), np.array(recurrent_state)
        )

    def test_falls_back_for_unsupported_recurrent_shape(self) -> None:
        # Arrange
        d_k = 33
        cache = _make_state_cache(key_head_dim=d_k)
        initial_state = mx.ones_like(cache.recurrent_states[0])
        cache.recurrent_states[0] = initial_state
        request = _recurrent_request(
            q=mx.zeros((1, 2, 1, d_k), dtype=mx.float32),
            k=mx.zeros((1, 2, 1, d_k), dtype=mx.float32),
            v=mx.zeros((1, 2, 1, 4), dtype=mx.float32),
            g=mx.zeros((1, 2, 1), dtype=mx.float32),
            beta=mx.zeros((1, 2, 1), dtype=mx.float32),
            cache=cache,
            slot_ids=[0, 1],
        )

        # Act
        result = GDNLazyDecodeKernels(
            enabled=True,
            conv_kernel=_RaisingKernel(),
            recurrent_kernel=_RaisingKernel(),
        ).try_recurrent_decode(request)

        # Assert
        assert result is None
        np.testing.assert_array_equal(
            np.array(cache.recurrent_states[0]), np.array(initial_state)
        )

    def test_kill_switch_disables_kernel_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        def fail_make_kernel(*_: Any) -> None:
            raise AssertionError("disabled lazy decode should not construct kernels")

        monkeypatch.setenv("VLLM_METAL_GDN_LAZY_DECODE", "0")
        monkeypatch.setattr(
            GDNLazyDecodeKernels,
            "_make_kernel",
            staticmethod(fail_make_kernel),
        )
        kernels = GDNLazyDecodeKernels.from_env()
        cache = _make_state_cache()
        request = _recurrent_request(
            q=mx.zeros((1, 1, 1, 32), dtype=mx.float32),
            k=mx.zeros((1, 1, 1, 32), dtype=mx.float32),
            v=mx.zeros((1, 1, 1, 4), dtype=mx.float32),
            g=mx.zeros((1, 1, 1), dtype=mx.float32),
            beta=mx.zeros((1, 1, 1), dtype=mx.float32),
            cache=cache,
            slot_ids=[0],
        )

        # Act
        conv_result = kernels.try_conv_decode(
            mx.zeros((1, 1, cache.conv_dim), dtype=mx.float32), object(), cache, 0, [0]
        )
        recurrent_result = kernels.try_recurrent_decode(request)

        # Assert
        assert conv_result is None
        assert recurrent_result is None


class TestGDNLazyDecodeSharedOwner:
    def test_reuses_kernel_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        calls = []

        def fake_make_kernel(name: str, *_: Any) -> object:
            calls.append(name)
            return object()

        monkeypatch.setenv("VLLM_METAL_GDN_LAZY_DECODE", "1")
        monkeypatch.setattr(
            GDNLazyDecodeKernels,
            "_make_kernel",
            staticmethod(fake_make_kernel),
        )

        # Act
        first = GDNLazyDecodeKernels.shared()
        second = GDNLazyDecodeKernels.shared()

        # Assert
        assert first is second
        assert calls == ["gdn_conv1d_silu_decode_v2", "gdn_recurrent_v2"]
        monkeypatch.setenv("VLLM_METAL_GDN_LAZY_DECODE", "0")
        disabled = GDNLazyDecodeKernels.shared()
        assert disabled is not first
        assert calls == ["gdn_conv1d_silu_decode_v2", "gdn_recurrent_v2"]


class TestGDNPagedAttentionWrapperLazyDecode:
    def test_kill_switch_uses_recurrent_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        class FakeOps:
            def __init__(self) -> None:
                self.called = False
                self.args: tuple[Any, ...] | None = None

            def gdn_linear_attention(self, *args: Any) -> None:
                self.called = True
                self.args = args
                state_pool = args[5]
                y_flat = args[8]
                state_pool[:] = 7
                y_flat[:] = 0

        def fail_make_kernel(*_: Any) -> None:
            raise AssertionError("disabled wrapper path should not construct kernels")

        fake_ops = FakeOps()
        monkeypatch.setenv("VLLM_METAL_GDN_LAZY_DECODE", "0")
        monkeypatch.setattr(attention_linear, "get_ops", lambda: fake_ops)
        monkeypatch.setattr(
            GDNLazyDecodeKernels,
            "_make_kernel",
            staticmethod(fail_make_kernel),
        )
        inner = _TinyGDNInner()
        cache = _make_state_cache(
            conv_kernel_dim=inner.conv_kernel_size,
            conv_dim=inner.conv_dim,
            num_v_heads=inner.num_v_heads,
            value_head_dim=inner.head_v_dim,
            key_head_dim=inner.head_k_dim,
        )
        wrapper = GDNPagedAttentionWrapper(
            inner, layer_idx=0, cache_idx=0, state_cache=cache
        )
        set_context(
            PagedAttentionContext(
                slot_mapping=[0, 1],
                cu_seqlens=[0, 1, 2],
                gdn_slot_mapping=[0, 1],
            )
        )

        # Act
        try:
            out = wrapper(mx.ones((1, 2, inner.conv_dim), dtype=mx.float32))
        finally:
            clear_context()

        # Assert
        assert fake_ops.called
        assert fake_ops.args is not None
        np.testing.assert_array_equal(np.array(fake_ops.args[6]), np.array([0, 1, 2]))
        np.testing.assert_array_equal(np.array(fake_ops.args[7]), np.array([0, 1]))
        np.testing.assert_array_equal(
            np.array(cache.conv_states[0][:2]),
            np.ones((2, inner.conv_kernel_size - 1, inner.conv_dim), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.array(cache.conv_states[0][2:]),
            np.zeros((1, inner.conv_kernel_size - 1, inner.conv_dim), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.array(cache.recurrent_states[0]),
            np.full(cache.recurrent_states[0].shape, 7, dtype=np.float32),
        )
        assert out.shape == (1, 2, inner.num_v_heads * inner.head_v_dim)
