# SPDX-License-Identifier: Apache-2.0
"""Focused regression tests for lazy GDN wrapper dispatch."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

import vllm_metal.metal_kernel_backend.attention_linear as attention_linear
from vllm_metal.metal_kernel_backend.attention_linear import GDNPagedAttentionWrapper
from vllm_metal.metal_kernel_backend.gdn_lazy import GDNLazyKernels
from vllm_metal.mlx_backend.gdn_cache import GDNPagedStateCache
from vllm_metal.paged_attention_common import (
    PagedAttentionContext,
    clear_context,
    set_context,
)


@pytest.fixture(autouse=True)
def _reset_lazy_gdn_kernels() -> Iterator[None]:
    GDNLazyKernels.reset_shared_for_tests()
    yield
    GDNLazyKernels.reset_shared_for_tests()


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


class _TinyQwen3NextGDNInner(_TinyGDNInner):
    def __init__(self) -> None:
        super().__init__()
        self.in_proj_qkvz = object()


class _TinyExpandedValueGDNInner(_TinyGDNInner):
    num_v_heads = 2
    head_v_dim = 4
    value_dim = 8
    conv_dim = 72

    def in_proj_a(self, x: mx.array) -> mx.array:
        return mx.zeros((1, x.shape[1], self.num_v_heads), dtype=x.dtype)


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


class TestGDNPagedAttentionWrapperLazyKernels:
    def test_mixed_batch_with_prefill_tries_recurrent_lazy_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        class FakeLazy:
            enabled = True

            def __init__(self) -> None:
                self.prefill_called = False
                self.prefill_request: Any | None = None

            def try_recurrent_decode(self, *_: Any) -> None:
                raise AssertionError("mixed batch is not decode-only")

            def try_recurrent_prefill(self, request: Any) -> mx.array:
                self.prefill_called = True
                self.prefill_request = request
                return lazy_out

        lazy_out = mx.zeros((3, 1, 4), dtype=mx.float32)
        fake_lazy = FakeLazy()
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
        object.__setattr__(wrapper, "_gdn_lazy", fake_lazy)
        monkeypatch.setattr(
            wrapper,
            "_run_recurrent_fallback",
            lambda *_: (_ for _ in ()).throw(
                AssertionError("mixed prefill should use lazy recurrent prefill")
            ),
        )
        state = attention_linear._GDNForwardState(
            x=mx.zeros((1, 3, inner.conv_dim), dtype=mx.float32),
            cu_seqlens=[0, 1, 3],
            num_requests=2,
            total_tokens=3,
            slot_ids=[0, 1],
            num_decode_requests=1,
        )

        # Act
        result = wrapper._run_recurrent(
            q=mx.zeros((1, 3, 1, 32), dtype=mx.float32),
            k=mx.zeros((1, 3, 1, 32), dtype=mx.float32),
            v=mx.zeros((1, 3, 1, 4), dtype=mx.float32),
            g=mx.zeros((1, 3, 1), dtype=mx.float32),
            beta=mx.zeros((1, 3, 1), dtype=mx.float32),
            state=state,
        )

        # Assert
        assert result is lazy_out
        assert fake_lazy.prefill_called
        assert fake_lazy.prefill_request is not None
        assert fake_lazy.prefill_request.cu_seqlens == [0, 1, 3]
        assert fake_lazy.prefill_request.slot_ids == [0, 1]

    def test_all_one_token_mixed_batch_does_not_try_decode_lazy_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        class FakeLazy:
            enabled = True

            def __init__(self) -> None:
                self.prefill_called = False

            def try_recurrent_decode(self, *_: Any) -> None:
                raise AssertionError("mixed batch is not decode-only")

            def try_recurrent_prefill(self, *_: Any) -> None:
                self.prefill_called = True
                return None

        fallback_out = mx.zeros((2, 1, 4), dtype=mx.float32)
        fake_lazy = FakeLazy()
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
        object.__setattr__(wrapper, "_gdn_lazy", fake_lazy)
        monkeypatch.setattr(
            wrapper,
            "_run_recurrent_fallback",
            lambda *_: fallback_out,
        )
        state = attention_linear._GDNForwardState(
            x=mx.zeros((1, 2, inner.conv_dim), dtype=mx.float32),
            cu_seqlens=[0, 1, 2],
            num_requests=2,
            total_tokens=2,
            slot_ids=[0, 1],
            num_decode_requests=1,
        )

        # Act
        result = wrapper._run_recurrent(
            q=mx.zeros((1, 2, 1, 32), dtype=mx.float32),
            k=mx.zeros((1, 2, 1, 32), dtype=mx.float32),
            v=mx.zeros((1, 2, 1, 4), dtype=mx.float32),
            g=mx.zeros((1, 2, 1), dtype=mx.float32),
            beta=mx.zeros((1, 2, 1), dtype=mx.float32),
            state=state,
        )

        # Assert
        assert result is fallback_out
        assert not fake_lazy.prefill_called

    def test_one_token_prefill_does_not_try_decode_lazy_conv(self) -> None:
        # Arrange
        class FakeLazy:
            enabled = True

            def try_conv_decode(self, *_: Any) -> None:
                raise AssertionError("pure prefill is not decode-only")

            def try_conv_prefill(self, *_: Any) -> None:
                return None

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
        object.__setattr__(wrapper, "_gdn_lazy", FakeLazy())
        state = attention_linear._GDNForwardState(
            x=mx.ones((1, 1, inner.conv_dim), dtype=mx.float32),
            cu_seqlens=[0, 1],
            num_requests=1,
            total_tokens=1,
            slot_ids=[0],
            num_decode_requests=0,
        )

        # Act
        result = wrapper._run_conv(state.x, state)

        # Assert
        assert result.shape == (1, 1, inner.conv_dim)

    def test_mixed_batch_with_prefill_tries_lazy_conv(self) -> None:
        # Arrange
        class FakeLazy:
            enabled = True

            def __init__(self) -> None:
                self.prefill_called = False

            def try_conv_decode(self, *_: Any) -> None:
                raise AssertionError("mixed batch is not decode-only")

            def try_conv_prefill(self, *_: Any) -> mx.array:
                self.prefill_called = True
                return lazy_out

        inner = _TinyExpandedValueGDNInner()
        lazy_out = mx.ones((1, 3, inner.conv_dim), dtype=mx.float32)
        fake_lazy = FakeLazy()
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
        object.__setattr__(wrapper, "_gdn_lazy", fake_lazy)
        state = attention_linear._GDNForwardState(
            x=mx.zeros((1, 3, inner.conv_dim), dtype=mx.float32),
            cu_seqlens=[0, 1, 3],
            num_requests=2,
            total_tokens=3,
            slot_ids=[0, 1],
            num_decode_requests=1,
        )

        # Act
        result = wrapper._run_conv(
            mx.ones((1, 3, inner.conv_dim), dtype=mx.float32), state
        )

        # Assert
        assert result is lazy_out
        assert fake_lazy.prefill_called

    def test_one_token_prefill_uses_recurrent_fallback_without_lazy_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        class FakeLazy:
            enabled = True

            def __init__(self) -> None:
                self.prefill_called = False

            def try_recurrent_decode(self, *_: Any) -> None:
                raise AssertionError("pure prefill is not decode-only")

            def try_recurrent_prefill(self, *_: Any) -> None:
                self.prefill_called = True
                return None

        fallback_out = mx.zeros((1, 1, 4), dtype=mx.float32)
        fake_lazy = FakeLazy()
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
        object.__setattr__(wrapper, "_gdn_lazy", fake_lazy)
        monkeypatch.setattr(
            wrapper,
            "_run_recurrent_fallback",
            lambda *_: fallback_out,
        )
        state = attention_linear._GDNForwardState(
            x=mx.zeros((1, 1, inner.conv_dim), dtype=mx.float32),
            cu_seqlens=[0, 1],
            num_requests=1,
            total_tokens=1,
            slot_ids=[0],
            num_decode_requests=0,
        )

        # Act
        result = wrapper._run_recurrent(
            q=mx.zeros((1, 1, 1, 32), dtype=mx.float32),
            k=mx.zeros((1, 1, 1, 32), dtype=mx.float32),
            v=mx.zeros((1, 1, 1, 4), dtype=mx.float32),
            g=mx.zeros((1, 1, 1), dtype=mx.float32),
            beta=mx.zeros((1, 1, 1), dtype=mx.float32),
            state=state,
        )

        # Assert
        assert result is fallback_out
        assert not fake_lazy.prefill_called

    def test_multi_request_prefill_separate_projection_defers_compact_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        class FakeLazy:
            enabled = True

            def __init__(self) -> None:
                self.prefill_called = False
                self.prefill_request: Any | None = None

            def try_recurrent_decode(self, *_: Any) -> None:
                raise AssertionError("pure prefill is not decode-only")

            def try_recurrent_prefill(self, request: Any) -> mx.array:
                self.prefill_called = True
                self.prefill_request = request
                return lazy_out

        lazy_out = mx.ones((4, 1, 4), dtype=mx.float32)
        fake_lazy = FakeLazy()
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
        object.__setattr__(wrapper, "_gdn_lazy", fake_lazy)
        monkeypatch.setattr(
            wrapper,
            "_run_recurrent_fallback",
            lambda *_: (_ for _ in ()).throw(
                AssertionError("lazy recurrent prefill should handle pure prefill")
            ),
        )
        state = attention_linear._GDNForwardState(
            x=mx.zeros((1, 4, inner.conv_dim), dtype=mx.float32),
            cu_seqlens=[0, 2, 4],
            num_requests=2,
            total_tokens=4,
            slot_ids=[0, 1],
            num_decode_requests=0,
        )

        # Act
        result = wrapper._run_recurrent(
            q=mx.zeros((1, 4, 1, 32), dtype=mx.float32),
            k=mx.zeros((1, 4, 1, 32), dtype=mx.float32),
            v=mx.zeros((1, 4, 1, 4), dtype=mx.float32),
            g=mx.zeros((1, 4, 1), dtype=mx.float32),
            beta=mx.zeros((1, 4, 1), dtype=mx.float32),
            state=state,
        )

        # Assert
        assert result is lazy_out
        assert fake_lazy.prefill_called
        assert fake_lazy.prefill_request is not None
        assert fake_lazy.prefill_request.materialize_outputs
        assert fake_lazy.prefill_request.compute_dtype is None
        assert fake_lazy.prefill_request.defer_state_scatter

    def test_multi_request_prefill_expanded_value_state_scatters_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        class FakeLazy:
            enabled = True

            def __init__(self) -> None:
                self.prefill_called = False
                self.prefill_request: Any | None = None

            def try_recurrent_decode(self, *_: Any) -> None:
                raise AssertionError("pure prefill is not decode-only")

            def try_recurrent_prefill(self, request: Any) -> mx.array:
                self.prefill_called = True
                self.prefill_request = request
                return lazy_out

        lazy_out = mx.ones((4, 2, 4), dtype=mx.float32)
        fake_lazy = FakeLazy()
        inner = _TinyExpandedValueGDNInner()
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
        object.__setattr__(wrapper, "_gdn_lazy", fake_lazy)
        monkeypatch.setattr(
            wrapper,
            "_run_recurrent_fallback",
            lambda *_: (_ for _ in ()).throw(
                AssertionError("lazy recurrent prefill should handle pure prefill")
            ),
        )
        state = attention_linear._GDNForwardState(
            x=mx.zeros((1, 4, inner.conv_dim), dtype=mx.float32),
            cu_seqlens=[0, 2, 4],
            num_requests=2,
            total_tokens=4,
            slot_ids=[0, 1],
            num_decode_requests=0,
        )

        # Act
        result = wrapper._run_recurrent(
            q=mx.zeros((1, 4, 1, 32), dtype=mx.float32),
            k=mx.zeros((1, 4, 1, 32), dtype=mx.float32),
            v=mx.zeros((1, 4, 2, 4), dtype=mx.float32),
            g=mx.zeros((1, 4, 2), dtype=mx.float32),
            beta=mx.zeros((1, 4, 2), dtype=mx.float32),
            state=state,
        )

        # Assert
        assert result is lazy_out
        assert fake_lazy.prefill_called
        assert fake_lazy.prefill_request is not None
        assert fake_lazy.prefill_request.materialize_outputs
        assert fake_lazy.prefill_request.compute_dtype == mx.float32
        assert not fake_lazy.prefill_request.defer_state_scatter

    def test_expanded_separate_projection_uses_wider_decode_threadgroup(self) -> None:
        # Arrange
        inner = _TinyExpandedValueGDNInner()
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

        # Act / Assert
        assert wrapper._recurrent_decode_threadgroup_dv() == 8

    def test_compact_and_combined_projection_keep_default_decode_threadgroup(
        self,
    ) -> None:
        # Arrange
        compact_inner = _TinyGDNInner()
        compact_cache = _make_state_cache(
            conv_kernel_dim=compact_inner.conv_kernel_size,
            conv_dim=compact_inner.conv_dim,
            num_v_heads=compact_inner.num_v_heads,
            value_head_dim=compact_inner.head_v_dim,
            key_head_dim=compact_inner.head_k_dim,
        )
        combined_inner = _TinyQwen3NextGDNInner()
        combined_inner.num_v_heads = 2
        combined_cache = _make_state_cache(
            conv_kernel_dim=combined_inner.conv_kernel_size,
            conv_dim=combined_inner.conv_dim,
            num_v_heads=combined_inner.num_v_heads,
            value_head_dim=combined_inner.head_v_dim,
            key_head_dim=combined_inner.head_k_dim,
        )

        # Act
        compact = GDNPagedAttentionWrapper(
            compact_inner, layer_idx=0, cache_idx=0, state_cache=compact_cache
        )
        combined = GDNPagedAttentionWrapper(
            combined_inner, layer_idx=0, cache_idx=0, state_cache=combined_cache
        )

        # Assert
        assert compact._recurrent_decode_threadgroup_dv() == 4
        assert combined._recurrent_decode_threadgroup_dv() == 4

    def test_single_request_prefill_scatters_recurrent_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        class FakeLazy:
            enabled = True

            def __init__(self) -> None:
                self.prefill_request: Any | None = None

            def try_recurrent_decode(self, *_: Any) -> None:
                raise AssertionError("pure prefill is not decode-only")

            def try_recurrent_prefill(self, request: Any) -> mx.array:
                self.prefill_request = request
                return lazy_out

        lazy_out = mx.ones((3, 1, 4), dtype=mx.float32)
        fake_lazy = FakeLazy()
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
        object.__setattr__(wrapper, "_gdn_lazy", fake_lazy)
        state = attention_linear._GDNForwardState(
            x=mx.zeros((1, 3, inner.conv_dim), dtype=mx.float32),
            cu_seqlens=[0, 3],
            num_requests=1,
            total_tokens=3,
            slot_ids=[2],
            num_decode_requests=0,
        )

        # Act
        result = wrapper._run_recurrent(
            q=mx.zeros((1, 3, 1, 32), dtype=mx.float32),
            k=mx.zeros((1, 3, 1, 32), dtype=mx.float32),
            v=mx.zeros((1, 3, 1, 4), dtype=mx.float32),
            g=mx.zeros((1, 3, 1), dtype=mx.float32),
            beta=mx.zeros((1, 3, 1), dtype=mx.float32),
            state=state,
        )

        # Assert
        assert result is lazy_out
        assert fake_lazy.prefill_request is not None
        assert not fake_lazy.prefill_request.materialize_outputs
        assert fake_lazy.prefill_request.compute_dtype is None
        assert not fake_lazy.prefill_request.defer_state_scatter
        assert fake_lazy.prefill_request.slot_ids == [2]

    def test_multi_request_prefill_separate_projection_materializes_compact_conv_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
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
        eval_args: list[tuple[Any, ...]] = []
        monkeypatch.setattr(
            attention_linear.mx,
            "eval",
            lambda *args: eval_args.append(args),
        )
        state = attention_linear._GDNForwardState(
            x=mx.zeros((1, 4, inner.conv_dim), dtype=mx.float32),
            cu_seqlens=[0, 2, 4],
            num_requests=2,
            total_tokens=4,
            slot_ids=[0, 1],
            num_decode_requests=0,
        )

        # Act
        wrapper._run_conv(mx.ones((1, 4, inner.conv_dim), dtype=mx.float32), state)

        # Assert
        pending_state = cache.pending_conv_state(0, [0, 1])
        assert pending_state is not None
        assert len(eval_args) == 1
        assert len(eval_args[0]) == 1
        assert eval_args[0][0] is pending_state

    def test_eager_conv_prefill_fallback_defers_compact_state(self) -> None:
        # Arrange
        inner = _TinyGDNInner()
        cache = _make_state_cache(
            conv_kernel_dim=inner.conv_kernel_size,
            conv_dim=inner.conv_dim,
            num_v_heads=inner.num_v_heads,
            value_head_dim=inner.head_v_dim,
            key_head_dim=inner.head_k_dim,
        )
        cache.set_pending_conv_state(
            0, [2], mx.full((1, inner.conv_kernel_size - 1, inner.conv_dim), 5)
        )
        wrapper = GDNPagedAttentionWrapper(
            inner, layer_idx=0, cache_idx=0, state_cache=cache
        )
        state = attention_linear._GDNForwardState(
            x=mx.zeros((1, 4, inner.conv_dim), dtype=mx.float32),
            cu_seqlens=[0, 2, 4],
            num_requests=2,
            total_tokens=4,
            slot_ids=[0, 1],
            num_decode_requests=0,
        )

        # Act
        result = wrapper._run_conv(
            mx.ones((1, 4, inner.conv_dim), dtype=mx.float32), state
        )

        # Assert
        assert result.shape[-1] == inner.conv_dim
        pending_state = cache.pending_conv_state(0, [0, 1])
        assert pending_state is not None
        mx.eval(cache.conv_states[0], pending_state)
        np.testing.assert_array_equal(
            np.array(cache.conv_states[0][2]),
            np.full((inner.conv_kernel_size - 1, inner.conv_dim), 5, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.array(cache.conv_states[0][:2]),
            np.zeros((2, inner.conv_kernel_size - 1, inner.conv_dim), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.array(pending_state),
            np.ones((2, inner.conv_kernel_size - 1, inner.conv_dim), dtype=np.float32),
        )

    def test_multi_request_prefill_compact_separate_projection_skips_lazy_conv(
        self,
    ) -> None:
        # Arrange
        class FakeLazy:
            enabled = True

            def try_conv_decode(self, *_: Any) -> None:
                raise AssertionError("pure prefill is not decode-only")

            def try_conv_prefill(self, *_: Any) -> None:
                raise AssertionError("compact separate projection keeps eager conv")

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
        object.__setattr__(wrapper, "_gdn_lazy", FakeLazy())
        state = attention_linear._GDNForwardState(
            x=mx.zeros((1, 4, inner.conv_dim), dtype=mx.float32),
            cu_seqlens=[0, 2, 4],
            num_requests=2,
            total_tokens=4,
            slot_ids=[0, 1],
            num_decode_requests=0,
        )

        # Act
        result = wrapper._run_conv(
            mx.ones((1, 4, inner.conv_dim), dtype=mx.float32), state
        )

        # Assert
        assert result.shape[-1] == inner.conv_dim

    def test_multi_request_prefill_expanded_value_state_tries_lazy_conv(
        self,
    ) -> None:
        # Arrange
        class FakeLazy:
            enabled = True

            def __init__(self) -> None:
                self.prefill_called = False

            def try_conv_decode(self, *_: Any) -> None:
                raise AssertionError("pure prefill is not decode-only")

            def try_conv_prefill(self, *_: Any) -> mx.array:
                self.prefill_called = True
                return lazy_out

        inner = _TinyExpandedValueGDNInner()
        lazy_out = mx.ones((1, 4, inner.conv_dim), dtype=mx.float32)
        fake_lazy = FakeLazy()
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
        object.__setattr__(wrapper, "_gdn_lazy", fake_lazy)
        state = attention_linear._GDNForwardState(
            x=mx.zeros((1, 4, inner.conv_dim), dtype=mx.float32),
            cu_seqlens=[0, 2, 4],
            num_requests=2,
            total_tokens=4,
            slot_ids=[0, 1],
            num_decode_requests=0,
        )

        # Act
        result = wrapper._run_conv(
            mx.ones((1, 4, inner.conv_dim), dtype=mx.float32), state
        )

        # Assert
        assert result is lazy_out
        assert fake_lazy.prefill_called

    def test_multi_request_prefill_qwen3_next_style_keeps_conv_state_lazy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        inner = _TinyQwen3NextGDNInner()
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
        eval_args: list[tuple[Any, ...]] = []
        monkeypatch.setattr(
            attention_linear.mx,
            "eval",
            lambda *args: eval_args.append(args),
        )
        state = attention_linear._GDNForwardState(
            x=mx.zeros((1, 4, inner.conv_dim), dtype=mx.float32),
            cu_seqlens=[0, 2, 4],
            num_requests=2,
            total_tokens=4,
            slot_ids=[0, 1],
            num_decode_requests=0,
        )

        # Act
        wrapper._run_conv(mx.ones((1, 4, inner.conv_dim), dtype=mx.float32), state)

        # Assert
        assert eval_args == []

    def test_multi_request_prefill_qwen3_next_style_tries_lazy_prefill(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        class FakeLazy:
            enabled = True

            def __init__(self) -> None:
                self.prefill_called = False
                self.prefill_request: Any | None = None

            def try_recurrent_decode(self, *_: Any) -> None:
                raise AssertionError("pure prefill is not decode-only")

            def try_recurrent_prefill(self, request: Any) -> mx.array:
                self.prefill_called = True
                self.prefill_request = request
                return lazy_out

        lazy_out = mx.ones((4, 1, 4), dtype=mx.float32)
        fake_lazy = FakeLazy()
        inner = _TinyQwen3NextGDNInner()
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
        object.__setattr__(wrapper, "_gdn_lazy", fake_lazy)
        monkeypatch.setattr(
            wrapper,
            "_run_recurrent_fallback",
            lambda *_: (_ for _ in ()).throw(
                AssertionError("lazy recurrent prefill should handle pure prefill")
            ),
        )
        state = attention_linear._GDNForwardState(
            x=mx.zeros((1, 4, inner.conv_dim), dtype=mx.float32),
            cu_seqlens=[0, 2, 4],
            num_requests=2,
            total_tokens=4,
            slot_ids=[0, 1],
            num_decode_requests=0,
        )

        # Act
        result = wrapper._run_recurrent(
            q=mx.zeros((1, 4, 1, 32), dtype=mx.float32),
            k=mx.zeros((1, 4, 1, 32), dtype=mx.float32),
            v=mx.zeros((1, 4, 1, 4), dtype=mx.float32),
            g=mx.zeros((1, 4, 1), dtype=mx.float32),
            beta=mx.zeros((1, 4, 1), dtype=mx.float32),
            state=state,
        )

        # Assert
        assert result is lazy_out
        assert fake_lazy.prefill_called
        assert fake_lazy.prefill_request is not None
        assert not fake_lazy.prefill_request.materialize_outputs
        assert fake_lazy.prefill_request.compute_dtype is None
        assert fake_lazy.prefill_request.defer_state_scatter

    def test_long_multi_request_prefill_qwen3_next_style_tries_lazy_prefill(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        class FakeLazy:
            enabled = True

            def __init__(self) -> None:
                self.prefill_called = False

            def try_recurrent_decode(self, *_: Any) -> None:
                raise AssertionError("pure prefill is not decode-only")

            def try_recurrent_prefill(self, *_: Any) -> mx.array:
                self.prefill_called = True
                return lazy_out

        total_tokens = 4096
        lazy_out = mx.ones((total_tokens, 1, 4), dtype=mx.float32)
        fake_lazy = FakeLazy()
        inner = _TinyQwen3NextGDNInner()
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
        object.__setattr__(wrapper, "_gdn_lazy", fake_lazy)
        monkeypatch.setattr(
            wrapper,
            "_run_recurrent_fallback",
            lambda *_: (_ for _ in ()).throw(
                AssertionError("lazy recurrent prefill should handle pure prefill")
            ),
        )
        state = attention_linear._GDNForwardState(
            x=mx.zeros((1, total_tokens, inner.conv_dim), dtype=mx.float32),
            cu_seqlens=[0, 2048, total_tokens],
            num_requests=2,
            total_tokens=total_tokens,
            slot_ids=[0, 1],
            num_decode_requests=0,
        )

        # Act
        result = wrapper._run_recurrent(
            q=mx.zeros((1, total_tokens, 1, 32), dtype=mx.float32),
            k=mx.zeros((1, total_tokens, 1, 32), dtype=mx.float32),
            v=mx.zeros((1, total_tokens, 1, 4), dtype=mx.float32),
            g=mx.zeros((1, total_tokens, 1), dtype=mx.float32),
            beta=mx.zeros((1, total_tokens, 1), dtype=mx.float32),
            state=state,
        )

        # Assert
        assert result is lazy_out
        assert fake_lazy.prefill_called

    def test_rejects_duplicate_gdn_slots(self) -> None:
        # Arrange
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
                gdn_slot_mapping=[1, 1],
            )
        )

        # Act / Assert
        try:
            with pytest.raises(RuntimeError, match="unique slots"):
                wrapper(mx.ones((1, 2, inner.conv_dim), dtype=mx.float32))
        finally:
            clear_context()

    def test_rejects_out_of_range_gdn_slots(self) -> None:
        # Arrange
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
                gdn_slot_mapping=[0, cache.max_seqs],
            )
        )

        # Act / Assert
        try:
            with pytest.raises(RuntimeError, match="out-of-range slot"):
                wrapper(mx.ones((1, 2, inner.conv_dim), dtype=mx.float32))
        finally:
            clear_context()

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
            raise AssertionError(
                "disabled lazy GDN wrapper path should not construct kernels"
            )

        fake_ops = FakeOps()
        monkeypatch.setenv("VLLM_METAL_GDN_LAZY_KERNELS", "0")
        monkeypatch.setattr(attention_linear, "get_ops", lambda: fake_ops)
        monkeypatch.setattr(
            GDNLazyKernels,
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
        pending_conv = cache.pending_conv_state(0, [0, 1])
        assert pending_conv is None
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
