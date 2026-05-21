# SPDX-License-Identifier: Apache-2.0
"""Focused regression tests for lazy GDN kernels."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from vllm_metal.metal import get_ops
from vllm_metal.metal_kernel_backend.gdn_lazy import (
    GDNLazyKernels,
    GDNRecurrentDecodeRequest,
    GDNRecurrentPrefillRequest,
)
from vllm_metal.mlx_backend.gdn_cache import GDNPagedStateCache


@pytest.fixture(autouse=True)
def _reset_lazy_gdn_kernels() -> Iterator[None]:
    GDNLazyKernels.reset_shared_for_tests()
    yield
    GDNLazyKernels.reset_shared_for_tests()


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


class _ConstantKernel:
    def __init__(self, output: mx.array, state_output: mx.array) -> None:
        outputs: tuple[mx.array, mx.array] = (output, state_output)
        self.outputs = outputs
        self.grid: tuple[int, int, int] | None = None
        self.output_shapes: list[tuple[int, ...]] | None = None

    def __call__(self, **kwargs: Any) -> tuple[mx.array, mx.array]:
        self.grid = kwargs.get("grid")
        self.output_shapes = kwargs.get("output_shapes")
        return self.outputs


class _RecordingStateKernel(_ConstantKernel):
    def __init__(
        self,
        *outputs: mx.array,
        state_input_index: int,
        slot_mapping_index: int,
    ) -> None:
        super().__init__(*outputs)
        self.state_input_index = state_input_index
        self.slot_mapping_index = slot_mapping_index
        self.state_input: mx.array | None = None
        self.slot_mapping: mx.array | None = None

    def __call__(self, **kwargs: Any) -> tuple[mx.array, mx.array]:
        self.state_input = kwargs["inputs"][self.state_input_index]
        self.slot_mapping = kwargs["inputs"][self.slot_mapping_index]
        return super().__call__(**kwargs)


class TestGDNPagedStateCache:
    def test_replacing_pending_conv_state_scatters_existing_update(self) -> None:
        # Arrange
        cache = _make_state_cache(max_seqs=4, conv_kernel_dim=3, conv_dim=4)
        cache.set_pending_conv_state(0, [2], mx.full((1, 2, 4), 5, dtype=mx.float32))

        # Act
        cache.set_pending_conv_state(0, [0, 1], mx.full((2, 2, 4), 9, dtype=mx.float32))

        # Assert
        pending_state = cache.pending_conv_state(0, [0, 1])
        assert pending_state is not None
        mx.eval(cache.conv_states[0], pending_state)
        np.testing.assert_array_equal(
            np.array(cache.conv_states[0][2]), np.full((2, 4), 5)
        )
        np.testing.assert_array_equal(
            np.array(cache.conv_states[0][:2]), np.zeros((2, 2, 4))
        )
        np.testing.assert_array_equal(np.array(pending_state), np.full((2, 2, 4), 9))

    def test_replacing_pending_recurrent_state_scatters_existing_update(self) -> None:
        # Arrange
        cache = _make_state_cache(max_seqs=4)
        cache.set_pending_recurrent_state(
            0, [2], mx.full((1, 1, 4, 32), 5, dtype=mx.float32)
        )

        # Act
        cache.set_pending_recurrent_state(
            0, [0, 1], mx.full((2, 1, 4, 32), 9, dtype=mx.float32)
        )

        # Assert
        pending_state = cache.pending_recurrent_state(0, [0, 1])
        assert pending_state is not None
        mx.eval(cache.recurrent_states[0], pending_state)
        np.testing.assert_array_equal(
            np.array(cache.recurrent_states[0][2]), np.full((1, 4, 32), 5)
        )
        np.testing.assert_array_equal(
            np.array(cache.recurrent_states[0][:2]), np.zeros((2, 1, 4, 32))
        )
        np.testing.assert_array_equal(
            np.array(pending_state), np.full((2, 1, 4, 32), 9)
        )

    def test_decode_state_view_owns_compact_slot_mapping(self) -> None:
        # Arrange
        cache = _make_state_cache(max_seqs=4, conv_kernel_dim=3, conv_dim=4)
        pending = mx.full((2, 2, 4), 9, dtype=mx.float32)
        cache.set_pending_conv_state(0, [3, 1], pending)

        # Act
        view = cache.conv_state_for_decode(0, [3, 1])

        # Assert
        assert view.state is pending
        assert view.uses_compact_state
        np.testing.assert_array_equal(np.array(view.cache_slot_ids), [3, 1])
        np.testing.assert_array_equal(np.array(view.state_slot_ids), [0, 1])


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
    threadgroup_dv: int = 4,
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
        threadgroup_dv=threadgroup_dv,
    )


def _recurrent_prefill_request(
    *,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    cache: GDNPagedStateCache,
    slot_ids: list[int],
    cu_seqlens: list[int],
    output_dtype: mx.Dtype = mx.float32,
    compute_dtype: mx.Dtype | None = None,
    defer_state_scatter: bool = False,
    materialize_outputs: bool = False,
) -> GDNRecurrentPrefillRequest:
    return GDNRecurrentPrefillRequest(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        state_cache=cache,
        cache_idx=0,
        slot_ids=slot_ids,
        output_dtype=output_dtype,
        cu_seqlens=cu_seqlens,
        compute_dtype=compute_dtype,
        defer_state_scatter=defer_state_scatter,
        materialize_outputs=materialize_outputs,
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
        actual = GDNLazyKernels(enabled=True).try_conv_decode(
            mixed_qkv, inner, cache, 0, slot_ids
        )

        # Assert
        assert actual is not None
        mx.eval(actual, expected, cache.conv_states[0], expected_state)
        np.testing.assert_allclose(np.array(actual), np.array(expected), atol=1e-5)
        np.testing.assert_allclose(
            np.array(cache.conv_states[0]), np.array(expected_state), atol=1e-5
        )

    def test_updates_only_active_conv_slots(self) -> None:
        # Arrange
        class FakeConvKernel:
            def __init__(self) -> None:
                self.grid: tuple[int, int, int] | None = None
                self.output_shapes: list[tuple[int, ...]] | None = None

            def __call__(self, **kwargs: Any) -> tuple[mx.array, mx.array]:
                self.grid = kwargs["grid"]
                self.output_shapes = kwargs["output_shapes"]
                return (
                    mx.ones((2, 4), dtype=mx.float32),
                    mx.full((2, 2, 4), 9, dtype=mx.float32),
                )

        fake_kernel = FakeConvKernel()
        cache = _make_state_cache(max_seqs=4, conv_kernel_dim=3, conv_dim=4)
        initial_state = mx.arange(4 * 2 * 4, dtype=mx.float32).reshape(4, 2, 4)
        cache.conv_states[0] = mx.array(initial_state)
        inner = SimpleNamespace(
            conv_kernel_size=3,
            conv1d=SimpleNamespace(weight=mx.ones((4, 3), dtype=mx.float32)),
        )
        slot_ids = [3, 1]

        # Act
        result = GDNLazyKernels(
            enabled=True,
            conv_kernel=fake_kernel,
            recurrent_decode_kernel=_RaisingKernel(),
        ).try_conv_decode(
            mx.zeros((1, 2, 4), dtype=mx.float32), inner, cache, 0, slot_ids
        )

        # Assert
        assert result is not None
        assert fake_kernel.grid == (8, 1, 1)
        assert fake_kernel.output_shapes == [(2, 4), (2, 2, 4)]
        mx.eval(cache.conv_states[0])
        expected_state = np.array(initial_state)
        expected_state[slot_ids] = 9
        np.testing.assert_array_equal(np.array(cache.conv_states[0]), expected_state)

    def test_mixed_dtype_state_update_matches_eager_conv(self) -> None:
        # Arrange
        _require_metal()
        mx.random.seed(1)
        conv_dim = 4
        kernel_size = 3
        cache = _make_state_cache(conv_kernel_dim=kernel_size, conv_dim=conv_dim)
        initial_state = mx.random.normal(cache.conv_states[0].shape).astype(mx.bfloat16)
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
            expected_state[slot : slot + 1] = conv_input[
                :, -(kernel_size - 1) :
            ].astype(mx.bfloat16)
            expected_outputs.append(nn.silu(inner.conv1d(conv_input))[:, -1:])
        expected = mx.concatenate(expected_outputs, axis=1)

        # Act
        actual = GDNLazyKernels(enabled=True).try_conv_decode(
            mixed_qkv, inner, cache, 0, slot_ids
        )

        # Assert
        assert actual is not None
        mx.eval(actual, expected, cache.conv_states[0], expected_state)
        assert actual.dtype == mx.float32
        assert cache.conv_states[0].dtype == mx.bfloat16
        np.testing.assert_allclose(np.array(actual), np.array(expected), atol=5e-3)
        np.testing.assert_allclose(
            np.array(cache.conv_states[0].astype(mx.float32)),
            np.array(expected_state.astype(mx.float32)),
            atol=5e-3,
        )

    def test_conv_state_update_template_uses_state_dtype(self) -> None:
        # Arrange
        class FakeConvKernel:
            def __init__(self) -> None:
                self.template: list[tuple[str, Any]] | None = None
                self.output_dtypes: list[mx.Dtype] | None = None

            def __call__(self, **kwargs: Any) -> tuple[mx.array, mx.array]:
                self.template = kwargs["template"]
                self.output_dtypes = kwargs["output_dtypes"]
                return (
                    mx.ones((1, 4), dtype=mx.float32),
                    mx.ones((1, 2, 4), dtype=mx.bfloat16),
                )

        fake_kernel = FakeConvKernel()
        cache = _make_state_cache(max_seqs=3, conv_kernel_dim=3, conv_dim=4)
        cache.conv_states[0] = cache.conv_states[0].astype(mx.bfloat16)
        inner = SimpleNamespace(
            conv_kernel_size=3,
            conv1d=SimpleNamespace(weight=mx.ones((4, 3), dtype=mx.float32)),
        )

        # Act
        result = GDNLazyKernels(
            enabled=True,
            conv_kernel=fake_kernel,
            recurrent_decode_kernel=_RaisingKernel(),
        ).try_conv_decode(mx.zeros((1, 1, 4), dtype=mx.float32), inner, cache, 0, [0])

        # Assert
        assert result is not None
        assert ("T", mx.float32) in (fake_kernel.template or [])
        assert ("StT", mx.bfloat16) in (fake_kernel.template or [])
        assert fake_kernel.output_dtypes == [mx.float32, mx.bfloat16]


class TestLazyConvPrefill:
    @pytest.mark.parametrize("cu_seqlens", [[0, 2, 5], [0, 1, 3]])
    def test_matches_eager_per_request_conv(self, cu_seqlens: list[int]) -> None:
        # Arrange
        _require_metal()
        mx.random.seed(2)
        conv_dim = 4
        kernel_size = 4
        total_tokens = cu_seqlens[-1]
        cache = _make_state_cache(
            max_seqs=4, conv_kernel_dim=kernel_size, conv_dim=conv_dim
        )
        initial_state = mx.random.normal(cache.conv_states[0].shape).astype(mx.float32)
        cache.conv_states[0] = mx.array(initial_state)
        weight = mx.random.normal((conv_dim, kernel_size)).astype(mx.float32)
        inner = SimpleNamespace(
            conv_kernel_size=kernel_size, conv1d=_DepthwiseConv1D(weight)
        )
        slot_ids = [3, 1]
        mixed_qkv = mx.random.normal((1, total_tokens, conv_dim)).astype(mx.float32)

        expected_state = mx.array(initial_state)
        expected_outputs = []
        for req_idx, slot in enumerate(slot_ids):
            start = cu_seqlens[req_idx]
            end = cu_seqlens[req_idx + 1]
            conv_input = mx.concatenate(
                [expected_state[slot : slot + 1], mixed_qkv[:, start:end]],
                axis=1,
            )
            expected_state[slot : slot + 1] = conv_input[:, -(kernel_size - 1) :]
            expected_outputs.append(
                nn.silu(inner.conv1d(conv_input))[:, -(end - start) :]
            )
        expected = mx.concatenate(expected_outputs, axis=1)

        # Act
        actual = GDNLazyKernels(enabled=True).try_conv_prefill(
            mixed_qkv, inner, cache, 0, slot_ids, cu_seqlens
        )

        # Assert
        assert actual is not None
        pending_state = cache.pending_conv_state(0, slot_ids)
        assert pending_state is not None
        mx.eval(actual, expected, pending_state, expected_state)
        np.testing.assert_allclose(np.array(actual), np.array(expected), atol=1e-5)
        np.testing.assert_allclose(
            np.array(pending_state), np.array(expected_state[slot_ids]), atol=1e-5
        )

        cache.apply_pending_conv_state(0)
        mx.eval(cache.conv_states[0])
        np.testing.assert_allclose(
            np.array(cache.conv_states[0]), np.array(expected_state), atol=1e-5
        )

    def test_updates_only_active_conv_slots(self) -> None:
        # Arrange
        fake_kernel = _ConstantKernel(
            mx.ones((5, 4), dtype=mx.float32),
            mx.full((2, 2, 4), 9, dtype=mx.float32),
        )
        cache = _make_state_cache(max_seqs=4, conv_kernel_dim=3, conv_dim=4)
        initial_state = mx.arange(4 * 2 * 4, dtype=mx.float32).reshape(4, 2, 4)
        cache.conv_states[0] = mx.array(initial_state)
        inner = SimpleNamespace(
            conv_kernel_size=3,
            conv1d=SimpleNamespace(weight=mx.ones((4, 3), dtype=mx.float32)),
        )
        slot_ids = [3, 1]

        # Act
        result = GDNLazyKernels(
            enabled=True,
            conv_kernel=_RaisingKernel(),
            conv_prefill_kernel=fake_kernel,
            recurrent_decode_kernel=_RaisingKernel(),
            recurrent_prefill_kernel=_RaisingKernel(),
        ).try_conv_prefill(
            mx.zeros((1, 5, 4), dtype=mx.float32),
            inner,
            cache,
            0,
            slot_ids,
            [0, 2, 5],
        )

        # Assert
        assert result is not None
        assert fake_kernel.grid == (36, 1, 1)
        assert fake_kernel.output_shapes == [(5, 4), (2, 2, 4)]
        pending_state = cache.pending_conv_state(0, slot_ids)
        assert pending_state is not None
        mx.eval(cache.conv_states[0], pending_state)
        np.testing.assert_array_equal(
            np.array(cache.conv_states[0]), np.array(initial_state)
        )
        np.testing.assert_array_equal(np.array(pending_state), np.full((2, 2, 4), 9))

        cache.apply_pending_conv_state(0)
        expected_state = np.array(initial_state)
        expected_state[slot_ids] = 9
        mx.eval(cache.conv_states[0])
        np.testing.assert_array_equal(np.array(cache.conv_states[0]), expected_state)

    def test_deferred_prefill_state_is_consumed_by_next_decode(self) -> None:
        # Arrange
        prefill_kernel = _ConstantKernel(
            mx.ones((5, 4), dtype=mx.float32),
            mx.full((2, 2, 4), 9, dtype=mx.float32),
        )
        decode_kernel = _RecordingStateKernel(
            mx.ones((2, 4), dtype=mx.float32),
            mx.full((2, 2, 4), 7, dtype=mx.float32),
            state_input_index=1,
            slot_mapping_index=3,
        )
        cache = _make_state_cache(max_seqs=4, conv_kernel_dim=3, conv_dim=4)
        initial_state = mx.arange(4 * 2 * 4, dtype=mx.float32).reshape(4, 2, 4)
        cache.conv_states[0] = mx.array(initial_state)
        inner = SimpleNamespace(
            conv_kernel_size=3,
            conv1d=SimpleNamespace(weight=mx.ones((4, 3), dtype=mx.float32)),
        )
        slot_ids = [3, 1]
        kernels = GDNLazyKernels(
            enabled=True,
            conv_kernel=decode_kernel,
            conv_prefill_kernel=prefill_kernel,
            recurrent_decode_kernel=_RaisingKernel(),
            recurrent_prefill_kernel=_RaisingKernel(),
        )

        # Act
        prefill_out = kernels.try_conv_prefill(
            mx.zeros((1, 5, 4), dtype=mx.float32),
            inner,
            cache,
            0,
            slot_ids,
            [0, 2, 5],
        )
        decode_out = kernels.try_conv_decode(
            mx.zeros((1, 2, 4), dtype=mx.float32), inner, cache, 0, slot_ids
        )

        # Assert
        assert prefill_out is not None
        assert decode_out is not None
        assert cache.pending_conv_state(0, slot_ids) is None
        assert decode_kernel.state_input is not None
        assert decode_kernel.state_input.shape == (2, 2, 4)
        mx.eval(cache.conv_states[0], decode_kernel.slot_mapping)
        np.testing.assert_array_equal(np.array(decode_kernel.slot_mapping), [0, 1])
        expected_state = np.array(initial_state)
        expected_state[slot_ids] = 7
        np.testing.assert_array_equal(np.array(cache.conv_states[0]), expected_state)

    def test_pending_conv_mismatched_decode_scatters_before_kernel(self) -> None:
        # Arrange
        decode_kernel = _RecordingStateKernel(
            mx.ones((2, 4), dtype=mx.float32),
            mx.full((2, 2, 4), 7, dtype=mx.float32),
            state_input_index=1,
            slot_mapping_index=3,
        )
        cache = _make_state_cache(max_seqs=4, conv_kernel_dim=3, conv_dim=4)
        initial_state = mx.arange(4 * 2 * 4, dtype=mx.float32).reshape(4, 2, 4)
        cache.conv_states[0] = mx.array(initial_state)
        cache.set_pending_conv_state(0, [3, 1], mx.full((2, 2, 4), 9, dtype=mx.float32))
        inner = SimpleNamespace(
            conv_kernel_size=3,
            conv1d=SimpleNamespace(weight=mx.ones((4, 3), dtype=mx.float32)),
        )

        # Act
        result = GDNLazyKernels(
            enabled=True,
            conv_kernel=decode_kernel,
            conv_prefill_kernel=_RaisingKernel(),
            recurrent_decode_kernel=_RaisingKernel(),
            recurrent_prefill_kernel=_RaisingKernel(),
        ).try_conv_decode(
            mx.zeros((1, 2, 4), dtype=mx.float32), inner, cache, 0, [1, 3]
        )

        # Assert
        assert result is not None
        assert cache.pending_conv_state(0, [3, 1]) is None
        assert decode_kernel.state_input is cache.conv_states[0]
        mx.eval(cache.conv_states[0], decode_kernel.slot_mapping)
        np.testing.assert_array_equal(np.array(decode_kernel.slot_mapping), [1, 3])
        expected_state = np.array(initial_state)
        expected_state[[3, 1]] = 9
        expected_state[[1, 3]] = 7
        np.testing.assert_array_equal(np.array(cache.conv_states[0]), expected_state)

    def test_next_conv_prefill_scatters_pending_state_before_replacing_it(self) -> None:
        # Arrange
        prefill_kernel = _ConstantKernel(
            mx.ones((5, 4), dtype=mx.float32),
            mx.full((2, 2, 4), 9, dtype=mx.float32),
        )

        cache = _make_state_cache(max_seqs=4, conv_kernel_dim=3, conv_dim=4)
        initial_state = mx.arange(4 * 2 * 4, dtype=mx.float32).reshape(4, 2, 4)
        cache.conv_states[0] = mx.array(initial_state)
        cache.set_pending_conv_state(0, [2], mx.full((1, 2, 4), 5, dtype=mx.float32))
        inner = SimpleNamespace(
            conv_kernel_size=3,
            conv1d=SimpleNamespace(weight=mx.ones((4, 3), dtype=mx.float32)),
        )

        # Act
        result = GDNLazyKernels(
            enabled=True,
            conv_kernel=_RaisingKernel(),
            conv_prefill_kernel=prefill_kernel,
            recurrent_decode_kernel=_RaisingKernel(),
            recurrent_prefill_kernel=_RaisingKernel(),
        ).try_conv_prefill(
            mx.zeros((1, 5, 4), dtype=mx.float32),
            inner,
            cache,
            0,
            [0, 1],
            [0, 2, 5],
        )

        # Assert
        assert result is not None
        pending_state = cache.pending_conv_state(0, [0, 1])
        assert pending_state is not None
        mx.eval(cache.conv_states[0], pending_state)
        expected_state = np.array(initial_state)
        expected_state[2] = 5
        np.testing.assert_array_equal(np.array(cache.conv_states[0]), expected_state)
        np.testing.assert_array_equal(np.array(pending_state), np.full((2, 2, 4), 9))

    @pytest.mark.parametrize("cu_seqlens", [[0, 4, 3, 5], [0, 0, 3, 5]])
    def test_rejects_ineligible_or_inconsistent_cu_seqlens(
        self, cu_seqlens: list[int]
    ) -> None:
        # Arrange
        cache = _make_state_cache(max_seqs=2, conv_kernel_dim=3, conv_dim=4)
        inner = SimpleNamespace(
            conv_kernel_size=3,
            conv1d=SimpleNamespace(weight=mx.ones((4, 3), dtype=mx.float32)),
        )

        # Act
        result = GDNLazyKernels(
            enabled=True,
            conv_kernel=_RaisingKernel(),
            conv_prefill_kernel=_RaisingKernel(),
            recurrent_decode_kernel=_RaisingKernel(),
            recurrent_prefill_kernel=_RaisingKernel(),
        ).try_conv_prefill(
            mx.zeros((1, 5, 4), dtype=mx.float32),
            inner,
            cache,
            0,
            [0, 1, 2],
            cu_seqlens,
        )

        # Assert
        assert result is None


class TestLazyRecurrentDecode:
    def test_updates_only_active_recurrent_slots(self) -> None:
        # Arrange
        class FakeRecurrentKernel:
            def __init__(self) -> None:
                self.grid: tuple[int, int, int] | None = None
                self.threadgroup: tuple[int, int, int] | None = None
                self.output_shapes: list[tuple[int, ...]] | None = None

            def __call__(self, **kwargs: Any) -> tuple[mx.array, mx.array]:
                self.grid = kwargs["grid"]
                self.threadgroup = kwargs["threadgroup"]
                self.output_shapes = kwargs["output_shapes"]
                return (
                    mx.ones((2, 2, 3), dtype=mx.float32),
                    mx.full((2, 2, 3, 32), 9, dtype=mx.float32),
                )

        fake_kernel = FakeRecurrentKernel()
        cache = _make_state_cache(
            max_seqs=4,
            num_v_heads=2,
            value_head_dim=3,
            key_head_dim=32,
        )
        initial_state = mx.arange(4 * 2 * 3 * 32, dtype=mx.float32).reshape(4, 2, 3, 32)
        cache.recurrent_states[0] = mx.array(initial_state)
        slot_ids = [3, 1]
        request = _recurrent_request(
            q=mx.zeros((1, 2, 1, 32), dtype=mx.float32),
            k=mx.zeros((1, 2, 1, 32), dtype=mx.float32),
            v=mx.zeros((1, 2, 2, 3), dtype=mx.float32),
            g=mx.zeros((1, 2, 2), dtype=mx.float32),
            beta=mx.zeros((1, 2, 2), dtype=mx.float32),
            cache=cache,
            slot_ids=slot_ids,
        )

        # Act
        result = GDNLazyKernels(
            enabled=True,
            conv_kernel=_RaisingKernel(),
            recurrent_decode_kernel=fake_kernel,
        ).try_recurrent_decode(request)

        # Assert
        assert result is not None
        assert fake_kernel.grid == (32, 3, 4)
        assert fake_kernel.threadgroup == (32, 4, 1)
        assert fake_kernel.output_shapes == [(2, 2, 3), (2, 2, 3, 32)]
        mx.eval(cache.recurrent_states[0])
        expected_state = np.array(initial_state)
        expected_state[slot_ids] = 9
        np.testing.assert_array_equal(
            np.array(cache.recurrent_states[0]), expected_state
        )

    def test_uses_requested_decode_threadgroup_dv(self) -> None:
        # Arrange
        class FakeRecurrentKernel:
            def __init__(self) -> None:
                self.threadgroup: tuple[int, int, int] | None = None

            def __call__(self, **kwargs: Any) -> tuple[mx.array, mx.array]:
                self.threadgroup = kwargs["threadgroup"]
                return (
                    mx.ones((2, 2, 3), dtype=mx.float32),
                    mx.zeros((2, 2, 3, 32), dtype=mx.float32),
                )

        fake_kernel = FakeRecurrentKernel()
        cache = _make_state_cache(
            max_seqs=2,
            num_v_heads=2,
            value_head_dim=3,
            key_head_dim=32,
        )
        request = _recurrent_request(
            q=mx.zeros((1, 2, 1, 32), dtype=mx.float32),
            k=mx.zeros((1, 2, 1, 32), dtype=mx.float32),
            v=mx.zeros((1, 2, 2, 3), dtype=mx.float32),
            g=mx.zeros((1, 2, 2), dtype=mx.float32),
            beta=mx.zeros((1, 2, 2), dtype=mx.float32),
            cache=cache,
            slot_ids=[0, 1],
            threadgroup_dv=8,
        )

        # Act
        result = GDNLazyKernels(
            enabled=True,
            conv_kernel=_RaisingKernel(),
            recurrent_decode_kernel=fake_kernel,
        ).try_recurrent_decode(request)

        # Assert
        assert result is not None
        assert fake_kernel.threadgroup == (32, 8, 1)

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
        lazy_out = GDNLazyKernels(enabled=True).try_recurrent_decode(request)

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
        # The lazy typed prefill path intentionally runs the recurrence at the
        # requested output dtype to avoid extra graph casts.  The fallback C++
        # oracle above runs in fp32, so the state can differ slightly more than
        # the rounded y output while still remaining numerically close.
        np.testing.assert_allclose(
            np.array(cache_lazy.recurrent_states[0]),
            np.array(cache_cpp.recurrent_states[0]),
            atol=1e-3,
        )


class TestLazyRecurrentPrefill:
    def test_updates_only_active_prefill_slots(self) -> None:
        # Arrange
        fake_kernel = _ConstantKernel(
            mx.ones((5, 2, 3), dtype=mx.float32),
            mx.full((2, 2, 3, 32), 9, dtype=mx.float32),
        )
        cache = _make_state_cache(
            max_seqs=4,
            num_v_heads=2,
            value_head_dim=3,
            key_head_dim=32,
        )
        initial_state = mx.arange(4 * 2 * 3 * 32, dtype=mx.float32).reshape(4, 2, 3, 32)
        cache.recurrent_states[0] = mx.array(initial_state)
        slot_ids = [3, 1]
        request = _recurrent_prefill_request(
            q=mx.zeros((1, 5, 1, 32), dtype=mx.float32),
            k=mx.zeros((1, 5, 1, 32), dtype=mx.float32),
            v=mx.zeros((1, 5, 2, 3), dtype=mx.float32),
            g=mx.zeros((1, 5, 2), dtype=mx.float32),
            beta=mx.zeros((1, 5, 2), dtype=mx.float32),
            cache=cache,
            slot_ids=slot_ids,
            cu_seqlens=[0, 2, 5],
        )

        # Act
        result = GDNLazyKernels(
            enabled=True,
            conv_kernel=_RaisingKernel(),
            recurrent_decode_kernel=_RaisingKernel(),
            recurrent_prefill_kernel=fake_kernel,
        ).try_recurrent_prefill(request)

        # Assert
        assert result is not None
        assert fake_kernel.grid == (32, 3, 4)
        assert fake_kernel.output_shapes == [(5, 2, 3), (2, 2, 3, 32)]
        mx.eval(cache.recurrent_states[0])
        expected_state = np.array(initial_state)
        expected_state[slot_ids] = 9
        np.testing.assert_array_equal(
            np.array(cache.recurrent_states[0]), expected_state
        )

    def test_mixed_prefill_accepts_one_token_decode_segments(self) -> None:
        # Arrange
        prefill_kernel = _ConstantKernel(
            mx.ones((3, 1, 4), dtype=mx.float32),
            mx.full((2, 1, 4, 32), 9, dtype=mx.float32),
        )

        cache = _make_state_cache(max_seqs=3)
        initial_state = mx.zeros_like(cache.recurrent_states[0])
        cache.recurrent_states[0] = initial_state
        slot_ids = [0, 1]
        request = _recurrent_prefill_request(
            q=mx.zeros((1, 3, 1, 32), dtype=mx.float32),
            k=mx.zeros((1, 3, 1, 32), dtype=mx.float32),
            v=mx.zeros((1, 3, 1, 4), dtype=mx.float32),
            g=mx.zeros((1, 3, 1), dtype=mx.float32),
            beta=mx.zeros((1, 3, 1), dtype=mx.float32),
            cache=cache,
            slot_ids=slot_ids,
            cu_seqlens=[0, 1, 3],
        )

        # Act
        result = GDNLazyKernels(
            enabled=True,
            conv_kernel=_RaisingKernel(),
            recurrent_decode_kernel=_RaisingKernel(),
            recurrent_prefill_kernel=prefill_kernel,
        ).try_recurrent_prefill(request)

        # Assert
        assert result is not None
        mx.eval(cache.recurrent_states[0])
        expected_state = np.zeros_like(np.array(initial_state))
        expected_state[slot_ids] = 9
        np.testing.assert_array_equal(
            np.array(cache.recurrent_states[0]), expected_state
        )

    def test_deferred_prefill_state_is_consumed_by_next_decode(self) -> None:
        # Arrange
        prefill_kernel = _ConstantKernel(
            mx.ones((5, 2, 3), dtype=mx.float32),
            mx.full((2, 2, 3, 32), 9, dtype=mx.float32),
        )
        decode_kernel = _RecordingStateKernel(
            mx.ones((2, 2, 3), dtype=mx.float32),
            mx.full((2, 2, 3, 32), 7, dtype=mx.float32),
            state_input_index=5,
            slot_mapping_index=6,
        )
        cache = _make_state_cache(
            max_seqs=4,
            num_v_heads=2,
            value_head_dim=3,
            key_head_dim=32,
        )
        initial_state = mx.arange(4 * 2 * 3 * 32, dtype=mx.float32).reshape(4, 2, 3, 32)
        cache.recurrent_states[0] = mx.array(initial_state)
        slot_ids = [3, 1]
        kernels = GDNLazyKernels(
            enabled=True,
            conv_kernel=_RaisingKernel(),
            recurrent_decode_kernel=decode_kernel,
            recurrent_prefill_kernel=prefill_kernel,
        )
        prefill_request = _recurrent_prefill_request(
            q=mx.zeros((1, 5, 1, 32), dtype=mx.float32),
            k=mx.zeros((1, 5, 1, 32), dtype=mx.float32),
            v=mx.zeros((1, 5, 2, 3), dtype=mx.float32),
            g=mx.zeros((1, 5, 2), dtype=mx.float32),
            beta=mx.zeros((1, 5, 2), dtype=mx.float32),
            cache=cache,
            slot_ids=slot_ids,
            cu_seqlens=[0, 2, 5],
            defer_state_scatter=True,
        )
        decode_request = _recurrent_request(
            q=mx.zeros((1, 2, 1, 32), dtype=mx.float32),
            k=mx.zeros((1, 2, 1, 32), dtype=mx.float32),
            v=mx.zeros((1, 2, 2, 3), dtype=mx.float32),
            g=mx.zeros((1, 2, 2), dtype=mx.float32),
            beta=mx.zeros((1, 2, 2), dtype=mx.float32),
            cache=cache,
            slot_ids=slot_ids,
        )

        # Act
        prefill_out = kernels.try_recurrent_prefill(prefill_request)
        decode_out = kernels.try_recurrent_decode(decode_request)

        # Assert
        assert prefill_out is not None
        assert decode_out is not None
        assert cache.pending_recurrent_state(0, slot_ids) is None
        assert decode_kernel.state_input is not None
        assert decode_kernel.state_input.shape == (2, 2, 3, 32)
        mx.eval(cache.recurrent_states[0], decode_kernel.slot_mapping)
        np.testing.assert_array_equal(np.array(decode_kernel.slot_mapping), [0, 1])
        expected_state = np.array(initial_state)
        expected_state[slot_ids] = 7
        np.testing.assert_array_equal(
            np.array(cache.recurrent_states[0]), expected_state
        )

    def test_materialized_deferred_prefill_evals_compact_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        prefill_kernel = _ConstantKernel(
            mx.ones((5, 2, 3), dtype=mx.float32),
            mx.full((2, 2, 3, 32), 9, dtype=mx.float32),
        )

        cache = _make_state_cache(
            max_seqs=4,
            num_v_heads=2,
            value_head_dim=3,
            key_head_dim=32,
        )
        initial_state = mx.arange(4 * 2 * 3 * 32, dtype=mx.float32).reshape(4, 2, 3, 32)
        cache.recurrent_states[0] = mx.array(initial_state)
        eval_args: list[tuple[Any, ...]] = []
        monkeypatch.setattr(mx, "eval", lambda *args: eval_args.append(args))
        slot_ids = [3, 1]
        request = _recurrent_prefill_request(
            q=mx.zeros((1, 5, 1, 32), dtype=mx.float32),
            k=mx.zeros((1, 5, 1, 32), dtype=mx.float32),
            v=mx.zeros((1, 5, 2, 3), dtype=mx.float32),
            g=mx.zeros((1, 5, 2), dtype=mx.float32),
            beta=mx.zeros((1, 5, 2), dtype=mx.float32),
            cache=cache,
            slot_ids=slot_ids,
            cu_seqlens=[0, 2, 5],
            defer_state_scatter=True,
            materialize_outputs=True,
        )

        # Act
        result = GDNLazyKernels(
            enabled=True,
            conv_kernel=_RaisingKernel(),
            recurrent_decode_kernel=_RaisingKernel(),
            recurrent_prefill_kernel=prefill_kernel,
        ).try_recurrent_prefill(request)

        # Assert
        assert result is not None
        assert len(eval_args) == 1
        assert len(eval_args[0]) == 2
        assert eval_args[0][1].shape == (2, 2, 3, 32)
        assert cache.pending_recurrent_state(0, slot_ids) is eval_args[0][1]
        np.testing.assert_array_equal(
            np.array(cache.recurrent_states[0]), np.array(initial_state)
        )

    def test_materialized_scattered_prefill_contiguates_state_pool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        prefill_kernel = _ConstantKernel(
            mx.ones((5, 2, 3), dtype=mx.float32),
            mx.full((2, 2, 3, 32), 9, dtype=mx.float32),
        )

        cache = _make_state_cache(
            max_seqs=4,
            num_v_heads=2,
            value_head_dim=3,
            key_head_dim=32,
        )
        contiguous_args: list[mx.array] = []
        eval_args: list[tuple[Any, ...]] = []

        def fake_contiguous(array: mx.array) -> mx.array:
            contiguous_args.append(array)
            return array

        monkeypatch.setattr(mx, "contiguous", fake_contiguous)
        monkeypatch.setattr(mx, "eval", lambda *args: eval_args.append(args))
        slot_ids = [3, 1]
        request = _recurrent_prefill_request(
            q=mx.zeros((1, 5, 1, 32), dtype=mx.float32),
            k=mx.zeros((1, 5, 1, 32), dtype=mx.float32),
            v=mx.zeros((1, 5, 2, 3), dtype=mx.float32),
            g=mx.zeros((1, 5, 2), dtype=mx.float32),
            beta=mx.zeros((1, 5, 2), dtype=mx.float32),
            cache=cache,
            slot_ids=slot_ids,
            cu_seqlens=[0, 2, 5],
            defer_state_scatter=False,
            materialize_outputs=True,
        )

        # Act
        result = GDNLazyKernels(
            enabled=True,
            conv_kernel=_RaisingKernel(),
            recurrent_decode_kernel=_RaisingKernel(),
            recurrent_prefill_kernel=prefill_kernel,
        ).try_recurrent_prefill(request)

        # Assert
        assert result is not None
        assert len(contiguous_args) == 1
        assert contiguous_args[0].shape == (4, 2, 3, 32)
        assert len(eval_args) == 1
        assert eval_args[0][1] is cache.recurrent_states[0]
        assert cache.pending_recurrent_state(0, slot_ids) is None

    @pytest.mark.parametrize(
        "cu_seqlens",
        [
            [0, 0, 3],
            [1, 3, 5],
            [0, 2, 4],
        ],
    )
    def test_prefill_rejects_ineligible_or_inconsistent_segments(
        self, cu_seqlens: list[int]
    ) -> None:
        # Arrange
        cache = _make_state_cache()
        initial_state = mx.ones_like(cache.recurrent_states[0])
        cache.recurrent_states[0] = initial_state
        kernels = GDNLazyKernels(
            enabled=True,
            conv_kernel=_RaisingKernel(),
            recurrent_decode_kernel=_RaisingKernel(),
            recurrent_prefill_kernel=_RaisingKernel(),
        )
        request = _recurrent_prefill_request(
            q=mx.zeros((1, 3, 1, 32), dtype=mx.float32),
            k=mx.zeros((1, 3, 1, 32), dtype=mx.float32),
            v=mx.zeros((1, 3, 1, 4), dtype=mx.float32),
            g=mx.zeros((1, 3, 1), dtype=mx.float32),
            beta=mx.zeros((1, 3, 1), dtype=mx.float32),
            cache=cache,
            slot_ids=[0, 1],
            cu_seqlens=cu_seqlens,
        )

        # Act
        result = kernels.try_recurrent_prefill(request)

        # Assert
        assert result is None
        np.testing.assert_array_equal(
            np.array(cache.recurrent_states[0]), np.array(initial_state)
        )

    def test_prefill_recurrent_template_uses_input_and_state_dtypes(self) -> None:
        # Arrange
        class FakePrefillKernel:
            def __init__(self) -> None:
                self.template: list[tuple[str, Any]] | None = None
                self.output_dtypes: list[mx.Dtype] | None = None

            def __call__(self, **kwargs: Any) -> tuple[mx.array, mx.array]:
                self.template = kwargs["template"]
                self.output_dtypes = kwargs["output_dtypes"]
                return (
                    mx.ones((3, 1, 4), dtype=kwargs["output_dtypes"][0]),
                    mx.ones((1, 1, 4, 32), dtype=kwargs["output_dtypes"][1]),
                )

        fake_kernel = FakePrefillKernel()
        cache = _make_state_cache(value_head_dim=4, key_head_dim=32)
        cache.recurrent_states[0] = mx.zeros(
            cache.recurrent_states[0].shape, dtype=mx.bfloat16
        )
        request = _recurrent_prefill_request(
            q=mx.zeros((1, 3, 1, 32), dtype=mx.bfloat16),
            k=mx.zeros((1, 3, 1, 32), dtype=mx.bfloat16),
            v=mx.zeros((1, 3, 1, 4), dtype=mx.bfloat16),
            g=mx.zeros((1, 3, 1), dtype=mx.bfloat16),
            beta=mx.zeros((1, 3, 1), dtype=mx.bfloat16),
            cache=cache,
            slot_ids=[0],
            cu_seqlens=[0, 3],
            output_dtype=mx.bfloat16,
        )

        # Act
        result = GDNLazyKernels(
            enabled=True,
            conv_kernel=_RaisingKernel(),
            recurrent_decode_kernel=_RaisingKernel(),
            recurrent_prefill_kernel=fake_kernel,
        ).try_recurrent_prefill(request)

        # Assert
        assert result is not None
        assert result.dtype == mx.bfloat16
        assert ("T", mx.bfloat16) in (fake_kernel.template or [])
        assert ("StT", mx.bfloat16) in (fake_kernel.template or [])
        assert fake_kernel.output_dtypes == [mx.bfloat16, mx.bfloat16]

    def test_prefill_compute_dtype_can_differ_from_output_dtype(self) -> None:
        # Arrange
        class FakePrefillKernel:
            def __init__(self) -> None:
                self.template: list[tuple[str, Any]] | None = None
                self.output_dtypes: list[mx.Dtype] | None = None

            def __call__(self, **kwargs: Any) -> tuple[mx.array, mx.array]:
                self.template = kwargs["template"]
                self.output_dtypes = kwargs["output_dtypes"]
                return (
                    mx.ones((3, 1, 4), dtype=kwargs["output_dtypes"][0]),
                    mx.ones((1, 1, 4, 32), dtype=kwargs["output_dtypes"][1]),
                )

        fake_kernel = FakePrefillKernel()
        cache = _make_state_cache(value_head_dim=4, key_head_dim=32)
        request = _recurrent_prefill_request(
            q=mx.zeros((1, 3, 1, 32), dtype=mx.bfloat16),
            k=mx.zeros((1, 3, 1, 32), dtype=mx.bfloat16),
            v=mx.zeros((1, 3, 1, 4), dtype=mx.bfloat16),
            g=mx.zeros((1, 3, 1), dtype=mx.bfloat16),
            beta=mx.zeros((1, 3, 1), dtype=mx.bfloat16),
            cache=cache,
            slot_ids=[0],
            cu_seqlens=[0, 3],
            output_dtype=mx.bfloat16,
            compute_dtype=mx.float32,
        )

        # Act
        result = GDNLazyKernels(
            enabled=True,
            conv_kernel=_RaisingKernel(),
            recurrent_decode_kernel=_RaisingKernel(),
            recurrent_prefill_kernel=fake_kernel,
        ).try_recurrent_prefill(request)

        # Assert
        assert result is not None
        assert result.dtype == mx.bfloat16
        assert ("T", mx.float32) in (fake_kernel.template or [])
        assert fake_kernel.output_dtypes == [mx.float32, mx.float32]

    @pytest.mark.parametrize("cu_seqlens", [[0, 2, 5], [0, 1, 3]])
    def test_matches_cpp_recurrent_path_for_prefill(
        self, cu_seqlens: list[int]
    ) -> None:
        # Arrange
        _require_metal()
        mx.random.seed(2)
        total_tokens = cu_seqlens[-1]
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
        request = _recurrent_prefill_request(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cache=cache_lazy,
            slot_ids=slot_ids,
            cu_seqlens=cu_seqlens,
        )

        # Act
        lazy_out = GDNLazyKernels(enabled=True).try_recurrent_prefill(request)

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
            mx.array(cu_seqlens, dtype=mx.int32),
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
        # The lazy typed prefill path intentionally runs the recurrence at the
        # requested output dtype to avoid extra graph casts.  The fallback C++
        # oracle above runs in fp32, so the state can differ slightly more than
        # the rounded y output while still remaining numerically close.
        np.testing.assert_allclose(
            np.array(cache_lazy.recurrent_states[0]),
            np.array(cache_cpp.recurrent_states[0]),
            atol=1e-3,
        )

    def test_prefill_uses_requested_output_dtype_without_losing_fallback_parity(
        self,
    ) -> None:
        # Arrange
        _require_metal()
        mx.random.seed(4)
        total_tokens = 5
        n_hk = 1
        n_hv = 1
        d_k = 32
        d_v = 4
        slot_ids = [1, 0]
        cu_seqlens = [0, 2, 5]
        cache_lazy = _make_state_cache(value_head_dim=d_v, key_head_dim=d_k)
        cache_cpp = _make_state_cache(value_head_dim=d_v, key_head_dim=d_k)
        initial_state = (
            mx.random.normal(cache_lazy.recurrent_states[0].shape) * 0.01
        ).astype(mx.float32)
        cache_lazy.recurrent_states[0] = mx.array(initial_state)
        cache_cpp.recurrent_states[0] = mx.array(initial_state)

        q = (mx.random.normal((1, total_tokens, n_hk, d_k)) * 0.01).astype(mx.bfloat16)
        k = (mx.random.normal((1, total_tokens, n_hk, d_k)) * 0.01).astype(mx.bfloat16)
        v = (mx.random.normal((1, total_tokens, n_hv, d_v)) * 0.01).astype(mx.bfloat16)
        g = mx.ones((1, total_tokens, n_hv), dtype=mx.bfloat16) * 0.99
        beta = mx.ones((1, total_tokens, n_hv), dtype=mx.bfloat16) * 0.5
        request = _recurrent_prefill_request(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cache=cache_lazy,
            slot_ids=slot_ids,
            cu_seqlens=cu_seqlens,
            output_dtype=mx.bfloat16,
        )

        # Act
        lazy_out = GDNLazyKernels(enabled=True).try_recurrent_prefill(request)

        # Assert
        assert lazy_out is not None
        assert lazy_out.dtype == mx.bfloat16

        q_flat = mx.contiguous(q.reshape(total_tokens, n_hk, d_k).astype(mx.float32))
        k_flat = mx.contiguous(k.reshape(total_tokens, n_hk, d_k).astype(mx.float32))
        v_flat = mx.contiguous(v.reshape(total_tokens, n_hv, d_v).astype(mx.float32))
        g_flat = mx.contiguous(g.reshape(total_tokens, n_hv).astype(mx.float32))
        beta_flat = mx.contiguous(beta.reshape(total_tokens, n_hv).astype(mx.float32))
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
            mx.array(cu_seqlens, dtype=mx.int32),
            mx.array(slot_ids, dtype=mx.int32),
            cpp_out,
            n_hk,
            n_hv,
            d_k,
            d_v,
        )
        mx.synchronize()
        lazy_cmp = lazy_out.astype(mx.float32)
        cpp_cmp = cpp_out.astype(mx.bfloat16).astype(mx.float32)
        mx.eval(
            lazy_cmp,
            cpp_cmp,
            cache_lazy.recurrent_states[0],
            cache_cpp.recurrent_states[0],
        )
        np.testing.assert_allclose(np.array(lazy_cmp), np.array(cpp_cmp), atol=1e-4)
        # The lazy typed prefill path intentionally runs the recurrence at the
        # requested output dtype to avoid extra graph casts.  The fallback C++
        # oracle above runs in fp32, so the state can differ slightly more than
        # the rounded y output while still remaining numerically close.
        np.testing.assert_allclose(
            np.array(cache_lazy.recurrent_states[0]),
            np.array(cache_cpp.recurrent_states[0]),
            atol=1e-3,
        )


class TestLazyDecodeFallbacks:
    def test_falls_back_for_multi_token_requests(self) -> None:
        # Arrange
        cache = _make_state_cache()
        conv_state = mx.ones_like(cache.conv_states[0])
        recurrent_state = mx.ones_like(cache.recurrent_states[0])
        cache.conv_states[0] = conv_state
        cache.recurrent_states[0] = recurrent_state
        kernels = GDNLazyKernels(
            enabled=True,
            conv_kernel=_RaisingKernel(),
            recurrent_decode_kernel=_RaisingKernel(),
            recurrent_prefill_kernel=_RaisingKernel(),
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
        result = GDNLazyKernels(
            enabled=True,
            conv_kernel=_RaisingKernel(),
            recurrent_decode_kernel=_RaisingKernel(),
            recurrent_prefill_kernel=_RaisingKernel(),
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
            raise AssertionError("disabled lazy GDN should not construct kernels")

        monkeypatch.setenv("VLLM_METAL_GDN_LAZY_KERNELS", "0")
        monkeypatch.setattr(
            GDNLazyKernels,
            "_make_kernel",
            staticmethod(fail_make_kernel),
        )
        kernels = GDNLazyKernels.from_env()
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


class TestGDNLazySharedOwner:
    def test_reuses_kernel_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        calls = []

        def fake_make_kernel(name: str, *_: Any) -> object:
            calls.append(name)
            return object()

        monkeypatch.setenv("VLLM_METAL_GDN_LAZY_KERNELS", "1")
        monkeypatch.setattr(
            GDNLazyKernels,
            "_make_kernel",
            staticmethod(fake_make_kernel),
        )

        # Act
        first = GDNLazyKernels.shared()
        second = GDNLazyKernels.shared()

        # Assert
        assert first is second
        assert calls == [
            "gdn_conv1d_silu_decode_v2",
            "gdn_conv1d_silu_prefill_v2",
            "gdn_recurrent_decode_v2",
            "gdn_recurrent_prefill_v2",
        ]

        monkeypatch.setenv("VLLM_METAL_GDN_LAZY_KERNELS", "0")
        disabled = GDNLazyKernels.shared()
        assert disabled is not first
        assert calls == [
            "gdn_conv1d_silu_decode_v2",
            "gdn_conv1d_silu_prefill_v2",
            "gdn_recurrent_decode_v2",
            "gdn_recurrent_prefill_v2",
        ]
