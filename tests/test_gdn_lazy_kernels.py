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

from tests.stub_runner import make_state_cache
from vllm_metal.attention.caches.state_cache import PagedStateCache
from vllm_metal.attention.impls.gdn_lazy import (
    GDNLazyKernels,
    GDNRecurrentDecodeRequest,
    GDNRecurrentPrefillRequest,
)
from vllm_metal.attention.impls.linear import GDNPagedAttentionWrapper, _GDNForwardState
from vllm_metal.metal import get_ops


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
        self.template: list[tuple[str, Any]] | None = None
        self.threadgroup: tuple[int, int, int] | None = None

    def __call__(self, **kwargs: Any) -> tuple[mx.array, mx.array]:
        self.grid = kwargs.get("grid")
        self.output_shapes = kwargs.get("output_shapes")
        self.template = kwargs.get("template")
        self.threadgroup = kwargs.get("threadgroup")
        return self.outputs


def _force_generic_prefill(kernels: GDNLazyKernels) -> None:
    """Exercise the generic specialization at packed-eligible shapes in tests."""
    kernel = kernels._recurrent_prefill_kernel
    assert kernel is not None

    def generic(**kwargs: Any) -> tuple[mx.array, mx.array]:
        template = dict(kwargs["template"])
        template["Packed"] = False
        kwargs["template"] = list(template.items())
        kwargs["grid"] = (32, template["Dv"], kwargs["grid"][2])
        kwargs["threadgroup"] = (32, 4, 1)
        return kernel(**kwargs)

    kernels._recurrent_prefill_kernel = generic


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


class TestPagedStateCache:
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

    def test_recurrent_decode_view_drains_mismatched_pending_state(self) -> None:
        # Arrange
        cache = _make_state_cache(max_seqs=4)
        initial_state = mx.arange(4 * 1 * 4 * 32, dtype=mx.float32).reshape(4, 1, 4, 32)
        cache.recurrent_states[0] = mx.array(initial_state)
        cache.set_pending_recurrent_state(
            0, [3, 1], mx.full((2, 1, 4, 32), 7, dtype=mx.float32)
        )

        # Act
        view = cache.recurrent_state_for_decode(0, [0, 2])

        # Assert
        assert not view.uses_compact_state
        assert not cache.has_pending_recurrent_state(0)
        mx.eval(view.state, view.state_slot_ids)
        expected_state = np.array(initial_state)
        expected_state[[3, 1]] = 7
        np.testing.assert_array_equal(np.array(view.state), expected_state)
        np.testing.assert_array_equal(np.array(view.state_slot_ids), [0, 2])

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

    def test_copy_slots_gathers_overlapping_sources_before_writing(self) -> None:
        cache = _make_state_cache(
            max_seqs=4,
            conv_kernel_dim=2,
            conv_dim=1,
            value_head_dim=1,
            key_head_dim=1,
        )
        cache.conv_states[0] = mx.arange(4, dtype=mx.float32).reshape(4, 1, 1)
        cache.recurrent_states[0] = mx.arange(4, dtype=mx.float32).reshape(4, 1, 1, 1)

        # Destination 2 is also the second source. Both source rows must be
        # gathered before the in-place writes begin.
        cache.copy_slots([1, 2], [2, 3], [0])
        mx.eval(cache.conv_states[0], cache.recurrent_states[0])

        np.testing.assert_array_equal(
            np.array(cache.conv_states[0]).reshape(-1), [0.0, 1.0, 1.0, 2.0]
        )
        np.testing.assert_array_equal(
            np.array(cache.recurrent_states[0]).reshape(-1), [0.0, 1.0, 1.0, 2.0]
        )

    def test_identity_layout_still_drains_every_layer(self) -> None:
        # mamba_cache_mode="none" never calls set_layer_layout: every layer is
        # its own pool and must still be drained.
        cache = _make_state_cache(num_layers=3, max_seqs=4)
        for layer_idx in range(3):
            cache.set_pending_recurrent_state(
                layer_idx, [layer_idx], mx.full((1, 1, 4, 32), 2, dtype=mx.float32)
            )

        cache.apply_pending_recurrent_states()

        for layer_idx in range(3):
            assert not cache.has_pending_recurrent_state(layer_idx)
            np.testing.assert_array_equal(
                np.array(cache.recurrent_states[layer_idx][layer_idx]), 2.0
            )


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
    num_layers: int = 1,
    max_seqs: int = 3,
    conv_kernel_dim: int = 3,
    conv_dim: int = 4,
    num_v_heads: int = 1,
    value_head_dim: int = 4,
    key_head_dim: int = 32,
    dtype: mx.Dtype = mx.float32,
    recurrent_dtype: mx.Dtype = mx.float32,
) -> PagedStateCache:
    return make_state_cache(
        num_layers=num_layers,
        max_seqs=max_seqs,
        conv_kernel_dim=conv_kernel_dim,
        conv_dim=conv_dim,
        num_v_heads=num_v_heads,
        value_head_dim=value_head_dim,
        key_head_dim=key_head_dim,
        dtype=dtype,
        recurrent_dtype=recurrent_dtype,
    )


def _recurrent_request(
    *,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    cache: PagedStateCache,
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
    cache: PagedStateCache,
    slot_ids: list[int],
    cu_seqlens: list[int],
    output_dtype: mx.Dtype = mx.float32,
    compute_dtype: mx.Dtype | None = None,
    defer_state_scatter: bool = False,
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

    @pytest.mark.parametrize(
        "input_dtype,state_dtype",
        [(mx.float32, mx.bfloat16), (mx.bfloat16, mx.float32)],
    )
    def test_mixed_dtype_state_update_matches_eager_conv(
        self, input_dtype, state_dtype
    ) -> None:
        # Arrange
        _require_metal()
        mx.random.seed(1)
        conv_dim = 4
        kernel_size = 3
        cache = _make_state_cache(
            conv_kernel_dim=kernel_size, conv_dim=conv_dim, dtype=state_dtype
        )
        initial_state = mx.random.normal(cache.conv_states[0].shape).astype(state_dtype)
        cache.conv_states[0] = mx.array(initial_state)
        weight = mx.random.normal((conv_dim, kernel_size)).astype(mx.float32)
        inner = SimpleNamespace(
            conv_kernel_size=kernel_size, conv1d=_DepthwiseConv1D(weight)
        )
        mixed_qkv = mx.random.normal((1, 2, conv_dim)).astype(input_dtype)
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
            ].astype(state_dtype)
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
        assert cache.conv_states[0].dtype == state_dtype
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
        mixed_qkv = mx.random.normal((1, total_tokens, conv_dim)).astype(mx.bfloat16)

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
        # The kernel must read the stable pool, not the compact pending rows.
        # Object identity no longer distinguishes them: the in-place scatter
        # mints a fresh handle onto the same buffer, so compare shape.
        assert decode_kernel.state_input.shape == cache.conv_states[0].shape
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
        # Decode defers its compact update; draining it must land the new state
        # on the active slots and leave every other slot untouched.
        assert cache.pending_recurrent_state(0, slot_ids) is not None
        cache.apply_pending_recurrent_state(0)
        mx.eval(cache.recurrent_states[0])
        expected_state = np.array(initial_state)
        expected_state[slot_ids] = 9
        np.testing.assert_array_equal(
            np.array(cache.recurrent_states[0]), expected_state
        )

    def test_consecutive_decodes_never_touch_the_stable_pool(self) -> None:
        # Arrange
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
        )
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
        for _ in range(2):
            assert kernels.try_recurrent_decode(request) is not None

        # Assert: each step fed the previous step's compact update straight
        # back into the kernel, so the pool still holds its pre-decode bytes.
        assert decode_kernel.state_input is not None
        assert decode_kernel.state_input.shape == (2, 2, 3, 32)
        mx.eval(
            cache.recurrent_states[0],
            decode_kernel.state_input,
            decode_kernel.slot_mapping,
        )
        np.testing.assert_array_equal(
            np.array(decode_kernel.state_input), np.full((2, 2, 3, 32), 7)
        )
        np.testing.assert_array_equal(np.array(decode_kernel.slot_mapping), [0, 1])
        np.testing.assert_array_equal(
            np.array(cache.recurrent_states[0]), np.array(initial_state)
        )
        # Draining once publishes the latest update to the active slots.
        cache.apply_pending_recurrent_state(0)
        mx.eval(cache.recurrent_states[0])
        expected_state = np.array(initial_state)
        expected_state[slot_ids] = 7
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

    @pytest.mark.parametrize("input_dtype", [mx.float32, mx.bfloat16, mx.float16])
    @pytest.mark.parametrize("state_dtype", [mx.float32, mx.bfloat16, mx.float16])
    def test_matches_cpp_recurrent_path(self, input_dtype, state_dtype) -> None:
        # Arrange
        _require_metal()
        mx.random.seed(0)
        total_tokens = 2
        n_hk = 1
        n_hv = 1
        d_k = 32
        d_v = 4
        slot_ids = [1, 0]
        cache_lazy = _make_state_cache(
            value_head_dim=d_v, key_head_dim=d_k, recurrent_dtype=state_dtype
        )
        cache_cpp = _make_state_cache(
            value_head_dim=d_v, key_head_dim=d_k, recurrent_dtype=state_dtype
        )
        initial_state = mx.random.normal(cache_lazy.recurrent_states[0].shape).astype(
            state_dtype
        )
        cache_lazy.recurrent_states[0] = mx.array(initial_state)
        cache_cpp.recurrent_states[0] = mx.array(initial_state)

        q = mx.random.normal((1, total_tokens, n_hk, d_k)).astype(input_dtype)
        k = mx.random.normal((1, total_tokens, n_hk, d_k)).astype(input_dtype)
        v = mx.random.normal((1, total_tokens, n_hv, d_v)).astype(input_dtype)
        g = mx.random.normal((1, total_tokens, n_hv)).astype(input_dtype)
        beta = mx.random.normal((1, total_tokens, n_hv)).astype(input_dtype)
        request = _recurrent_request(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cache=cache_lazy,
            slot_ids=slot_ids,
            output_dtype=input_dtype,
        )

        # Act
        lazy_out = GDNLazyKernels(enabled=True).try_recurrent_decode(request)

        # Assert
        assert lazy_out is not None
        # The lazy decode path parks its state update; drain it so the pool
        # can be compared against the C++ oracle's in-place pool.
        cache_lazy.apply_pending_recurrent_state(0)
        mx.eval(lazy_out, cache_lazy.recurrent_states[0])

        _get_native_ops_or_skip()
        fallback = GDNPagedAttentionWrapper(
            _TinyGDNInner(), layer_idx=0, cache_idx=0, state_cache=cache_cpp
        )
        fallback_args = (
            q,
            k,
            v,
            g,
            beta,
            _GDNForwardState(
                x=mx.zeros((1, total_tokens, 68), dtype=input_dtype),
                cu_seqlens=[0, 1, 2],
                num_requests=2,
                total_tokens=total_tokens,
                slot_ids=slot_ids,
                num_decode_requests=2,
            ),
        )
        if mx.result_type(input_dtype, state_dtype) != state_dtype:
            with pytest.raises(RuntimeError, match="--mamba-ssm-cache-dtype float32"):
                fallback._run_recurrent_fallback(*fallback_args)
            return
        cpp_out = fallback._run_recurrent_fallback(*fallback_args)
        mx.eval(
            lazy_out,
            cpp_out,
            cache_lazy.recurrent_states[0],
            cache_cpp.recurrent_states[0],
        )
        np.testing.assert_allclose(
            np.array(lazy_out.astype(mx.float32)),
            np.array(cpp_out.astype(mx.float32)),
            atol=1e-4,
        )
        assert cache_lazy.recurrent_states[0].dtype == state_dtype
        assert cache_cpp.recurrent_states[0].dtype == state_dtype
        np.testing.assert_allclose(
            np.array(cache_lazy.recurrent_states[0].astype(mx.float32)),
            np.array(cache_cpp.recurrent_states[0].astype(mx.float32)),
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

    @pytest.mark.parametrize(
        ("d_k", "d_v", "use_packed"),
        [
            (128, 8, True),
            (128, 24, True),
            (64, 8, False),
            (256, 8, False),
            (128, 7, False),
        ],
    )
    def test_prefill_specialization_follows_shape_independently_of_nax(
        self,
        monkeypatch: pytest.MonkeyPatch,
        d_k: int,
        d_v: int,
        use_packed: bool,
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_DISABLE_NAX", "1")
        # Exercise real dispatch with stub kernels and deferred state on CPU.
        with mx.stream(mx.cpu):
            selected_kernel = _ConstantKernel(
                mx.ones((2, 1, d_v)), mx.full((1, 1, d_v, d_k), 9.0)
            )
            cache = _make_state_cache(value_head_dim=d_v, key_head_dim=d_k)
            result = GDNLazyKernels(
                enabled=True,
                conv_kernel=_RaisingKernel(),
                conv_prefill_kernel=_RaisingKernel(),
                recurrent_decode_kernel=_RaisingKernel(),
                recurrent_prefill_kernel=selected_kernel,
            ).try_recurrent_prefill(
                _recurrent_prefill_request(
                    q=mx.zeros((1, 2, 1, d_k)),
                    k=mx.zeros((1, 2, 1, d_k)),
                    v=mx.zeros((1, 2, 1, d_v)),
                    g=mx.ones((1, 2, 1)),
                    beta=mx.ones((1, 2, 1)),
                    cache=cache,
                    slot_ids=[2],
                    cu_seqlens=[0, 2],
                    defer_state_scatter=True,
                )
            )
            assert result is not None
            assert dict(selected_kernel.template or [])["Packed"] is use_packed
            assert selected_kernel.grid == (32, d_v // 8 if use_packed else d_v, 1)
            assert selected_kernel.threadgroup == (32, 2 if use_packed else 4, 1)
            np.testing.assert_array_equal(np.array(result), 1)
            np.testing.assert_array_equal(
                np.array(cache.pending_recurrent_state(0, [2])), 9
            )

    @pytest.mark.parametrize(
        ("dtype", "compute_dtype", "state_dtype", "d_v", "n_hv"),
        [
            (mx.float32, None, mx.float32, 8, 2),
            (mx.float16, None, mx.float32, 24, 4),
            pytest.param(mx.float16, None, mx.float16, 24, 4, id="fp16-state"),
            (mx.bfloat16, None, mx.bfloat16, 128, 8),
            (mx.bfloat16, mx.float32, mx.float32, 24, 4),
            pytest.param(
                mx.bfloat16, mx.float32, mx.float32, 128, 6, id="gqa3-bf16-fp32"
            ),
        ],
    )
    def test_packed_prefill_matches_generic_with_ragged_strided_state(
        self,
        dtype: mx.Dtype,
        compute_dtype: mx.Dtype | None,
        state_dtype: mx.Dtype,
        d_v: int,
        n_hv: int,
    ) -> None:
        """Typed ragged prefill respects GQA, slot strides and untouched padding."""
        _require_metal()
        mx.random.seed(7)
        d_k, n_hk, total_tokens = 128, 2, 37
        slot_ids, cu_seqlens = [3, 0, 2], [0, 1, 4, total_tokens]
        # Transpose after normalization to exercise the contiguous input wrapper.
        q, k = [mx.random.normal((1, n_hk, total_tokens, d_k)) for _ in range(2)]
        q = (q / mx.linalg.norm(q, axis=-1, keepdims=True) / d_k**0.5).astype(dtype)
        k = (k / mx.linalg.norm(k, axis=-1, keepdims=True)).astype(dtype)
        q, k = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3)
        v = (mx.random.normal((1, n_hv, total_tokens, d_v)) * 0.1).astype(dtype)
        v = v.transpose(0, 2, 1, 3)
        g = mx.random.uniform(0.5, 1.0, (1, n_hv, total_tokens)).astype(dtype)
        beta = mx.random.uniform(0.0, 1.0, g.shape).astype(dtype)
        g, beta = g.transpose(0, 2, 1), beta.transpose(0, 2, 1)
        width = n_hv * d_v * d_k
        initial = np.random.default_rng(7).normal(0, 0.1, (4, width + 32))
        initial = initial.astype(np.float32)
        initial[:, width:] = 99
        results = []
        for packed in (False, True):
            kernels = GDNLazyKernels(enabled=True)
            if not packed:
                _force_generic_prefill(kernels)
            cache = _make_state_cache(
                max_seqs=4,
                num_v_heads=n_hv,
                value_head_dim=d_v,
                key_head_dim=d_k,
                recurrent_dtype=state_dtype,
            )
            # Native scatter mutates storage: each arm needs its own allocation.
            backing = mx.array(initial).astype(state_dtype)
            cache.recurrent_states[0] = _get_native_ops_or_skip().as_strided(
                backing, (4, n_hv, d_v, d_k), (width + 32, d_v * d_k, d_k, 1)
            )
            untouched = np.array(backing.astype(mx.float32))
            out = kernels.try_recurrent_prefill(
                _recurrent_prefill_request(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    cache=cache,
                    slot_ids=slot_ids,
                    cu_seqlens=cu_seqlens,
                    output_dtype=dtype,
                    compute_dtype=compute_dtype,
                )
            )
            assert out is not None
            assert out.dtype == dtype
            assert cache.recurrent_states[0].dtype == state_dtype
            mx.eval(out, cache.recurrent_states[0])
            stored = np.array(backing.astype(mx.float32))
            np.testing.assert_array_equal(stored[1], untouched[1])
            np.testing.assert_array_equal(stored[:, width:], untouched[:, width:])
            results.append((np.array(out.astype(mx.float32)), stored))
        # simd_sum lowering is not specified; allow rounding across GPU families.
        for packed, generic, result_dtype in zip(
            results[1], results[0], (dtype, state_dtype), strict=True
        ):
            rtol = {mx.float32: 2e-5, mx.float16: 1e-3, mx.bfloat16: 1e-2}[result_dtype]
            np.testing.assert_allclose(packed, generic, rtol=rtol, atol=2e-6)

    @pytest.mark.parametrize(
        ("n_hk", "n_hv", "d_v"),
        [(1, 2, 24), pytest.param(2, 6, 128, id="gqa3-fp32")],
    )
    def test_packed_prefill_continuation_and_deferred_decode_match_full_sequence(
        self,
        n_hk: int,
        n_hv: int,
        d_v: int,
    ) -> None:
        """Reordered prefill publishes prior updates; decode consumes compact state."""
        _require_metal()
        mx.random.seed(8)
        d_k = 128
        q, k = [mx.random.normal((1, 12, n_hk, d_k)) for _ in range(2)]
        q = q / mx.linalg.norm(q, axis=-1, keepdims=True) / d_k**0.5
        k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
        v = mx.random.normal((1, 12, n_hv, d_v)) * 0.1
        g = mx.random.uniform(0.5, 1.0, (1, 12, n_hv))
        beta = mx.random.uniform(0.0, 1.0, g.shape)
        initial = np.random.default_rng(8).normal(0, 0.1, (4, n_hv, d_v, d_k))
        initial = initial.astype(np.float32)
        full_cache, chunked_cache = [
            _make_state_cache(
                max_seqs=4, num_v_heads=n_hv, value_head_dim=d_v, key_head_dim=d_k
            )
            for _ in range(2)
        ]
        for cache in (full_cache, chunked_cache):
            cache.recurrent_states[0] = mx.array(initial)
        generic_kernels = GDNLazyKernels(enabled=True)
        _force_generic_prefill(generic_kernels)
        expected = generic_kernels.try_recurrent_prefill(
            _recurrent_prefill_request(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                cache=full_cache,
                slot_ids=[3, 1],
                cu_seqlens=[0, 7, 12],
            )
        )
        assert expected is not None
        mx.eval(expected, full_cache.recurrent_states[0])
        kernels = GDNLazyKernels(enabled=True)
        for indices, slots, lengths in (
            ([0, 1, 7, 8], [3, 1], [0, 2, 4]),
            ([9, 10, 2, 3, 4, 5], [1, 3], [0, 2, 6]),
            ([11, 6], [1, 3], None),
        ):
            ids = mx.array(indices, dtype=mx.int32)
            inputs = {
                "q": q[:, ids],
                "k": k[:, ids],
                "v": v[:, ids],
                "g": g[:, ids],
                "beta": beta[:, ids],
                "cache": chunked_cache,
                "slot_ids": slots,
            }
            if lengths is not None:
                actual = kernels.try_recurrent_prefill(
                    _recurrent_prefill_request(
                        **inputs, cu_seqlens=lengths, defer_state_scatter=True
                    )
                )
            else:
                actual = kernels.try_recurrent_decode(_recurrent_request(**inputs))
            assert actual is not None
            assert chunked_cache.pending_recurrent_state(0, slots) is not None
            np.testing.assert_allclose(
                np.array(actual), np.array(expected[ids]), rtol=2e-5, atol=2e-6
            )
            if slots == [3, 1]:
                np.testing.assert_array_equal(
                    np.array(chunked_cache.recurrent_states[0]), initial
                )
        chunked_cache.apply_pending_recurrent_state(0)
        np.testing.assert_allclose(
            np.array(chunked_cache.recurrent_states[0]),
            np.array(full_cache.recurrent_states[0]),
            rtol=2e-5,
            atol=2e-6,
        )
        np.testing.assert_array_equal(
            np.array(chunked_cache.recurrent_states[0][mx.array([0, 2])]),
            initial[[0, 2]],
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
        # Decode consumed the prefill's compact update in place of the pool and
        # replaced it with its own, so the pool is never touched in between.
        pending = cache.pending_recurrent_state(0, slot_ids)
        assert pending is not None
        mx.eval(pending)
        np.testing.assert_array_equal(np.array(pending), np.full((2, 2, 3, 32), 7))
        assert decode_kernel.state_input is not None
        assert decode_kernel.state_input.shape == (2, 2, 3, 32)
        cache.apply_pending_recurrent_state(0)
        mx.eval(cache.recurrent_states[0], decode_kernel.slot_mapping)
        np.testing.assert_array_equal(np.array(decode_kernel.slot_mapping), [0, 1])
        expected_state = np.array(initial_state)
        expected_state[slot_ids] = 7
        np.testing.assert_array_equal(
            np.array(cache.recurrent_states[0]), expected_state
        )

    def test_deferred_prefill_does_not_materialize_outputs(
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
        assert not eval_args
        assert cache.pending_recurrent_state(0, slot_ids) is not None
        np.testing.assert_array_equal(
            np.array(cache.recurrent_states[0]), np.array(initial_state)
        )

    def test_scattered_prefill_does_not_force_materialization(
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
        assert not contiguous_args
        assert not eval_args
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

    @pytest.mark.parametrize(
        ("d_k", "d_v", "n_hk", "n_hv", "cu_seqlens"),
        [
            (32, 4, 1, 1, [0, 2, 5]),
            (64, 7, 1, 1, [0, 1, 3]),
            (256, 7, 1, 1, [0, 1, 3]),
            (128, 24, 1, 1, [0, 1, 4]),
            pytest.param(128, 128, 2, 6, [0, 1, 4], id="gqa3-fp32"),
        ],
    )
    def test_matches_cpp_recurrent_path_for_prefill(
        self,
        d_k: int,
        d_v: int,
        n_hk: int,
        n_hv: int,
        cu_seqlens: list[int],
    ) -> None:
        # Arrange
        _require_metal()
        mx.random.seed(2)
        total_tokens = cu_seqlens[-1]
        slot_ids = [1, 0]
        cache_lazy = _make_state_cache(
            num_v_heads=n_hv, value_head_dim=d_v, key_head_dim=d_k
        )
        cache_cpp = _make_state_cache(
            num_v_heads=n_hv, value_head_dim=d_v, key_head_dim=d_k
        )
        initial_state = np.array(mx.random.normal(cache_lazy.recurrent_states[0].shape))
        cache_lazy.recurrent_states[0] = mx.array(initial_state)
        cache_cpp.recurrent_states[0] = mx.array(initial_state)

        q = mx.random.normal((1, total_tokens, n_hk, d_k))
        k = mx.random.normal((1, total_tokens, n_hk, d_k))
        q = q / mx.linalg.norm(q, axis=-1, keepdims=True) / d_k**0.5
        k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
        v = mx.random.normal((1, total_tokens, n_hv, d_v))
        g = mx.random.uniform(0.5, 1.0, (1, total_tokens, n_hv))
        beta = mx.random.uniform(0.0, 1.0, g.shape)
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
        mx.eval(
            q_flat,
            k_flat,
            v_flat,
            g_flat,
            beta_flat,
            cache_cpp.recurrent_states[0],
        )
        cpp_out, cpp_state = _get_native_ops_or_skip().gdn_linear_attention(
            q_flat,
            k_flat,
            v_flat,
            g_flat,
            beta_flat,
            cache_cpp.recurrent_states[0],
            mx.array(cu_seqlens, dtype=mx.int32),
            mx.array(slot_ids, dtype=mx.int32),
            n_hk,
            n_hv,
            d_k,
            d_v,
        )
        cache_cpp.store_recurrent_state(0, cpp_state)
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

    @pytest.mark.parametrize(
        ("d_k", "d_v", "compute_dtype"),
        [(32, 4, None), (128, 128, mx.float32)],
    )
    def test_prefill_uses_requested_output_dtype_without_losing_fallback_parity(
        self,
        d_k: int,
        d_v: int,
        compute_dtype: mx.Dtype | None,
    ) -> None:
        # Arrange
        _require_metal()
        mx.random.seed(4)
        total_tokens = 5
        n_hk = 1
        n_hv = 1
        slot_ids = [1, 0]
        cu_seqlens = [0, 2, 5]
        cache_lazy = _make_state_cache(value_head_dim=d_v, key_head_dim=d_k)
        cache_cpp = _make_state_cache(value_head_dim=d_v, key_head_dim=d_k)
        initial_state = np.array(
            mx.random.normal(cache_lazy.recurrent_states[0].shape) * 0.01
        )
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
            compute_dtype=compute_dtype,
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
        mx.eval(
            q_flat,
            k_flat,
            v_flat,
            g_flat,
            beta_flat,
            cache_cpp.recurrent_states[0],
        )
        cpp_out, cpp_state = _get_native_ops_or_skip().gdn_linear_attention(
            q_flat,
            k_flat,
            v_flat,
            g_flat,
            beta_flat,
            cache_cpp.recurrent_states[0],
            mx.array(cu_seqlens, dtype=mx.int32),
            mx.array(slot_ids, dtype=mx.int32),
            n_hk,
            n_hv,
            d_k,
            d_v,
        )
        cache_cpp.store_recurrent_state(0, cpp_state)
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


class TestGDNPrecision:
    @pytest.mark.parametrize(
        "dtype,boundary_values",
        [
            (mx.float16, False),
            (mx.bfloat16, False),
            (mx.float32, False),
            (mx.float16, True),
        ],
        ids=["fp16", "bf16", "fp32", "fp16-boundaries"],
    )
    @pytest.mark.parametrize("prefill", [False, True])
    def test_conv_matches_native_activation_rounding(
        self, dtype: mx.Dtype, boundary_values: bool, prefill: bool
    ) -> None:
        mx.random.seed(314)
        channels, tokens = (3 if boundary_values else 8), (2 if prefill else 1)
        cache = _make_state_cache(
            max_seqs=1, conv_kernel_dim=4, conv_dim=channels, dtype=dtype
        )
        conv = nn.Conv1d(channels, channels, 4, groups=channels, bias=False)
        conv.weight = mx.random.normal(conv.weight.shape).astype(dtype)
        cache.conv_states[0] = mx.random.normal(cache.conv_states[0].shape).astype(
            dtype
        )
        x = mx.random.normal((1, tokens, channels)).astype(dtype)
        if boundary_values:
            # Exact accumulators -70016, +70016 and -20 exercise FP16
            # convolution overflow and sigmoid underflow in the MLX reference.
            conv.weight = mx.zeros_like(conv.weight)
            conv.weight[:, -1, 0] = 2
            cache.conv_states[0] = mx.zeros_like(cache.conv_states[0])
            x = mx.broadcast_to(mx.array([-35008, 35008, -10], dtype=dtype), x.shape)
        inner = SimpleNamespace(conv_kernel_size=4, conv1d=conv)
        expected = nn.silu(conv(mx.concatenate([cache.conv_states[0], x], axis=1)))
        mx.eval(expected)
        kernels = GDNLazyKernels(enabled=True)
        if prefill:
            actual = kernels.try_conv_prefill(x, inner, cache, 0, [0], [0, tokens])
        else:
            actual = kernels.try_conv_decode(x, inner, cache, 0, [0])
        assert actual is not None
        assert actual.dtype == dtype
        got = np.array(actual.astype(mx.float32))
        ref = np.array(expected.astype(mx.float32))
        np.testing.assert_array_equal(got, ref)
        if boundary_values:
            assert np.isnan(got[..., 0]).all()
            assert (got[..., 1] == np.inf).all()
            assert (got[..., 2] == 0).all() and np.signbit(got[..., 2]).all()
            assert np.signbit(ref[..., 2]).all()

    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
    @pytest.mark.parametrize("half_state", [False, True])
    def test_decay_gate_stays_fp32(self, dtype: mx.Dtype, half_state: bool) -> None:
        inner = _TinyGDNInner()
        cache = _make_state_cache(
            conv_dim=inner.conv_dim,
            dtype=dtype,
            recurrent_dtype=dtype if half_state else mx.float32,
        )
        wrapper = GDNPagedAttentionWrapper(inner, 0, 0, cache)
        state = _GDNForwardState(
            mx.zeros((1, 1, inner.conv_dim), dtype=dtype), [0, 1], 1, 1, [0], 1
        )
        a = mx.full((1, 1, 1), -9.25, dtype=dtype)
        g, beta = wrapper._compute_gates(a, mx.zeros_like(a), state)
        assert g.dtype == mx.float32 and beta.dtype == dtype
        assert g.item() < 1.0
        assert g.astype(dtype).item() == 1.0  # The old cast erased this decay.

    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
    @pytest.mark.parametrize(
        "route,half_state",
        [
            ("decode", False),
            ("prefill", False),
            ("prefill_fp32", False),
            ("decode", True),
            ("prefill", True),
        ],
    )
    def test_recurrent_decay_is_not_rounded_to_one(
        self, dtype: mx.Dtype, route: str, half_state: bool
    ) -> None:
        tokens = 1 if route == "decode" else 256
        state_dtype = dtype if half_state else mx.float32
        cache = _make_state_cache(max_seqs=1, dtype=dtype, recurrent_dtype=state_dtype)
        cache.recurrent_states[0] = mx.ones_like(cache.recurrent_states[0])
        q = mx.zeros((1, tokens, 1, 32), dtype=dtype)
        # Decode stores state each step, so a half cache needs a representable
        # decay. Prefill accumulates all 256 steps in float before storing.
        decay = 0.99 if half_state and route == "decode" else 0.9999
        common = {
            "q": q,
            "k": mx.zeros_like(q),
            "v": mx.zeros((1, tokens, 1, 4), dtype=dtype),
            "g": mx.full((1, tokens, 1), decay, dtype=mx.float32),
            "beta": mx.zeros((1, tokens, 1), dtype=dtype),
            "cache": cache,
            "slot_ids": [0],
            "output_dtype": dtype,
        }
        kernels = GDNLazyKernels(enabled=True)
        if route == "decode":
            output = kernels.try_recurrent_decode(_recurrent_request(**common))
            updated = cache.pending_recurrent_state(0, [0])
        else:
            output = kernels.try_recurrent_prefill(
                _recurrent_prefill_request(
                    **common,
                    cu_seqlens=[0, tokens],
                    compute_dtype=mx.float32 if route == "prefill_fp32" else None,
                )
            )
            updated = cache.recurrent_states[0]
        assert output is not None and updated is not None
        assert output.dtype == dtype and updated.dtype == state_dtype
        expected = np.float32(1)
        for _ in range(tokens):
            expected *= np.float32(decay)
        expected = mx.array(expected).astype(state_dtype).astype(mx.float32).item()
        np.testing.assert_allclose(
            np.array(updated.astype(mx.float32)), expected, atol=1e-7, rtol=1e-6
        )
        assert expected < 1.0

    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
    def test_fallback_casts_fp32_gate_to_half_cache(self, dtype: mx.Dtype) -> None:
        _get_native_ops_or_skip()
        cache = _make_state_cache(max_seqs=1, dtype=dtype, recurrent_dtype=dtype)
        cache.recurrent_states[0] = mx.ones_like(cache.recurrent_states[0])
        wrapper = GDNPagedAttentionWrapper(_TinyGDNInner(), 0, 0, cache)
        q = mx.zeros((1, 1, 1, 32), dtype=dtype)
        state = _GDNForwardState(
            mx.zeros((1, 1, 68), dtype=dtype), [0, 1], 1, 1, [0], 1
        )
        out = wrapper._run_recurrent_fallback(
            q,
            mx.zeros_like(q),
            mx.zeros((1, 1, 1, 4), dtype=dtype),
            mx.full((1, 1, 1), 0.5, dtype=mx.float32),
            mx.zeros((1, 1, 1), dtype=dtype),
            state,
        )
        assert out.dtype == dtype
        assert cache.recurrent_states[0].dtype == dtype
        np.testing.assert_array_equal(
            np.array(cache.recurrent_states[0][0].astype(mx.float32)), 0.5
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
                conv_kernel_size=(cache.conv_states[0].shape[1] + 1),
                conv1d=_RaisingKernel(),
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


class TestNativeGDNRecurrent:
    @pytest.mark.parametrize("compiled", [False, True])
    def test_consecutive_recurrences_keep_distinct_outputs(self, compiled):
        _require_metal()
        ops = _get_native_ops_or_skip()

        def two_steps(q, v, g, beta, state, cu_seqlens, slots):
            first, state = ops.gdn_linear_attention(
                q, q, v, g, beta, state, cu_seqlens, slots, 1, 1, 32, 4
            )
            second, state = ops.gdn_linear_attention(
                q, q, v, g, beta, state, cu_seqlens, slots, 1, 1, 32, 4
            )
            return first, second, state

        run = mx.compile(two_steps) if compiled else two_steps
        first, second, state = run(
            mx.array([1.0] + [0.0] * 31).reshape(1, 1, 32),
            mx.full((1, 1, 4), 2, dtype=mx.float32),
            mx.ones((1, 1)),
            mx.full((1, 1), 0.5),
            mx.zeros((3, 1, 4, 32)),
            mx.array([0, 1], dtype=mx.int32),
            mx.array([2], dtype=mx.int32),
        )
        mx.eval(first, second, state)
        # The selected column follows s <- (s + 2) / 2 from an initial zero.
        np.testing.assert_array_equal(np.array(first), 1)
        np.testing.assert_array_equal(np.array(second), 1.5)
        expected = np.zeros((3, 1, 4, 32), dtype=np.float32)
        expected[2, ..., 0] = 1.5
        np.testing.assert_array_equal(np.array(state), expected)


class TestNativeGDNStateScatter:
    """The in-place row scatter that replaces MLX's whole-pool copy."""

    @staticmethod
    def _scatter_fn() -> Any:
        return _get_native_ops_or_skip().gdn_state_scatter

    @pytest.mark.parametrize(
        ("shape", "dtype", "slots"),
        [
            ((8, 3, 4), mx.float32, [0, 5, 2]),
            ((8, 1, 4, 32), mx.float32, [7, 1]),
            ((8, 3, 4), mx.float16, [3]),
            ((8, 3, 4), mx.bfloat16, [6, 0]),
            ((16, 2, 6144), mx.float32, [15, 4, 9]),
            # row lengths not divisible by 4 take the scalar kernel
            ((8, 3), mx.float32, [0, 5]),
            ((8, 7), mx.float16, [2]),
            ((8, 5, 3), mx.bfloat16, [6, 1]),
        ],
    )
    def test_matches_mlx_indexed_assignment(
        self, shape: tuple[int, ...], dtype: mx.Dtype, slots: list[int]
    ) -> None:
        scatter = self._scatter_fn()
        rng = np.random.default_rng(0)
        pool_np = rng.standard_normal(shape).astype(np.float32)
        rows_np = rng.standard_normal((len(slots), *shape[1:])).astype(np.float32)
        ids = mx.array(slots, dtype=mx.int32)

        reference = mx.array(pool_np).astype(dtype)
        reference[ids] = mx.array(rows_np).astype(dtype)
        mx.eval(reference)

        pool = mx.array(pool_np).astype(dtype)
        mx.eval(pool)
        out = scatter(pool, mx.array(rows_np).astype(dtype), ids)
        mx.eval(out)

        np.testing.assert_array_equal(
            np.array(out.astype(mx.float32)), np.array(reference.astype(mx.float32))
        )

    def test_writes_through_the_original_buffer(self) -> None:
        # The output aliases the pool buffer -- that is what avoids the copy,
        # and what makes the caller's rebind mandatory.
        scatter = self._scatter_fn()
        pool = mx.zeros((4, 2, 2), dtype=mx.float32)
        mx.eval(pool)
        out = scatter(
            pool, mx.ones((1, 2, 2), dtype=mx.float32), mx.array([2], mx.int32)
        )
        mx.eval(out)
        np.testing.assert_array_equal(np.array(pool[2]), 1.0)

    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
    @pytest.mark.parametrize("width", [5, 12])
    def test_paged_scatter_preserves_padding_and_untouched_slots(
        self, dtype, width
    ) -> None:
        ops = _get_native_ops_or_skip()
        page_stride = 4 * width
        backing = mx.full((2 * page_stride,), 99, dtype=dtype)
        pool = ops.as_strided(backing, (2, 3, width), (page_stride, width, 1))
        slots = mx.array([4, 1, 5], dtype=mx.int32)
        values = mx.arange(3 * width).reshape(3, width).astype(dtype)

        updated = ops.gdn_state_scatter(pool, values, slots, paged=True)
        mx.eval(updated)

        expected = np.full(2 * page_stride, 99, dtype=np.float32)
        for row, slot in enumerate([4, 1, 5]):
            start = (slot // 3) * page_stride + (slot % 3) * width
            expected[start : start + width] = np.arange(row * width, (row + 1) * width)
        np.testing.assert_array_equal(np.array(backing.astype(mx.float32)), expected)

    @pytest.mark.parametrize("dtype", [mx.uint8, mx.int8, mx.float16, mx.float32])
    @pytest.mark.parametrize(("dst_offset", "src_offset"), [(1, 0), (0, 1), (3, 1)])
    @pytest.mark.parametrize("zero", [False, True])
    def test_unaligned_views_preserve_bytes_outside_the_write(
        self, dtype, dst_offset, src_offset, zero
    ) -> None:
        ops = _get_native_ops_or_skip()
        backing = mx.full((80,), 99, dtype=dtype)
        source = mx.arange(16).astype(dtype)
        pool = ops.as_strided(backing, (4, 8), (16, 1), offset=dst_offset)
        rows = ops.as_strided(source, (1, 8), (8, 1), offset=src_offset)

        updated = ops.gdn_state_scatter(pool, rows, mx.array([2]), zero=zero)
        mx.eval(updated)

        expected = np.full(80, 99, dtype=np.float32)
        expected[32 + dst_offset : 40 + dst_offset] = (
            0 if zero else np.arange(src_offset, src_offset + 8)
        )
        np.testing.assert_array_equal(np.array(backing.astype(mx.float32)), expected)

    def test_empty_update_leaves_the_pool_alone(self) -> None:
        scatter = self._scatter_fn()
        pool = mx.ones((4, 2, 2), dtype=mx.float32)
        mx.eval(pool)
        out = scatter(
            pool,
            mx.zeros((0, 2, 2), dtype=mx.float32),
            mx.array([], dtype=mx.int32),
        )
        mx.eval(out)
        np.testing.assert_array_equal(np.array(out), 1.0)

    def test_rejects_mismatched_inputs(self) -> None:
        scatter = self._scatter_fn()
        pool = mx.zeros((4, 2, 2), dtype=mx.float32)
        mx.eval(pool)
        with pytest.raises(RuntimeError, match="dtypes differ"):
            scatter(
                pool,
                mx.ones((1, 2, 2), dtype=mx.float16),
                mx.array([0], dtype=mx.int32),
            )
        with pytest.raises(RuntimeError, match="dst_ids must be int32"):
            scatter(
                pool,
                mx.ones((1, 2, 2), dtype=mx.float32),
                mx.array([0], dtype=mx.int64),
            )
        with pytest.raises(RuntimeError, match="one dst id per src row"):
            scatter(
                pool,
                mx.ones((2, 2, 2), dtype=mx.float32),
                mx.array([0], dtype=mx.int32),
            )
        with pytest.raises(TypeError):
            scatter(pool, mx.ones((1, 2, 2), dtype=mx.float32), [0])
        with pytest.raises(RuntimeError, match="row shape does not match"):
            scatter(
                pool,
                mx.ones((1, 3, 2), dtype=mx.float32),
                mx.array([0], dtype=mx.int32),
            )
        with pytest.raises(RuntimeError, match="row shape does not match"):
            scatter(
                pool,
                mx.array(1, dtype=mx.float32),
                mx.array([0], dtype=mx.int32),
            )

    def test_rejects_unsupported_pool_dtype(self) -> None:
        scatter = self._scatter_fn()
        with pytest.raises(RuntimeError, match="pool dtype must be"):
            scatter(
                mx.zeros((4, 2), dtype=mx.int32),
                mx.ones((1, 2), dtype=mx.int32),
                mx.array([0], dtype=mx.int32),
            )

    def test_materializes_noncontiguous_source_and_ids(self) -> None:
        scatter = self._scatter_fn()
        pool = mx.zeros((4, 2), dtype=mx.float32)
        rows = mx.arange(8, dtype=mx.float32).reshape(2, 4)[:, ::2]
        ids = mx.array([0, 3, 1, 3], dtype=mx.int32)[::2]

        updated = scatter(pool, rows, ids)
        mx.eval(updated)

        np.testing.assert_array_equal(np.array(updated[:2]), [[0.0, 2.0], [4.0, 6.0]])

    def test_materializes_noncontiguous_pool(self) -> None:
        scatter = self._scatter_fn()
        base = mx.arange(8, dtype=mx.float32).reshape(2, 4)
        mx.eval(base)
        pool = base.T
        updated = scatter(
            pool,
            mx.array([[9.0, 10.0]], dtype=mx.float32),
            mx.array([1], dtype=mx.int32),
        )
        mx.eval(updated)

        np.testing.assert_array_equal(
            np.array(updated),
            [[0.0, 4.0], [9.0, 10.0], [2.0, 6.0], [3.0, 7.0]],
        )

    def test_converts_source_dtype_like_mlx_does(self) -> None:
        # MLX's indexed assignment converts the source implicitly, so callers
        # pass int literals into fp32 pools; the primitive demands an exact
        # match, so the cache helper must convert before dispatching.
        self._scatter_fn()  # skip when the primitive is unavailable
        cache = _make_state_cache(max_seqs=4, conv_kernel_dim=3, conv_dim=4)
        pool = cache.conv_states[0]
        rows = mx.full((1, 2, 4), 5)  # int32, unlike the fp32 pool
        assert rows.dtype != pool.dtype

        updated = cache._scatter_rows(pool, rows, mx.array([2], dtype=mx.int32))
        mx.eval(updated)

        np.testing.assert_array_equal(np.array(updated[2]), 5.0)
