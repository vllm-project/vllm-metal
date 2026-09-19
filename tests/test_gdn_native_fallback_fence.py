# SPDX-License-Identifier: Apache-2.0
"""Exercise native GDN graph dependencies with real MLX/Metal consumers."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)

from vllm_metal.attention.caches.state_cache import PagedStateCache
from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.attention.impls.linear import GDNPagedAttentionWrapper, _GDNForwardState
from vllm_metal.metal import get_ops


@pytest.fixture(
    params=[("float16", "float16"), ("float16", "float32"), ("float32", "float32")],
    ids=["fp16-state-fp16", "fp16-state-fp32", "fp32-state-fp32"],
)
def fallback_case(request):
    if not mx.metal.is_available():
        pytest.skip("MLX Metal is not available")
    get_ops()  # Require the real extension, never a queued-write stand-in.
    input_name, state_name = request.param
    input_dtype = getattr(mx, input_name)
    torch_input, torch_state = getattr(torch, input_name), getattr(torch, state_name)
    page_bytes, num_blocks = 1024, 4
    spec = MambaSpec(
        block_size=16,
        shapes=((2, 4), (1, 4, 32)),
        dtypes=(torch_input, torch_state),
        page_size_padded=page_bytes,
        mamba_cache_mode="align",
    )
    storage = KVCacheStorage(
        KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["s0", "s1"], kv_cache_spec=spec)
            ],
            kv_cache_tensors=[
                KVCacheTensor(
                    size=2 * num_blocks * page_bytes,
                    layers=["s0", "s1"],
                    layer_stride=num_blocks * page_bytes,
                    block_stride=page_bytes,
                )
            ],
            kv_cache_layout="LBNHC",
        )
    )
    cache = PagedStateCache(storage.state_views(["s0", "s1"]))
    inner = SimpleNamespace(num_k_heads=1, num_v_heads=1, head_k_dim=32, head_v_dim=4)
    wrapper = GDNPagedAttentionWrapper(
        inner, layer_idx=0, cache_idx=0, state_cache=cache
    )
    mx.eval(storage.buffer)
    mx.synchronize()
    yield SimpleNamespace(
        wrapper=wrapper,
        cache=cache,
        storage=storage,
        input_dtype=input_dtype,
        torch_state=torch_state,
        state_offset=8 * torch.empty((), dtype=torch_input).element_size(),
    )
    # Cleanup only: no assertion may rely on this synchronization.
    mx.synchronize()


def _step(case, slots=(2, 0)):
    count = len(slots)
    unit = np.zeros((1, count, 1, 32), dtype=np.float32)
    unit[..., 0] = 1
    q = mx.array(unit, dtype=case.input_dtype)
    # With g=1 and beta=1/2, each selected state's first column follows
    # s <- (s + 2)/2; the output equals s exactly in both tested dtypes.
    return case.wrapper._run_recurrent_fallback(
        q,
        q,
        mx.full((1, count, 1, 4), 2, dtype=case.input_dtype),
        mx.ones((1, count, 1), dtype=case.input_dtype),
        mx.full((1, count, 1), 0.5, dtype=case.input_dtype),
        _GDNForwardState(
            x=mx.zeros((1, count, 4), dtype=case.input_dtype),
            cu_seqlens=list(range(count + 1)),
            num_requests=count,
            total_tokens=count,
            slot_ids=list(slots),
            num_decode_requests=count,
        ),
    )


def _expected_pool(values):
    expected = np.zeros((4, 1, 4, 32), dtype=np.float32)
    for slot, value in values.items():
        expected[slot, ..., 0] = value
    return expected


def _host_state(case):
    # Called after explicit evaluation: read the original upstream-owned
    # strided backing without adding a GPU consumer to mask a missing edge.
    pages = case.storage.tensors["s0"].squeeze(dim=(1, 2))
    nbytes = 128 * torch.empty((), dtype=case.torch_state).element_size()
    return (
        pages[:, case.state_offset : case.state_offset + nbytes]
        .view(case.torch_state)
        .reshape(4, 1, 4, 32)
        .numpy()
        .copy()
    )


def test_fallback_output_eval_completes_native_write(fallback_case):
    output = _step(fallback_case)
    mx.eval(output)
    np.testing.assert_array_equal(np.array(output), 1)


def test_fallback_shared_buffer_eval_completes_native_state_write(fallback_case):
    _step(fallback_case)
    mx.eval(fallback_case.storage.buffer)
    np.testing.assert_array_equal(
        _host_state(fallback_case), _expected_pool({2: 1, 0: 1})
    )
    assert not fallback_case.storage.tensors["s1"].any()


def test_fallback_builds_graph_without_host_wait(fallback_case, monkeypatch):
    with monkeypatch.context() as guard:

        def no_host_wait(*args, **kwargs):
            pytest.fail("fallback construction must not evaluate or synchronize")

        guard.setattr(mx, "eval", no_host_wait)
        guard.setattr(mx, "synchronize", no_host_wait)
        output = _step(fallback_case)
    mx.eval(output)
    np.testing.assert_array_equal(np.array(output), 1)


def test_fallback_custom_producer_feeds_other_stream(fallback_case):
    producer = mx.new_stream(mx.gpu)
    consumer = mx.new_stream(mx.gpu)
    with mx.stream(producer):
        output = _step(fallback_case)
    with mx.stream(consumer):
        observed = output + 0
        state = fallback_case.cache.recurrent_states[0] + 0
        mx.eval(observed, state)
    np.testing.assert_array_equal(np.array(observed), 1)
    np.testing.assert_array_equal(np.array(state), _expected_pool({2: 1, 0: 1}))


@pytest.mark.parametrize("compiled", [False, True])
def test_native_recurrences_retain_distinct_outputs(fallback_case, compiled):
    ops = get_ops()

    def two_steps(q, v, g, beta, state, cu_seqlens, slots):
        first, state = ops.gdn_linear_attention(
            q, q, v, g, beta, state, cu_seqlens, slots, 1, 1, 32, 4
        )
        second, state = ops.gdn_linear_attention(
            q, q, v, g, beta, state, cu_seqlens, slots, 1, 1, 32, 4
        )
        return first, second, state

    run = mx.compile(two_steps) if compiled else two_steps
    state = fallback_case.cache.recurrent_states[0]
    unit = np.zeros((2, 1, 32), dtype=np.float32)
    unit[..., 0] = 1
    first, second, state = run(
        mx.array(unit, dtype=state.dtype),
        mx.full((2, 1, 4), 2, dtype=state.dtype),
        mx.ones((2, 1), dtype=state.dtype),
        mx.full((2, 1), 0.5, dtype=state.dtype),
        state,
        mx.array([0, 1, 2], dtype=mx.int32),
        mx.array([2, 0], dtype=mx.int32),
    )
    fallback_case.cache.store_recurrent_state(0, state)
    mx.eval(first, second, fallback_case.storage.buffer)
    np.testing.assert_array_equal(np.array(first), 1)
    np.testing.assert_array_equal(np.array(second), 1.5)
    np.testing.assert_array_equal(
        _host_state(fallback_case), _expected_pool({2: 1.5, 0: 1.5})
    )


@pytest.mark.parametrize("different_stream", [False, True])
def test_consecutive_fallback_updates_feed_mlx_consumers(
    fallback_case, different_stream
):
    consumer = mx.new_stream(mx.gpu) if different_stream else mx.default_stream(mx.gpu)
    outputs = [_step(fallback_case) for _ in range(3)]
    with mx.stream(consumer):
        observed_outputs = mx.stack(outputs)
        observed_state = fallback_case.cache.recurrent_states[0] + 0
        mx.eval(observed_outputs, observed_state)
    expected_outputs = np.broadcast_to(
        np.array([1, 1.5, 1.75])[:, None, None, None], (3, 2, 1, 4)
    )
    np.testing.assert_array_equal(np.array(observed_outputs), expected_outputs)
    np.testing.assert_array_equal(
        np.array(observed_state), _expected_pool({2: 1.75, 0: 1.75})
    )


@pytest.mark.parametrize("different_stream", [False, True])
def test_fallback_state_survives_copy_and_reset(fallback_case, different_stream):
    consumer = mx.new_stream(mx.gpu) if different_stream else mx.default_stream(mx.gpu)
    _step(fallback_case, slots=(2,))
    with mx.stream(consumer):
        fallback_case.cache.copy_slots([2], [1], [0])
        fallback_case.cache.zero_slots([2], [0])
    output = _step(fallback_case, slots=(1, 2))
    with mx.stream(consumer):
        observed_output = output + 0
        observed_state = fallback_case.cache.recurrent_states[0] + 0
        mx.eval(observed_output, observed_state)
    np.testing.assert_array_equal(np.array(observed_output)[:, 0, 0], [1.5, 1])
    np.testing.assert_array_equal(
        np.array(observed_state), _expected_pool({1: 1.5, 2: 1})
    )
