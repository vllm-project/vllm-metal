# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for the native PyTorch SDPA attention backend."""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional
from vllm.v1.attention.backend import CommonAttentionMetadata

from vllm_metal.pytorch_backend.attention import (
    TorchAttentionBackend,
    TorchAttentionImpl,
)


def _impl(*, heads: int = 2, kv_heads: int = 2, window: int | None = None):
    return TorchAttentionImpl(
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_size=4,
        scale=0.5,
        sliding_window=window,
    )


def _metadata(
    query_lens: list[int],
    seq_lens: list[int],
    block_table: list[list[int]],
    slots: list[int],
    *,
    num_actual_tokens: int | None = None,
) -> CommonAttentionMetadata:
    starts = [0]
    for length in query_lens:
        starts.append(starts[-1] + length)
    query_start_loc = torch.tensor(starts, dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        num_reqs=len(query_lens),
        num_actual_tokens=sum(query_lens)
        if num_actual_tokens is None
        else num_actual_tokens,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        block_table_tensor=torch.tensor(block_table, dtype=torch.int32),
        slot_mapping=torch.tensor(slots, dtype=torch.int64),
    )


def _forward(impl, q, k, v, cache, metadata, output_rows=None):
    rows = q.shape[0] if output_rows is None else output_rows
    output = torch.full(
        (rows, impl.num_heads, impl.head_size), math.nan, device=q.device
    )
    return impl.forward(None, q, k, v, cache, metadata, output)


def test_full_prefill_matches_uncached_sdpa() -> None:
    torch.manual_seed(4)
    impl = _impl()
    q = torch.randn(4, 2, 4)
    k = torch.randn(4, 2, 4)
    v = torch.randn(4, 2, 4)
    cache = torch.zeros(TorchAttentionBackend.get_kv_cache_shape(2, 2, 2, 4))
    metadata = _metadata([4], [4], [[0, 1]], [0, 1, 2, 3])

    actual = _forward(impl, q, k, v, cache, metadata)
    expected = functional.scaled_dot_product_attention(
        q.transpose(0, 1),
        k.transpose(0, 1),
        v.transpose(0, 1),
        is_causal=True,
        scale=0.5,
    ).transpose(0, 1)

    torch.testing.assert_close(actual, expected)


def test_chunked_prefill_uses_causal_offset_and_physical_blocks() -> None:
    torch.manual_seed(5)
    impl = _impl()
    all_q = torch.randn(5, 2, 4)
    all_k = torch.randn(5, 2, 4)
    all_v = torch.randn(5, 2, 4)
    cache = torch.zeros(TorchAttentionBackend.get_kv_cache_shape(4, 2, 2, 4))

    _forward(
        impl,
        all_q[:3],
        all_k[:3],
        all_v[:3],
        cache,
        _metadata([3], [3], [[2, 0, 3]], [4, 5, 0]),
    )
    actual = _forward(
        impl,
        all_q[3:],
        all_k[3:],
        all_v[3:],
        cache,
        _metadata([2], [5], [[2, 0, 3]], [1, 6]),
    )
    full = functional.scaled_dot_product_attention(
        all_q.transpose(0, 1),
        all_k.transpose(0, 1),
        all_v.transpose(0, 1),
        is_causal=True,
        scale=0.5,
    ).transpose(0, 1)

    torch.testing.assert_close(actual, full[3:])


def test_decode_reuses_prefix_cache_and_supports_gqa() -> None:
    torch.manual_seed(6)
    impl = _impl(heads=4, kv_heads=2)
    q = torch.randn(4, 4, 4)
    k = torch.randn(4, 2, 4)
    v = torch.randn(4, 2, 4)
    cache = torch.zeros(TorchAttentionBackend.get_kv_cache_shape(2, 2, 2, 4))

    _forward(impl, q[:3], k[:3], v[:3], cache, _metadata([3], [3], [[0, 1]], [0, 1, 2]))
    actual = _forward(
        impl, q[3:], k[3:], v[3:], cache, _metadata([1], [4], [[0, 1]], [3])
    )
    repeated_k = k.repeat_interleave(2, dim=1)
    repeated_v = v.repeat_interleave(2, dim=1)
    expected = functional.scaled_dot_product_attention(
        q[3:].transpose(0, 1),
        repeated_k.transpose(0, 1),
        repeated_v.transpose(0, 1),
        scale=0.5,
    ).transpose(0, 1)

    torch.testing.assert_close(actual, expected)


def test_padding_slots_are_ignored_and_output_padding_is_untouched() -> None:
    torch.manual_seed(7)
    impl = _impl()
    q = torch.randn(3, 2, 4)
    k = torch.randn(3, 2, 4)
    v = torch.randn(3, 2, 4)
    cache = torch.zeros(TorchAttentionBackend.get_kv_cache_shape(1, 2, 2, 4))
    metadata = _metadata([2], [2], [[0]], [0, 1, -1], num_actual_tokens=2)

    actual = _forward(impl, q, k, v, cache, metadata, output_rows=3)

    assert torch.isnan(actual[2]).all()
    torch.testing.assert_close(cache[0, 0], k[:2])
    torch.testing.assert_close(cache[1, 0], v[:2])


def test_negative_slot_inside_active_cache_update_is_ignored() -> None:
    key = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    value = key + 100
    key_cache = torch.zeros(1, 2, 2, 4)
    value_cache = torch.zeros_like(key_cache)

    TorchAttentionImpl._write_cache(
        key,
        value,
        key_cache,
        value_cache,
        torch.tensor([0, -1, 1]),
        3,
    )

    torch.testing.assert_close(key_cache[0, 0], key[0])
    torch.testing.assert_close(key_cache[0, 1], key[2])
    torch.testing.assert_close(value_cache[0, 0], value[0])
    torch.testing.assert_close(value_cache[0, 1], value[2])


def test_current_main_packed_strided_cache_updates_backing_storage() -> None:
    torch.manual_seed(8)
    impl = _impl()
    # Current main exposes each layer as logical [block, head, token, 2 * dim].
    # A transposed backing makes this view non-contiguous, like layout selection.
    backing = torch.zeros(2, 2, 2, 8)
    cache = backing.permute(0, 2, 1, 3)
    assert not cache.is_contiguous()
    q = torch.randn(2, 2, 4)
    k = torch.randn(2, 2, 4)
    v = torch.randn(2, 2, 4)

    actual = _forward(impl, q, k, v, cache, _metadata([2], [2], [[0]], [0, 1]))
    expected = functional.scaled_dot_product_attention(
        q.transpose(0, 1),
        k.transpose(0, 1),
        v.transpose(0, 1),
        is_causal=True,
        scale=0.5,
    ).transpose(0, 1)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(cache[0, :, :, :4], k.permute(1, 0, 2))
    torch.testing.assert_close(cache[0, :, :, 4:], v.permute(1, 0, 2))
    assert torch.count_nonzero(backing).item() > 0


def test_sliding_window_does_not_read_freed_prefix_blocks() -> None:
    torch.manual_seed(9)
    impl = _impl(window=2)
    q = torch.randn(1, 2, 4)
    k = torch.randn(1, 2, 4)
    v = torch.randn(1, 2, 4)
    cache = torch.randn(TorchAttentionBackend.get_kv_cache_shape(2, 2, 2, 4))
    # At sequence length four, a two-token window only needs logical block 1.
    # The scheduler may have released logical block 0 and left a null entry.
    metadata = _metadata([1], [4], [[-1, 1]], [3])

    actual = _forward(impl, q, k, v, cache, metadata)

    assert torch.isfinite(actual).all()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"alibi_slopes": [1.0, 1.0]}, "ALiBi"),
        ({"logits_soft_cap": 30.0}, "soft-cap"),
        ({"kv_cache_dtype": "fp8"}, "Quantized KV"),
    ],
)
def test_unsupported_modifiers_fail_loudly(kwargs, match) -> None:
    with pytest.raises(NotImplementedError, match=match):
        TorchAttentionImpl(2, 4, 0.5, num_kv_heads=2, **kwargs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_tiny_mps_decode_matches_cpu(dtype: torch.dtype) -> None:
    torch.manual_seed(10)
    device = torch.device("mps")
    impl = _impl()
    q_cpu = torch.randn(1, 2, 4, dtype=dtype)
    k_cpu = torch.randn(1, 2, 4, dtype=dtype)
    v_cpu = torch.randn(1, 2, 4, dtype=dtype)
    cpu_cache = torch.zeros(
        TorchAttentionBackend.get_kv_cache_shape(1, 2, 2, 4), dtype=dtype
    )
    expected = _forward(
        impl, q_cpu, k_cpu, v_cpu, cpu_cache, _metadata([1], [1], [[0]], [0])
    )
    q = q_cpu.to(device)
    k = k_cpu.to(device)
    v = v_cpu.to(device)
    cache = torch.zeros(
        TorchAttentionBackend.get_kv_cache_shape(1, 2, 2, 4),
        dtype=dtype,
        device=device,
    )
    metadata = _metadata([1], [1], [[0]], [0])

    output = _forward(impl, q, k, v, cache, metadata)

    assert output.device.type == "mps"
    assert torch.isfinite(output).all().item()
    torch.testing.assert_close(
        output.cpu(), expected, rtol=2e-2, atol=2e-2, check_dtype=True
    )
