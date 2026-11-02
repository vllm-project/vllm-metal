# SPDX-License-Identifier: Apache-2.0
"""Numerical coverage for the TurboQuant GQA decode producers.

Default routing admits TurboQuant decode under the GQA work budget at P256
(test_gqa_decode_routing.py). These tests force each producer through the
private test entry and compare it with the established TurboQuant kernels,
reached with GQA disabled, and with an FP32 reference over the dequantized
cache.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from tests.gqa_test_utils import (
    _assert_close,
    _dispatch_family,
    _grouped_paged_reference,
    _interleaved_table,
)
from vllm_metal.attention.caches.turboquant import (
    get_v_centroids,
    turbo_quant_decode,
    turbo_quant_encode,
)
from vllm_metal.metal import get_ops

GQA_CONFIG = get_ops()._gqa_decode_config_for_test()
GQA_GEOMETRIES = tuple(tuple(row) for row in GQA_CONFIG["geometries"])
GQA_PARTITIONS = tuple(sorted(GQA_CONFIG["partitions"]))
assert GQA_GEOMETRIES and GQA_PARTITIONS

# Signed int8, unsigned uint8, and packed 4-, 5- (byte-straddling) and 2-bit K.
K_QUANTS = ("q8_0", "uint8", "q4_0", "q5_0", "int2")
V_BITS = (2, 3, 4, 5, 8)


@pytest.fixture(scope="module", autouse=True)
def _enable_dispatch_diagnostics():
    ops = get_ops()
    previous = ops._set_paged_dispatch_diagnostics(True)
    try:
        yield
    finally:
        mx.synchronize()
        ops._set_paged_dispatch_diagnostics(previous)


def _run(
    *,
    kv_lens: list[int],
    geometry: tuple[int, int, int, int],
    partition: int,
    dtype: mx.Dtype,
    k_quant: str = "q8_0",
    v_bits: int = 3,
    seed: int = 0,
    length_plan: bool = False,
) -> tuple[mx.array, mx.array, mx.array]:
    """Return (GQA producer, established TurboQuant kernel, FP32 reference)."""
    query_heads, kv_heads, head_size, block_size = geometry
    mx.random.seed(seed)
    pages_per_seq = -(-max(kv_lens) // block_size)
    tables = np.array(
        [
            [page + s * pages_per_seq for page in _interleaved_table(pages_per_seq)]
            for s in range(len(kv_lens))
        ],
        dtype=np.int32,
    )
    shape = (tables.max() + 2, block_size, kv_heads, head_size)
    keys = mx.random.normal(shape).astype(dtype)
    values = mx.random.normal(shape).astype(dtype)
    query = mx.random.normal((len(kv_lens), query_heads, head_size)).astype(dtype)
    (k_cache, k_scale, k_zero), (v_cache, v_scale) = turbo_quant_encode(
        keys, values, k_quant, value_bits=v_bits
    )
    k_ref, v_ref = turbo_quant_decode(
        (k_cache, k_scale, k_zero),
        (v_cache, v_scale),
        output_dtype=dtype,
        key_quant_type=k_quant,
        value_bits=v_bits,
    )
    centroids = get_v_centroids(v_bits)
    block_tables = mx.array(tables)
    lengths = mx.array(kv_lens, dtype=mx.int32)
    scale = head_size**-0.5
    mx.eval(query, k_cache, k_scale, k_zero, v_cache, v_scale, k_ref, v_ref)

    ops = get_ops()
    plan = ops.gqa_decode_length_plan(kv_lens) if length_plan else None
    out = ops._gqa_turboquant_paged_attention_for_test(
        query,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        k_zero,
        centroids,
        k_quant,
        v_bits,
        scale,
        block_tables,
        lengths,
        block_size,
        max(kv_lens),
        partition,
        plan,
    )
    mx.eval(out)
    assert _dispatch_family() == "gqa_decode"
    assert ops.last_gqa_partition_size() == partition

    batch = len(kv_lens)
    established = ops.paged_attention_primitive(
        query,
        k_cache,
        v_cache,
        kv_heads,
        scale,
        0.0,
        block_tables,
        lengths,
        mx.arange(batch + 1, dtype=mx.int32),
        block_size,
        max(kv_lens),
        -1,
        key_scale_cache=k_scale,
        value_scale_cache=v_scale,
        key_zero_cache=k_zero,
        v_centroids=centroids,
        use_turboquant=True,
        quant_type=k_quant,
        v_bits=v_bits,
        num_decode_requests=batch,
        num_decode_tokens=batch,
        max_decode_context_len=max(kv_lens),
        gqa_disabled=True,
    )
    mx.eval(established)
    assert _dispatch_family() in {"per_token_ps0", "per_token_ps512"}

    reference = _grouped_paged_reference(
        query=query,
        key_cache=k_ref,
        value_cache=v_ref,
        query_lens=[1] * batch,
        kv_lens=kv_lens,
        block_tables=tables,
        scale=scale,
    )
    mx.eval(reference)
    return out, established, reference


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("partition", GQA_PARTITIONS)
@pytest.mark.parametrize("geometry", GQA_GEOMETRIES)
def test_matches_established_turboquant_kernel(geometry, partition, dtype) -> None:
    """Every shipped geometry and partition, with a partial final partition."""
    out, established, reference = _run(
        kv_lens=[2 * partition + 37],
        geometry=geometry,
        partition=partition,
        dtype=dtype,
    )
    _assert_close(out, established, dtype)
    _assert_close(out, reference, dtype)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("k_quant", K_QUANTS)
def test_key_quant_types(k_quant, dtype) -> None:
    out, established, reference = _run(
        kv_lens=[1300],
        geometry=(32, 8, 128, 16),
        partition=256,
        dtype=dtype,
        k_quant=k_quant,
    )
    _assert_close(out, established, dtype)
    _assert_close(out, reference, dtype)


@pytest.mark.parametrize("v_bits", V_BITS)
def test_value_bit_widths(v_bits) -> None:
    out, established, reference = _run(
        kv_lens=[1100],
        geometry=(16, 2, 256, 32),
        partition=512,
        dtype=mx.float16,
        v_bits=v_bits,
    )
    _assert_close(out, established, mx.float16)
    _assert_close(out, reference, mx.float16)


@pytest.mark.parametrize("length_plan", [False, True])
@pytest.mark.parametrize("partition", GQA_PARTITIONS)
def test_uneven_batch(partition, length_plan) -> None:
    """Rectangular and compact scratch with rows ending in different partitions."""
    out, established, reference = _run(
        kv_lens=[3 * partition + 5, 17, partition, 2 * partition - 1],
        geometry=(24, 4, 256, 16),
        partition=partition,
        dtype=mx.bfloat16,
        length_plan=length_plan,
    )
    _assert_close(out, established, mx.bfloat16)
    _assert_close(out, reference, mx.bfloat16)


def test_rejects_non_turboquant_cache() -> None:
    query = mx.zeros((1, 32, 128), dtype=mx.float16)
    cache = mx.zeros((4, 16, 8, 128), dtype=mx.float16)
    scales = mx.zeros((4, 16, 8, 4), dtype=mx.float16)
    with pytest.raises(ValueError, match="TurboQuant test entry"):
        get_ops()._gqa_turboquant_paged_attention_for_test(
            query,
            cache,
            cache,
            scales,
            scales,
            scales,
            get_v_centroids(3),
            "q8_0",
            3,
            128**-0.5,
            mx.zeros((1, 4), dtype=mx.int32),
            mx.array([64], dtype=mx.int32),
            16,
            64,
            256,
        )
