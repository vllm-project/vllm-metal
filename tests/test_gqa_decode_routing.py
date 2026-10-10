# SPDX-License-Identifier: Apache-2.0
"""P256/P512 production selection and fallback, verified after native evaluation.

Kernel correctness is covered independently in test_gqa_paged_decode.py.
Tests inject core counts only when hardware detection is unavailable.
"""

from __future__ import annotations

import contextlib

import mlx.core as mx
import numpy as np
import pytest

from tests.gqa_test_utils import (
    _assert_close,
    _assert_fallback,
    _dispatch_family,
    _grouped_paged_reference,
    _run_primitive,
)
from tests.test_gqa_paged_decode import (
    GQA_GEOMETRIES,
    GQA_PARTITIONS,
    GQA_SIMD_GROUPS_PER_CORE,
    NUM_QUERY_HEADS,
)
from vllm_metal.attention.block_tables import build_block_tables
from vllm_metal.metal import get_ops


def test_native_reports_public_routing_capabilities():
    assert get_ops().paged_attention_capabilities() == {
        "gqa_decode": True,
        "gqa_disable": True,
        "decode_routing_metadata": True,
        "gqa_batch_context_lens": True,
        "gqa_length_plan": True,
        "gqa_mixed_decode_plan": True,
    }


def _restore_test_gpu_cores() -> None:
    ops = get_ops()
    ops._override_detected_gpu_core_count_for_test(-1)
    if ops.detected_gpu_core_count() <= 0:
        ops._override_detected_gpu_core_count_for_test(10)


@pytest.fixture(scope="module", autouse=True)
def _enable_dispatch_diagnostics():
    ops = get_ops()
    previous = ops._set_paged_dispatch_diagnostics(True)
    try:
        yield
    finally:
        mx.synchronize()
        ops._set_paged_dispatch_diagnostics(previous)


@pytest.fixture(scope="module", autouse=True)
def _inject_test_gpu_cores():
    """CI runners often omit IORegistry gpu-core-count; keep routing tests on."""
    _restore_test_gpu_cores()
    yield
    get_ops()._override_detected_gpu_core_count_for_test(-1)


def _require_grid(kv_len: int, query_heads: int) -> None:
    """Positive route tests need the measured grid guard on this GPU."""
    cores = get_ops().detected_gpu_core_count()
    if cores <= 0:
        pytest.skip("GPU core count unavailable: GQA conservatively disabled")
    if (kv_len // min(GQA_PARTITIONS)) * query_heads < GQA_SIMD_GROUPS_PER_CORE * cores:
        pytest.skip("This boundary is below the GQA grid guard on this GPU")


def _eligible_context(query_heads: int = NUM_QUERY_HEADS, minimum: int = 32768) -> int:
    cores = get_ops().detected_gpu_core_count()
    if cores <= 0:
        pytest.skip("GPU core count unavailable: GQA conservatively disabled")
    # Round to whole partitions; this helper only sizes positive-test inputs.
    # Expected shape/length decisions are explicit in the boundary tests.
    partitions = (GQA_SIMD_GROUPS_PER_CORE * cores + query_heads - 1) // query_heads
    n = max(minimum, partitions * min(GQA_PARTITIONS))
    return n


def test_core_count_override_restores_hardware_detection() -> None:
    ops = get_ops()
    original = ops.detected_gpu_core_count()
    ops._override_detected_gpu_core_count_for_test(7)
    try:
        assert ops.detected_gpu_core_count() == 7
    finally:
        _restore_test_gpu_cores()
    assert ops.detected_gpu_core_count() == original


def test_core_count_override_changes_default_routing() -> None:
    """Production dispatch reads the override through detected_gpu_core_count()."""
    ops = get_ops()
    # Q=32, KV=8000: 31 full P256 partitions. Eligible iff 31*32 >= 33*cores;
    # forty cores require 10752 KV, while ten cores can select GQA.
    ops._override_detected_gpu_core_count_for_test(40)
    try:
        assert ops.detected_gpu_core_count() == 40
        out, ref = _run_primitive(
            [8000],
            mx.bfloat16,
            interleaved=False,
            seed=41,
            num_decode_requests=1,
        )
        _assert_fallback()
        _assert_close(out, ref, mx.bfloat16)
        ops._override_detected_gpu_core_count_for_test(10)
        out, ref = _run_primitive(
            [8000],
            mx.bfloat16,
            interleaved=False,
            seed=41,
            num_decode_requests=1,
        )
        assert _dispatch_family() == "gqa_decode"
        _assert_close(out, ref, mx.bfloat16)
    finally:
        _restore_test_gpu_cores()


def test_unknown_core_count_keeps_measured_shape_on_baseline() -> None:
    ops = get_ops()
    ops._override_detected_gpu_core_count_for_test(0)
    try:
        assert ops.detected_gpu_core_count() == 0
        out, ref = _run_primitive(
            [32768],
            mx.bfloat16,
            interleaved=False,
            seed=37,
            num_decode_requests=1,
        )
        _assert_fallback()
        _assert_close(out, ref, mx.bfloat16)
    finally:
        _restore_test_gpu_cores()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("offset,interleaved", [(0, False), (17, True)])
def test_gqa_decode_matches_reference(dtype, offset, interleaved) -> None:
    n = _eligible_context() + offset
    out, ref = _run_primitive(
        [n],
        dtype,
        interleaved=interleaved,
        seed=0,
        native_reference=not interleaved,
    )
    assert _dispatch_family() == "gqa_decode"
    _assert_close(out, ref, dtype)


@pytest.mark.parametrize("num_decode_requests", [-1, 1])
def test_one_decode_request_takes_gqa(num_decode_requests) -> None:
    out, ref = _run_primitive(
        [_eligible_context()],
        mx.bfloat16,
        interleaved=True,
        seed=4,
        num_decode_requests=num_decode_requests,
    )
    assert _dispatch_family() == "gqa_decode"
    _assert_close(out, ref, mx.bfloat16)


@pytest.mark.parametrize("q,kv,head,block_size", GQA_GEOMETRIES)
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_each_geometry_lower_boundary_dispatch(q, kv, head, block_size, offset):
    cores = get_ops().detected_gpu_core_count()
    if cores <= 0:
        pytest.skip("GPU core count unavailable")
    minimum = min(GQA_PARTITIONS) * ((GQA_SIMD_GROUPS_PER_CORE * cores + q - 1) // q)
    n = minimum + offset
    if offset >= 0:
        _require_grid(n, q)
    out, ref = _run_primitive(
        [n],
        mx.bfloat16,
        interleaved=True,
        seed=715,
        num_query_heads=q,
        num_kv_heads=kv,
        head_size=head,
        block_size=block_size,
    )
    if offset >= 0:
        assert _dispatch_family() == "gqa_decode"
    else:
        _assert_fallback()
    _assert_close(out, ref, mx.bfloat16)


@pytest.mark.parametrize("q,kv,head,block_size", GQA_GEOMETRIES)
@pytest.mark.parametrize("n", [131071, 131072, 131073, 196608, 262144, 262145])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.slow
def test_each_geometry_stays_on_gqa_at_long_context(q, kv, head, block_size, n, dtype):
    _require_grid(n, q)
    out, ref = _run_primitive(
        [n],
        dtype,
        interleaved=True,
        seed=716,
        num_query_heads=q,
        num_kv_heads=kv,
        head_size=head,
        block_size=block_size,
    )
    assert _dispatch_family() == "gqa_decode"
    _assert_close(out, ref, dtype)


@pytest.mark.parametrize(
    "q,kv,head,n",
    [
        (16, 4, 256, 16384),
        (16, 4, 256, 32768),  # 16/4/256 is outside default routing.
        (16, 4, 256, 65536),
        (16, 4, 256, 131073),
        (32, 4, 64, 32768),  # Head dim 64 is not a shipped GQA specialization.
        (64, 8, 64, 32768),
        (32, 4, 64, 65536),
        (16, 2, 96, 65536),
        (32, 4, 256, 65536),  # Unmeasured 32/4/256 stays on the established path.
        (16, 8, 128, 65536),
        (16, 16, 128, 65536),  # MHA.
    ],
)
def test_unselected_shapes_really_use_fallback(q, kv, head, n):
    out, ref = _run_primitive(
        [n],
        mx.bfloat16,
        interleaved=True,
        seed=717,
        num_query_heads=q,
        num_kv_heads=kv,
        head_size=head,
    )
    _assert_fallback()
    _assert_close(out, ref, mx.bfloat16)


@pytest.mark.parametrize(
    "q,kv,head,block_size",
    [
        (32, 8, 128, 8),
        (32, 8, 128, 32),
        (16, 2, 256, 8),
        (16, 4, 256, 32),
        (24, 4, 256, 32),
    ],
)
def test_unmeasured_kernel_page_sizes_use_fallback(q, kv, head, block_size):
    out, ref = _run_primitive(
        [32768],
        mx.float16,
        interleaved=True,
        seed=718,
        block_size=block_size,
        num_query_heads=q,
        num_kv_heads=kv,
        head_size=head,
    )
    _assert_fallback()
    _assert_close(out, ref, mx.float16)


@pytest.mark.parametrize("num_decode_requests", [-1, 2])
def test_multi_request_decode_stays_off_gqa(num_decode_requests):
    n = 32768
    out, ref = _run_primitive(
        [n, n + 1],
        mx.float16,
        interleaved=True,
        seed=2,
        num_decode_requests=num_decode_requests,
    )
    _assert_fallback()
    _assert_close(out, ref, mx.float16)


@pytest.mark.parametrize("num_decode_requests", [-2, 0, 2])
def test_scheduler_must_confirm_one_decode_or_omit_count(num_decode_requests):
    out, ref = _run_primitive(
        [32768],
        mx.float16,
        interleaved=True,
        seed=3,
        num_decode_requests=num_decode_requests,
    )
    _assert_fallback()
    _assert_close(out, ref, mx.float16)


@pytest.mark.parametrize("query_lens", [[3], [1, 2]])
def test_prefill_and_mixed_batches_stay_off_gqa(query_lens):
    n = 32768
    out, ref = _run_primitive(
        [n] * len(query_lens),
        mx.float16,
        interleaved=True,
        seed=8,
        query_lens=query_lens,
        num_decode_requests=1 if len(query_lens) == 2 else 0,
    )
    assert _dispatch_family() in {"nax_prefill", "tiled_prefill"}
    _assert_close(out, ref, mx.float16)


@pytest.mark.parametrize("disabled", [False, True])
def test_split_mixed_decode_stays_on_upstream_path(disabled, force_tiled_prefill):
    """Splitting one decode row out of a mixed batch must not enable GQA."""
    out, ref = _run_primitive(
        [8192, 128],
        mx.float16,
        interleaved=True,
        seed=720,
        query_lens=[1, 32],
        num_decode_requests=1,
        num_decode_tokens=1,
        max_decode_context_len=8192,
        gqa_disabled=disabled,
    )
    assert _dispatch_family() == "mixed_prefill_decode"
    assert get_ops().last_gqa_partition_size() == 0
    _assert_close(out, ref, mx.float16)


def test_split_nax_mixed_decode_stays_on_upstream_path():
    """The NAX split routes its decode rows exactly like the tiled split."""
    ops = get_ops()
    if not (ops.nax_supported() and ops.nax_ready()):
        pytest.skip("NAX prefill needs an M5 GPU and the NAX metallib")
    out, ref = _run_primitive(
        [8192, 128],
        mx.float16,
        interleaved=True,
        seed=720,
        query_lens=[1, 32],
        num_decode_requests=1,
        num_decode_tokens=1,
        max_decode_context_len=8192,
    )
    assert _dispatch_family() == "mixed_nax_prefill_decode"
    assert ops.last_gqa_partition_size() == 0
    _assert_close(out, ref, mx.float16)


_SERVE_GEOMETRY = (32, 8, 128, 16)
_MIXED_FAMILIES = {"mixed_prefill_decode", "mixed_nax_prefill_decode"}


@contextlib.contextmanager
def _gpu_cores(count: int):
    """Pin the core count that GQA admission reads, then restore detection.

    Mixed-prefix expectations depend on the grid guard (33 SIMD groups per
    core), so every such test fixes the count instead of inheriting the host's.
    """
    get_ops()._override_detected_gpu_core_count_for_test(count)
    try:
        yield
    finally:
        _restore_test_gpu_cores()


def _mixed_inputs(
    decode_lengths,
    prefills,
    *,
    geometry=_SERVE_GEOMETRY,
    dtype=mx.bfloat16,
    cache_block=None,
    table_tokens=0,
    seed=1060,
):
    """Leading one-row decode sequences followed by (query, kv) prefill chunks.

    With ``cache_block``, K and V are strided views into one interleaved buffer
    of scheduler pages larger than the kernel block, with block tables
    translated by ``build_block_tables`` exactly as the runner passes them.
    ``table_tokens`` pads every table row to address that many tokens.
    """
    q_heads, kv_heads, head, block = geometry
    mx.random.seed(seed)
    seqs = [(1, n) for n in decode_lengths] + list(prefills)
    query = mx.random.normal((sum(q for q, _ in seqs), q_heads, head)).astype(dtype)
    if cache_block is None:
        num_blocks = 2048
        shape = (num_blocks, block, kv_heads, head)
        key = mx.random.normal(shape).astype(dtype)
        value = mx.random.normal(shape).astype(dtype)
        width = max(-(-kv // block) for _, kv in seqs)
        tables = mx.random.randint(0, num_blocks, (len(seqs), width)).astype(mx.int32)
        ref_key, ref_value = key, value
    else:
        counts = [-(-kv // cache_block) for _, kv in seqs]
        order = np.random.default_rng(seed).permutation(sum(counts)).tolist()
        pages = []
        for count in counts:
            pages.append(order[:count])
            order = order[count:]
        num_pages = sum(counts)
        raw = mx.random.normal((num_pages, cache_block, kv_heads, 2 * head))
        raw = raw.astype(dtype)
        strides = (cache_block * kv_heads * 2 * head, kv_heads * 2 * head, 2 * head, 1)
        shape = (num_pages, cache_block, kv_heads, head)
        key = get_ops().as_strided(raw, shape, strides, 0)
        value = get_ops().as_strided(raw, shape, strides, head)
        tables, kernel_block = build_block_tables(pages, cache_block)
        assert kernel_block == block
        ref_key = (key + 0).reshape(-1, block, kv_heads, head)
        ref_value = (value + 0).reshape(-1, block, kv_heads, head)
    if table_tokens:
        pad = table_tokens // block - tables.shape[1]
        tables = mx.concatenate([tables, mx.zeros((len(seqs), pad), mx.int32)], 1)
    kv_lens = mx.array([kv for _, kv in seqs], dtype=mx.int32)
    cu = mx.cumsum(mx.array([0] + [q for q, _ in seqs], dtype=mx.int32))
    mx.eval(query, key, value, ref_key, ref_value, tables, kv_lens, cu)
    return {
        "seqs": seqs,
        "geometry": geometry,
        "dtype": dtype,
        "k": key,
        "v": value,
        "ref_k": ref_key,
        "ref_v": ref_value,
        "q": query,
        "tables": tables,
        "kv_lens": kv_lens,
        "cu": cu,
    }


def _mixed_call(d, rows, max_seq_len, **kwargs):
    """Run the first ``rows`` sequences, or the whole batch when rows is None."""
    _, kv_heads, head, block = d["geometry"]
    if rows is None:
        query, tables, kv_lens, cu = d["q"], d["tables"], d["kv_lens"], d["cu"]
    else:
        query, tables, kv_lens = d["q"][:rows], d["tables"][:rows], d["kv_lens"][:rows]
        cu = mx.arange(rows + 1, dtype=mx.int32)
    out = mx.array(0)
    get_ops().paged_attention_primitive(
        query,
        d["k"],
        d["v"],
        kv_heads,
        head**-0.5,
        0.0,
        tables,
        kv_lens,
        cu,
        block,
        max_seq_len,
        -1,
        out,
        **kwargs,
    )
    mx.eval(out)
    return out


def _decode_meta(decode_lengths):
    rows = len(decode_lengths)
    return {
        "num_decode_requests": rows,
        "num_decode_tokens": rows,
        "max_decode_context_len": max(decode_lengths),
    }


def _prefill_backend(name, request):
    ops = get_ops()
    if name == "tiled":
        request.getfixturevalue("force_tiled_prefill")
        return "mixed_prefill_decode"
    if not (ops.nax_supported() and ops.nax_ready()):
        pytest.skip("NAX prefill needs an M5 GPU and the NAX metallib")
    return "mixed_nax_prefill_decode"


def _assert_prefix_runs_pure_decode_gqa(d, decode_lengths, family):
    """Decode rows match the same rows as a pure GQA batch, bit for bit."""
    ops = get_ops()
    rows = len(decode_lengths)
    max_seq_len = max(kv for _, kv in d["seqs"])
    plan = ops.gqa_decode_length_plan(decode_lengths)
    meta = _decode_meta(decode_lengths)

    mixed = _mixed_call(d, None, max_seq_len, gqa_length_plan=plan, **meta)
    assert _dispatch_family() == family
    partition = ops.last_gqa_partition_size()
    assert partition in GQA_PARTITIONS
    assert ops.last_gqa_num_requests() == rows

    pure = _mixed_call(d, rows, max(decode_lengths), gqa_length_plan=plan, **meta)
    assert _dispatch_family() == "gqa_decode"
    assert ops.last_gqa_partition_size() == partition
    whole = _mixed_call(d, None, max_seq_len)

    assert mx.array_equal(mixed[:rows], pure).item()
    assert mx.array_equal(mixed[rows:], whole[rows:]).item()
    head = d["geometry"][2]
    ref = _grouped_paged_reference(
        query=d["q"][:rows].astype(mx.float32),
        key_cache=d["ref_k"].astype(mx.float32),
        value_cache=d["ref_v"].astype(mx.float32),
        query_lens=[1] * rows,
        kv_lens=list(decode_lengths),
        block_tables=np.array(d["tables"][:rows]),
        scale=head**-0.5,
    )
    _assert_close(mixed[:rows], ref, d["dtype"])
    return partition


@pytest.mark.parametrize("prefill_kernel", ["tiled", "nax"])
@pytest.mark.parametrize(
    "decode_lengths,partition",
    [([16384, 9000, 4096, 12000], 512), ([16384], 256)],
    ids=["d4", "d1"],
)
def test_mixed_decode_prefix_uses_gqa_with_its_length_plan(
    prefill_kernel, decode_lengths, partition, request
):
    """With a prefix plan the decode rows run pure-decode GQA bit-for-bit.

    At 40 cores the lone 16384-token row clears the grid guard only with P256
    (64 * 32 >= 1320); the prefill chunk is longer than every decode row, so
    the prefix must be planned from its own lengths, not max_seq_len.
    """
    family = _prefill_backend(prefill_kernel, request)
    d = _mixed_inputs(decode_lengths, [(256, 18000), (64, 64)])
    with _gpu_cores(40):
        assert (
            _assert_prefix_runs_pure_decode_gqa(d, decode_lengths, family) == partition
        )


_GEOMETRY_DECODE_LENGTHS = [8192, 6000, 5000, 4500]


@pytest.mark.parametrize("prefill_kernel", ["tiled", "nax"])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("geometry", GQA_GEOMETRIES)
def test_mixed_decode_prefix_gqa_across_geometries(
    geometry, dtype, prefill_kernel, request
):
    """Every whitelisted geometry admits the prefix (44 full P512 partitions
    times at least 16 heads >= 33 * 20) and keeps the pure-decode bits."""
    family = _prefill_backend(prefill_kernel, request)
    lengths = _GEOMETRY_DECODE_LENGTHS
    d = _mixed_inputs(lengths, [(128, 9000)], geometry=geometry, dtype=dtype)
    with _gpu_cores(20):
        _assert_prefix_runs_pure_decode_gqa(d, lengths, family)


@pytest.mark.parametrize("prefill_kernel", ["tiled", "nax"])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "geometry,cache_block",
    [((24, 4, 256, 16), 784), ((16, 2, 256, 32), 1056)],
)
def test_mixed_decode_prefix_gqa_with_converted_page_stride(
    geometry, cache_block, dtype, prefill_kernel, request
):
    """Hybrid-model pages: strided K/V views of scheduler pages larger than
    the kernel block, read through translated block tables."""
    family = _prefill_backend(prefill_kernel, request)
    lengths = _GEOMETRY_DECODE_LENGTHS
    d = _mixed_inputs(
        lengths,
        [(128, 9000)],
        geometry=geometry,
        dtype=dtype,
        cache_block=cache_block,
    )
    with _gpu_cores(20):
        _assert_prefix_runs_pure_decode_gqa(d, lengths, family)


@pytest.mark.parametrize("prefill_kernel", ["tiled", "nax"])
@pytest.mark.parametrize("with_plan", [True, False])
@pytest.mark.parametrize("prefill_kv", [18000, 65536])
def test_short_decode_prefix_is_gated_by_its_own_length(
    prefill_kv, with_plan, prefill_kernel, request
):
    """The GQA gate counts only the decode rows, never the prefill chunk.

    At 20 cores a lone 4096-token row has 16 full P256 partitions
    (16 * 32 < 660) and must stay per-token. The split still happens, and the
    long prefill KV, which would clear the gate if it were counted, must not
    admit the prefix.
    """
    family = _prefill_backend(prefill_kernel, request)
    ops = get_ops()
    lengths = [4096]
    d = _mixed_inputs(lengths, [(256, prefill_kv)])
    meta = _decode_meta(lengths)
    plan = {"gqa_length_plan": ops.gqa_decode_length_plan(lengths)} if with_plan else {}
    with _gpu_cores(20):
        mixed = _mixed_call(d, None, prefill_kv, **plan, **meta)
        assert _dispatch_family() == family
        assert ops.last_gqa_partition_size() == 0
        pure = _mixed_call(d, 1, 4096, gqa_disabled=True, **meta)
        whole = _mixed_call(d, None, prefill_kv)
    assert mx.array_equal(mixed[:1], pure).item()
    assert mx.array_equal(mixed[1:], whole[1:]).item()


@pytest.mark.parametrize("prefill_kernel", ["tiled", "nax"])
@pytest.mark.parametrize("disabled", [False, True])
def test_mixed_prefix_plan_with_allocation_wide_max_seq_len(
    disabled, prefill_kernel, request
):
    """A legal 4M-token max_seq_len must not size the prefix's partitions.

    At 12 cores the 4096-token row is admitted with P256 (16 * 32 >= 396); its
    16 partitions follow the plan. Disabled, the per-token prefix needs 8 P512
    partitions where the allocation bound would plan 8192.
    """
    family = _prefill_backend(prefill_kernel, request)
    ops = get_ops()
    lengths = [4096]
    bound = 4 * 1024 * 1024
    d = _mixed_inputs(lengths, [(2, 2)], table_tokens=bound)
    meta = _decode_meta(lengths)
    plan = ops.gqa_decode_length_plan(lengths)
    with _gpu_cores(12):
        mixed = _mixed_call(
            d, None, bound, gqa_length_plan=plan, gqa_disabled=disabled, **meta
        )
        assert _dispatch_family() == family
        assert ops.last_gqa_partition_size() == (0 if disabled else 256)
        pure = _mixed_call(
            d, 1, 4096, gqa_length_plan=plan, gqa_disabled=disabled, **meta
        )
        whole = _mixed_call(d, None, bound)
    assert mx.array_equal(mixed[:1], pure).item()
    assert mx.array_equal(mixed[1:], whole[1:]).item()


@pytest.mark.parametrize("prefill_kernel", ["tiled", "nax"])
def test_mixed_decode_prefix_without_plan_or_disabled_stays_per_token(
    prefill_kernel, request
):
    family = _prefill_backend(prefill_kernel, request)
    ops = get_ops()
    lengths = [16384, 9000, 4096, 12000]
    d = _mixed_inputs(lengths, [(256, 18000)])
    meta = _decode_meta(lengths)
    plan = ops.gqa_decode_length_plan(lengths)
    for kwargs in ({}, {"gqa_length_plan": plan, "gqa_disabled": True}):
        _mixed_call(d, None, 18000, **meta, **kwargs)
        assert _dispatch_family() == family
        assert ops.last_gqa_partition_size() == 0


@pytest.mark.parametrize("bad", ["count", "length"])
def test_mixed_prefix_plan_must_match_the_decode_rows(bad):
    ops = get_ops()
    lengths = [8192, 8192]
    d = _mixed_inputs(lengths, [(64, 4096)])
    plan_lengths = lengths + [4096] if bad == "count" else [8192, 9000]
    with pytest.raises(ValueError, match="leading ordinary decode rows"):
        _mixed_call(
            d,
            None,
            9000,
            num_decode_requests=2,
            num_decode_tokens=2,
            max_decode_context_len=8192,
            gqa_length_plan=ops.gqa_decode_length_plan(plan_lengths),
        )


def test_spec_window_does_not_switch_kernel_family() -> None:
    out, ref = _run_primitive(
        [32768 + 4],
        mx.float16,
        interleaved=True,
        seed=9,
        window_seqlen_q=4,
        query_lens=[4],
    )
    assert _dispatch_family().startswith("window_")
    _assert_close(out, ref, mx.float16)


@pytest.mark.parametrize(
    "feature", [{"softcap": 2.0}, {"sliding_window": 4096}, {"sink_value": 12.0}]
)
def test_special_attention_features_use_fallback(feature):
    out, ref = _run_primitive(
        [32768],
        mx.float16,
        interleaved=True,
        seed=10,
        **feature,
    )
    _assert_fallback()
    _assert_close(out, ref, mx.float16)


def test_turboquant_uses_fallback_with_dequantized_reference() -> None:
    out, ref = _run_primitive(
        [65536],
        mx.float16,
        interleaved=False,
        seed=12,
        num_query_heads=16,
        num_kv_heads=2,
        turboquant=True,
    )
    _assert_fallback()
    _assert_close(out, ref, mx.float16)


def test_matching_float32_uses_fallback() -> None:
    out, ref = _run_primitive([32768], mx.float32, interleaved=True, seed=11)
    _assert_fallback()
    _assert_close(out, ref, mx.float32)


@pytest.mark.parametrize("minimum", [32768, 196609])
@pytest.mark.parametrize("q,kv,head,block_size", GQA_GEOMETRIES)
def test_gqa_disable_flag_forces_established_kernels(
    monkeypatch, minimum, q, kv, head, block_size
) -> None:
    from vllm_metal import envs

    monkeypatch.delenv("VLLM_METAL_DISABLE_GQA_DECODE", raising=False)
    assert envs.VLLM_METAL_DISABLE_GQA_DECODE is False
    n = _eligible_context(q, minimum=minimum)
    shape = {
        "num_query_heads": q,
        "num_kv_heads": kv,
        "head_size": head,
        "block_size": block_size,
    }
    out_on, ref = _run_primitive([n], mx.bfloat16, interleaved=True, seed=7, **shape)
    assert _dispatch_family() == "gqa_decode"
    out_off, _ = _run_primitive(
        [n], mx.bfloat16, interleaved=True, seed=7, gqa_disabled=True, **shape
    )
    _assert_fallback()
    _assert_close(out_on, ref, mx.bfloat16)
    _assert_close(out_off, ref, mx.bfloat16)

    monkeypatch.setenv("VLLM_METAL_DISABLE_GQA_DECODE", "1")
    assert envs.VLLM_METAL_DISABLE_GQA_DECODE is True
    out_env, _ = _run_primitive(
        [n],
        mx.bfloat16,
        interleaved=True,
        seed=7,
        gqa_disabled=envs.VLLM_METAL_DISABLE_GQA_DECODE,
        **shape,
    )
    _assert_fallback()
    _assert_close(out_env, ref, mx.bfloat16)


# Golden boundaries remain independent of the exported budget. Sharing the
# supported geometries must not turn these into an implementation self-check.
_ADMISSION_AT_CORE_COUNT = {
    32: {20: 5376, 40: 10752},
    24: {40: 14080, 86: 30464},
    16: {40: 21248, 80: 42240, 86: 45568},
}


@pytest.mark.parametrize(
    "q,kv,head,block,cores,minimum",
    [
        (*geometry, cores, minimum)
        for geometry in GQA_GEOMETRIES
        for cores, minimum in _ADMISSION_AT_CORE_COUNT[geometry[0]].items()
    ],
)
def test_shape_and_device_performance_gate(q, kv, head, block, cores, minimum):
    ops = get_ops()
    assert not ops.gqa_decode_shape_eligible(q, kv, head, minimum - 1, cores, block)
    assert ops.gqa_decode_shape_eligible(q, kv, head, minimum, cores, block)
    assert ops.gqa_decode_shape_eligible(q, kv, head, minimum + 1, cores, block)


@pytest.mark.parametrize("q,kv,head,block", GQA_GEOMETRIES)
@pytest.mark.parametrize("n,cores", [(0, 40), (-1, 40), (262145, 0), (65536, -1)])
def test_unknown_core_count_or_empty_length_is_not_eligible(
    q, kv, head, block, n, cores
):
    assert not get_ops().gqa_decode_shape_eligible(q, kv, head, n, cores, block)


@pytest.mark.parametrize("q,kv,head,block", GQA_GEOMETRIES)
@pytest.mark.parametrize("n", [131072, 131073, 262145, 524289, 1048576])
def test_policy_has_no_fixed_long_context_ceiling(q, kv, head, block, n):
    assert get_ops().gqa_decode_partition_size(q, kv, head, n, 40, block) == 512


@pytest.mark.parametrize(
    "q,kv,head,block",
    [
        (16, 4, 256, 16),
        (32, 4, 64, 16),
        (64, 8, 64, 16),
        (16, 2, 96, 16),
        (32, 4, 256, 16),
        (16, 8, 128, 16),
        (16, 16, 128, 16),
        (8, 1, 128, 16),
        (18, 4, 128, 16),
        (16, 0, 128, 16),
        (32, 8, 128, 32),
        (24, 4, 256, 32),
    ],
)
def test_policy_rejects_unmeasured_geometries(q, kv, head, block):
    assert not get_ops().gqa_decode_shape_eligible(q, kv, head, 65536, 40, block)
    assert get_ops().gqa_decode_partition_size(q, kv, head, 65536, 40, block) == 0


@pytest.mark.parametrize("q,kv,head,block", GQA_GEOMETRIES)
def test_default_partition_thresholds(q, kv, head, block):
    thresholds = {32: [10752, 21504], 24: [14080, 28160], 16: [21248, 42496]}[q]
    ops = get_ops()
    for previous, part, threshold in zip([0, 256], [256, 512], thresholds, strict=True):
        assert (
            ops.gqa_decode_partition_size(q, kv, head, threshold - 1, 40, block)
            == previous
        )
        assert ops.gqa_decode_partition_size(q, kv, head, threshold, 40, block) == part
        assert (
            ops.gqa_decode_partition_size(q, kv, head, threshold + 1, 40, block) == part
        )


@pytest.mark.parametrize("q,kv,head,block", GQA_GEOMETRIES)
@pytest.mark.parametrize("part", GQA_PARTITIONS)
@pytest.mark.parametrize("offset", [-1, 0, 17])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_default_partition_executes_and_matches_oracle(
    q, kv, head, block, part, offset, dtype
):
    ops = get_ops()
    cores = ops.detected_gpu_core_count()
    if cores <= 0:
        pytest.skip("GPU core count unavailable")
    threshold = part * ((GQA_SIMD_GROUPS_PER_CORE * cores + q - 1) // q)
    out, ref = _run_primitive(
        [threshold + offset],
        dtype,
        interleaved=True,
        seed=719,
        num_query_heads=q,
        num_kv_heads=kv,
        head_size=head,
        block_size=block,
        num_decode_requests=1,
    )
    expected = part if offset >= 0 else (0 if part == 256 else part // 2)
    if expected:
        assert _dispatch_family() == "gqa_decode"
    else:
        _assert_fallback()
    assert ops.last_gqa_partition_size() == expected
    _assert_close(out, ref, dtype)
