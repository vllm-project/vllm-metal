# SPDX-License-Identifier: Apache-2.0
"""Exercise production routing, fused cache writes and native TQ attention."""

import gc
from contextlib import nullcontext
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest
import torch
from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec

from tools.benchmark.tq_prefill_case import build_case
from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.attention.caches.turboquant import (
    get_v_centroids,
    prefill_bytes_per_token,
    prefill_workspace_bytes,
    turbo_quant_decode,
)
from vllm_metal.attention.context import (
    PagedAttentionContext,
    clear_context,
    get_context,
    prepare_grouped,
)
from vllm_metal.attention.impls import sdpa, turboquant_prefill
from vllm_metal.attention.impls.turboquant_prefill import (
    unsupported_reason,
    workspace_upper_bound,
)
from vllm_metal.metal import get_ops
from vllm_metal.v1.cache_policy import TurboQuantAttentionSpec


@pytest.fixture(autouse=True)
def enable_prefill(monkeypatch):
    # Exercise tiled correctness on older CI machines as well as NAX on M5.
    # Automatic rollout is tested separately below.
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL", "1")
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", "256")


@pytest.fixture
def recorded_ops(monkeypatch):
    native = get_ops()
    calls = []

    class RecordingOps:
        def __getattr__(self, name):
            return getattr(native, name)

        def paged_attention_primitive(self, *args, **kwargs):
            calls.append((args, kwargs))
            return native.paged_attention_primitive(*args, **kwargs)

    monkeypatch.setattr(sdpa, "get_ops", lambda: RecordingOps())
    return calls


@pytest.fixture
def materialized_lengths(monkeypatch):
    original = sdpa.materialize_turboquant_pages
    lengths = []

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        # Record sizes without retaining tensors and extending their lifetime.
        lengths.append(result[0].shape[0])
        return result

    monkeypatch.setattr(sdpa, "materialize_turboquant_pages", record)
    return lengths


@pytest.fixture(params=[False, True], ids=["native-default", "tiled"])
def prefill_backend(request):
    if request.param:
        request.getfixturevalue("force_tiled_prefill")


def assert_parity(case):
    # No eval between sdpa_forward's tq_encode and attention: in particular,
    # current-token cache cells were NOT populated by the fixture.
    output = case.forward()
    mx.eval(output)
    reference = case.reference()
    mx.eval(reference)
    assert mx.all(mx.isfinite(output)).item()
    np.testing.assert_allclose(
        np.array(output.astype(mx.float32)),
        np.array(reference.astype(mx.float32)),
        atol=0.02,
        rtol=0.03,
    )
    return output, reference


def test_unsupported_reason_covers_each_gate() -> None:
    """Every rejection arm names what disqualifies the model or layout."""
    ok = {
        "dtype": mx.float16,
        "head_dim": 128,
        "kernel_block_size": 16,
        "cache_block_size": 32,
        "stored_block_size": 32,
    }
    assert unsupported_reason(**ok) is None
    assert "dtype" in unsupported_reason(**{**ok, "dtype": mx.float32})
    assert "head_dim" in unsupported_reason(**{**ok, "head_dim": 96})
    # Layout gates: kernel block must be a supported divisor of the cache
    # block, and the stored layout must match the scheduler's.
    for bad in (
        {"kernel_block_size": 64},
        {"stored_block_size": 16},
        {"cache_block_size": 33},
    ):
        assert "cache layout" in unsupported_reason(**{**ok, **bad})


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    ("k_quant", "v_quant"),
    [
        ("q8_0", "q3_0"),
        ("q4_0", "q4_0"),
        ("q5_0", "q5_0"),
        ("int2", "q2_0"),
        ("uint8", "q8_0"),
    ],
)
def test_prefill_quantization_formats(
    recorded_ops, prefill_backend, dtype, k_quant, v_quant
):
    case = build_case(dtype=dtype, k_quant=k_quant, v_quant=v_quant)
    assert_parity(case)
    assert len(recorded_ops) == 1
    args, kwargs = recorded_ops[0]
    assert not kwargs.get("use_turboquant", False)
    assert kwargs["window_seqlen_q"] == case.ctx.verify_window_q == 1
    assert args[1].dtype == dtype
    assert args[1].shape == (17, 16, 2, 128)


@pytest.mark.parametrize(
    ("head_dim", "block_size"), [(64, 8), (128, 32), (256, 544), (512, 64)]
)
def test_prefill_strided_translated_pages(
    recorded_ops, prefill_backend, head_dim, block_size
):
    case = build_case(
        head_dim=head_dim,
        block_size=block_size,
        page_padding=512,
        qlens=(256,),
        context_lens=(1153,),
        k_quant="q4_0",
        v_quant="q4_0",
        extra_table_pages=3,
    )
    assert_parity(case)
    args, kwargs = recorded_ops[0]
    assert not kwargs.get("use_turboquant", False)
    assert kwargs.get("nax_split_hint", -1) == (1 if get_ops().nax_ready() else -1)
    kb = args[9]
    # Only ceil(valid KV / kernel block) pages, even if the scheduler page
    # contains many unused sub-blocks and the input table has spare capacity.
    assert args[1].shape[0] == (1153 + kb - 1) // kb


def test_mixed_batch_routes_and_restores_rows(recorded_ops, prefill_backend):
    case = build_case(
        qlens=(1, 128, 3, 129),
        context_lens=(4097, 384, 515, 641),
        page_padding=512,
        shared_prefix=True,
        softcap=2.0,
    )
    # Whole-batch decode routing must not leak into the compacted TQ prefill.
    case.ctx.num_decode_requests = 1
    case.ctx.num_decode_tokens = 1
    case.ctx.max_decode_context_len = 4097
    assert_parity(case)
    assert len(recorded_ops) == 2
    prefill, fallback = recorded_ops
    assert not prefill[1].get("use_turboquant", False)
    assert fallback[1]["use_turboquant"]
    assert prefill[0][0].shape[0] == 257
    assert fallback[0][0].shape[0] == 4
    assert prefill[0][7].tolist() == [384, 641]
    assert fallback[0][7].tolist() == [4097, 515]
    # The two selected requests share 16 physical prefix pages. Neither the
    # long decode's 4097 tokens nor its rectangular padding is materialized.
    assert prefill[0][1].shape[0] == 24 + 41 - 16
    assert prefill[0][6][0, 0].item() == prefill[0][6][1, 0].item()


def test_nax_hint_describes_only_the_dequantized_prefill(recorded_ops):
    """The compressed short request must not suppress the full prefill blocks."""
    case = build_case(qlens=(256, 63), context_lens=(257, 2111))
    assert_parity(case)
    prefill = [kw for _, kw in recorded_ops if not kw.get("use_turboquant", False)]
    assert len(prefill) == 1
    assert prefill[0].get("nax_split_hint", -1) == (1 if get_ops().nax_ready() else -1)


@pytest.mark.parametrize("qlen", [1, 2, 32, 127, 128])
def test_short_suffix_crossover(recorded_ops, qlen):
    case = build_case(qlens=(qlen,), context_lens=(2049,))
    output, reference = assert_parity(case)
    quantized = recorded_ops[0][1].get("use_turboquant", False)
    assert quantized == (qlen < 128)
    if quantized:
        assert mx.array_equal(output, reference).item()


@pytest.mark.parametrize("past", [0, 2048])
def test_independent_histories_batch_when_they_fit_workspace(
    recorded_ops, prefill_backend, past
):
    case = build_case(
        qlens=(128,) * 4, context_lens=(past + 128,) * 4, page_padding=512
    )
    assert_parity(case)
    assert len(recorded_ops) == 1
    assert not recorded_ops[0][1].get("use_turboquant", False)
    assert recorded_ops[0][0][1].shape[0] == (32 if past == 0 else 544)


def test_capacity_fallback_restores_interleaved_rows(recorded_ops, monkeypatch):
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", "5")
    case = build_case(
        qlens=(128, 1, 129, 5, 128), context_lens=(2049, 4097, 2063, 515, 2177)
    )
    assert_parity(case)
    assert len(recorded_ops) == 2
    assert not recorded_ops[0][1].get("use_turboquant", False)
    assert recorded_ops[-1][1]["use_turboquant"]
    assert recorded_ops[-1][0][0].shape[0] == 263


def test_prefill_scratch_does_not_scale_with_unrelated_histories(monkeypatch):
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", "64")
    peaks = []
    for count in [1, 4]:
        case = build_case(
            qlens=(128,) * count,
            context_lens=(8192,) * count,
            head_dim=256,
            n_heads=24,
            n_kv_heads=4,
        )
        mx.eval(case.forward())
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        before = mx.get_active_memory()
        output = case.forward()
        mx.eval(output)
        mx.synchronize()
        peaks.append(mx.get_peak_memory() - before)
        del case, output
        gc.collect()
        mx.clear_cache()
    # Output/query rows grow, but four independent histories must not retain
    # four full dequantizations. Generous margin avoids allocator-size noise.
    assert peaks[1] < 2 * peaks[0] + 32 * 2**20, peaks


@pytest.mark.parametrize("k_quant,v_quant", [("q8_0", "q3_0"), ("q5_0", "q5_0")])
def test_peak_materialization_fits_reserved_allowance(
    materialized_lengths, monkeypatch, k_quant, v_quant
):
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", "64")
    case = build_case(
        qlens=(128,),
        context_lens=(14080,),
        head_dim=256,
        n_heads=24,
        n_kv_heads=4,
        k_quant=k_quant,
        v_quant=v_quant,
    )
    mx.eval(case.forward())
    mx.eval(case.reference())
    materialized_lengths.clear()
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    before = mx.get_active_memory()
    mx.reset_peak_memory()
    result = case.forward()
    mx.eval(result)
    mx.synchronize()
    extra = mx.get_peak_memory() - before
    assert materialized_lengths == [14080]
    assert extra < prefill_workspace_bytes(), extra


@pytest.mark.parametrize("mib", [0, 1, 16])
def test_admission_uses_rounded_blocks_at_budget_boundary(monkeypatch, mib):
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", str(mib))
    case = build_case()
    block_bytes = 16 * prefill_bytes_per_token(2, 128) + 4
    blocks = mib * 2**20 // block_bytes
    for extra, selected in [(0, mib != 0), (1, False)]:
        case.ctx.context_lens = [max(128, (blocks + extra) * 16)]
        case.ctx.block_tables = [list(range(blocks + extra))]
        meta = sdpa._kernel_metadata(case.ctx, None, [], case.ctx.block_tables, 16)
        plan = sdpa._turboquant_prefill_plan(
            case.ctx, meta, case.ctx.block_tables, 16, 8, 2, 128
        )
        assert (plan is not None) == selected
        case.ctx.kernel_metadata_cache.clear()


@pytest.mark.parametrize("qlen", [128, 255, 256])
def test_mha_requires_more_rows_to_amortize_dequant(recorded_ops, qlen):
    case = build_case(
        qlens=(qlen,), context_lens=(1025,), n_heads=8, n_kv_heads=8, head_dim=64
    )
    output, reference = assert_parity(case)
    quantized = recorded_ops[0][1].get("use_turboquant", False)
    assert quantized == (qlen < 256)
    if quantized:
        assert mx.array_equal(output, reference).item()


@pytest.mark.parametrize("qlen", [128, 255, 256])
def test_wide_heads_keep_short_chunks_compressed(recorded_ops, qlen):
    case = build_case(qlens=(qlen,), context_lens=(1025,), head_dim=512)
    output, reference = assert_parity(case)
    quantized = recorded_ops[0][1].get("use_turboquant", False)
    assert quantized == (qlen < 256)
    if quantized:
        assert mx.array_equal(output, reference).item()


@pytest.mark.parametrize("mode", ["verify", "fp32", "sliding"])
def test_preserves_compressed_fallbacks(recorded_ops, mode):
    kwargs = {}
    if mode == "verify":
        kwargs["qlens"] = (5,)
    elif mode == "fp32":
        kwargs["dtype"] = mx.float32
    else:
        kwargs["sliding_window"] = 32
    case = build_case(**kwargs)
    if mode == "verify":
        case.ctx.verify_window_q = 5
    output, reference = assert_parity(case)
    assert recorded_ops[0][1]["use_turboquant"]
    assert mx.array_equal(output, reference).item()


@pytest.mark.parametrize("qlen", [1, 128])
def test_rejects_sinks_during_prefill_and_decode(recorded_ops, qlen):
    case = build_case(qlens=(qlen,))
    case.inner.sinks = mx.zeros((case.inner.n_heads,), mx.float32)
    with pytest.raises(ValueError, match="sinks are not supported with TurboQuant"):
        case.forward()
    assert recorded_ops[0][1]["use_turboquant"]


def test_preserves_mm_prefix_rejection(recorded_ops):
    case = build_case()
    case.ctx.segment_bidi_ranges = [[(129, 257)]]
    case.ctx.bidi_layer_kinds = frozenset({"full"})
    with pytest.raises(
        ValueError, match="mm_prefix ranges are not supported with TurboQuant"
    ):
        case.forward()
    assert recorded_ops[0][1]["use_turboquant"]


def test_prefill_metadata_reused_only_within_forward(recorded_ops):
    case = build_case()
    first = case.forward()
    mx.eval(first)
    meta = next(iter(case.ctx.kernel_metadata_cache.values()))
    plan = next(iter(meta.tq_prefill_plans.values()))
    second = case.forward()
    mx.eval(second)
    assert next(iter(case.ctx.kernel_metadata_cache.values())) is meta
    assert next(iter(meta.tq_prefill_plans.values())) is plan
    assert mx.array_equal(first, second).item()
    case.ctx.kernel_metadata_cache.clear()
    third = case.forward()
    mx.eval(third)
    fresh = next(iter(case.ctx.kernel_metadata_cache.values()))
    assert fresh is not meta
    assert next(iter(fresh.tq_prefill_plans.values())) is not plan


@pytest.mark.parametrize("block_size", [16, 544])
def test_hybrid_groups_keep_separate_plans_and_reuse_them_within_group(
    block_size, prefill_backend
):
    """Two SDPA groups interleaved with recurrent state use their own page maps."""
    case = build_case(block_size=block_size)
    attention = TurboQuantAttentionSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.int8,
        k_quant="q8_0",
        v_quant="q3_0",
    )
    state = MambaSpec(
        block_size=block_size,
        shapes=((2, 4), (1, 4, 32)),
        dtypes=(torch.float16, torch.float32),
        page_size_padded=attention.page_size_bytes,
        mamba_cache_mode="align",
    )
    groups = [
        KVCacheGroupSpec(layer_names=["a0", "a1"], kv_cache_spec=attention),
        KVCacheGroupSpec(layer_names=["s0", "s1"], kv_cache_spec=state),
        KVCacheGroupSpec(layer_names=["a2", "a3"], kv_cache_spec=attention),
    ]
    pages = (257 + block_size - 1) // block_size
    count = 2 * pages + 3
    config = VllmConfig()
    config.cache_config.kv_cache_layout = "LBNHC"
    layout = get_kv_cache_config_from_groups(
        config, groups, count * attention.page_size_bytes * 2
    )
    layout.kv_cache_layout = "LBNHC"
    storage = KVCacheStorage(layout)
    storage.zero_blocks(list(range(layout.num_blocks)))
    cache = MetalPagedKVCache.from_upstream(
        storage, ["a0", "a1", "a2", "a3"], dtype=mx.bfloat16
    )
    tables = [list(range(1, 1 + pages)), list(range(pages + 2, 2 * pages + 2))]
    for layer, group in enumerate([0, 0, 1, 1]):
        slots = [
            tables[group][t // block_size] * block_size + t % block_size
            for t in range(129)
        ]
        k = mx.random.normal((129, 2, 128), key=mx.random.key(101 + layer)).astype(
            mx.bfloat16
        )
        v = mx.random.normal(k.shape, key=mx.random.key(201 + layer)).astype(
            mx.bfloat16
        )
        arrays = (
            cache.key_caches,
            cache.value_caches,
            cache.key_scale_caches,
            cache.value_scale_caches,
            cache.key_zero_caches,
        )
        updated = get_ops().tq_encode(
            k,
            v,
            *(array[layer] for array in arrays),
            mx.array(slots, mx.int64),
            get_v_centroids(3),
            3,
            8,
            True,
        )
        for array, value in zip(arrays, updated, strict=True):
            array[layer] = value
        mx.eval(*updated)
    allowance = workspace_upper_bound(
        max_model_len=257,
        max_num_seqs=1,
        max_num_batched_tokens=128,
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=128,
        block_size=block_size,
    )
    prepare_grouped(
        [],
        [(tables, 128, 129)],
        [block_size, block_size],
        tq_prefill_workspace_bytes=allowance,
    )
    ctx = get_context()
    assert ctx is not None
    try:
        reference = []
        with patch.object(sdpa, "_turboquant_prefill_plan", return_value=None):
            for layer in range(3):
                ctx.kernel_metadata_cache.clear()
                out = sdpa.sdpa_forward(case.inner, case.x, ctx, cache, layer)[0]
                mx.eval(out)
                reference.append(out)
        ctx.kernel_metadata_cache.clear()
        first_plan = None
        for layer in range(3):
            out = sdpa.sdpa_forward(case.inner, case.x, ctx, cache, layer)[0]
            mx.eval(out)
            np.testing.assert_allclose(
                np.array(out.astype(mx.float32)),
                np.array(reference[layer].astype(mx.float32)),
                atol=0.02,
                rtol=0.03,
            )
            group = cache.group_index_for_layer(layer)
            meta = ctx.kernel_metadata_cache[(group, block_size)]
            plan = next(iter(meta.tq_prefill_plans.values()))
            assert plan is not None
            assert plan.workspace_bytes <= ctx.tq_prefill_workspace_bytes
            assert set(plan.pool_pages.tolist()) == set(tables[group])
            if layer == 0:
                first_plan = plan
            elif layer == 1:
                assert plan is first_plan
            else:
                assert plan is not first_plan
        assert len(ctx.kernel_metadata_cache) == 2
    finally:
        clear_context()


def test_read_existing_cache_stays_read_only(recorded_ops):
    case = build_case(k_quant="q4_0", v_quant="q4_0")
    output = case.forward()
    mx.eval(output)
    before = np.array(case.cache._storage.buffers[0])
    read_output, _ = sdpa.sdpa_forward(
        case.inner,
        case.x,
        case.ctx,
        case.cache,
        0,
        read_existing_kv=True,
    )
    mx.eval(read_output)
    assert mx.array_equal(output, read_output).item()
    np.testing.assert_array_equal(before, np.array(case.cache._storage.buffers[0]))


def test_admission_thresholds_computed_once_per_forward():
    # The thresholds depend on geometry, format and context, never on the
    # layer, so one forward computes them once and each layer reuses them.
    case = build_case(qlens=(128, 128), context_lens=(257, 1153))
    meta = sdpa._kernel_metadata(case.ctx, None, [], case.ctx.block_tables, 16)
    seen = []
    original = turboquant_prefill.min_prefill_tokens

    def counting(*args, **kwargs):
        seen.append(kwargs["context_len"])
        return original(*args, **kwargs)

    with patch.object(turboquant_prefill, "min_prefill_tokens", counting):
        for _layer in range(3):
            sdpa._turboquant_prefill_plan(
                case.ctx, meta, case.ctx.block_tables, 16, 8, 2, 128
            )
        case.ctx.kernel_metadata_cache.clear()
        fresh = sdpa._kernel_metadata(case.ctx, None, [], case.ctx.block_tables, 16)
        sdpa._turboquant_prefill_plan(
            case.ctx, fresh, case.ctx.block_tables, 16, 8, 2, 128
        )

    assert seen == [257, 1153, 257, 1153]


@pytest.mark.parametrize("block_size", [16, 544])
def test_prefill_addressing_preserves_large_physical_page_ids(block_size):
    case = build_case(block_size=block_size, context_lens=(1153,))
    # After translation kernel IDs still fit int32, while absolute token
    # offsets do not. No multi-terabyte physical allocation is required.
    page_count = (1153 + block_size - 1) // block_size
    first_page = 2**27 if block_size == 16 else 2**26
    assert first_page * block_size >= 2**31
    case.ctx.block_tables = [[first_page + i for i in range(page_count)]]
    meta = sdpa._kernel_metadata(case.ctx, None, [], case.ctx.block_tables, block_size)
    plan = sdpa._turboquant_prefill_plan(
        case.ctx, meta, case.ctx.block_tables, block_size, 8, 2, 128
    )
    assert plan is not None
    size = plan.pool_pages.shape[0]
    assert plan.pool_pages.tolist() == [
        case.ctx.block_tables[0][t // block_size] for t in range(size)
    ]
    assert plan.pool_offsets.tolist() == [t % block_size for t in range(size)]


def test_over_budget_history_falls_back_before_dequantization(
    recorded_ops, monkeypatch
):
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", "1")
    case = build_case(qlens=(128,), context_lens=(1153,))

    def no_dequant(*args, **kwargs):
        pytest.fail("over-budget history must not allocate dequantization buffers")

    monkeypatch.setattr(sdpa, "materialize_turboquant_pages", no_dequant)
    output, reference = assert_parity(case)
    assert len(recorded_ops) == 1
    assert recorded_ops[0][1]["use_turboquant"]
    assert mx.array_equal(output, reference).item()


@pytest.mark.parametrize("workspace_mib,selected", [(6, False), (8, True)])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_split_budget_includes_query_and_output_copies(
    recorded_ops, monkeypatch, workspace_mib, selected, dtype
):
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", str(workspace_mib))
    # KV alone fits 6 MiB. Three 2-byte routing copies push it above 6 MiB
    # but below 8 MiB; counting the element width twice would reject both.
    case = build_case(
        qlens=(128, 1),
        context_lens=(1153, 32),
        n_heads=64,
        n_kv_heads=2,
        dtype=dtype,
    )
    output, reference = assert_parity(case)
    assert len(recorded_ops) == (2 if selected else 1)
    assert recorded_ops[0][1].get("use_turboquant", False) == (not selected)
    if not selected:
        assert mx.array_equal(output, reference).item()


@pytest.mark.parametrize("workspace_mib,selected", [(1, False), (2, True)])
def test_split_budget_includes_long_fallback_block_table(
    monkeypatch, workspace_mib, selected
):
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", str(workspace_mib))
    case = build_case(qlens=(128, 1), context_lens=(216, 32))
    # Admission only: a short prefill shares metadata with a long decode.
    case.ctx.context_lens[1] = 131072
    case.ctx.block_tables[1] = list(range(8192))
    meta = sdpa._kernel_metadata(case.ctx, None, [], case.ctx.block_tables, 16)
    plan = sdpa._turboquant_prefill_plan(
        case.ctx, meta, case.ctx.block_tables, 16, 8, 2, 128
    )
    assert (plan is not None) == selected
    if plan is None:
        return
    assert plan.fallback is not None
    metadata = [plan.pool_pages, plan.pool_offsets, plan.restore_indices]
    for batch in (plan.prefill, plan.fallback):
        metadata.extend(
            [
                batch.query_indices,
                batch.block_tables,
                batch.seq_lens,
                batch.cu_seqlens_q,
            ]
        )
    # Size the actual retained metadata and data-copy shapes independently
    # of the planner's estimate, including the copied fallback table.
    query = mx.zeros((129, 8, 128), dtype=mx.bfloat16)
    kv = mx.zeros((2, plan.pool_pages.size, 2, 128), dtype=mx.bfloat16)
    required = kv.nbytes + 3 * query.nbytes + sum(a.nbytes for a in metadata)
    assert required <= plan.workspace_bytes <= workspace_mib * 2**20


@pytest.mark.parametrize("shared_prefix", [False, True])
def test_whole_batch_does_not_reserve_unused_split_copies(
    recorded_ops, monkeypatch, shared_prefix
):
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", "6")
    # Both histories fit, whereas the unnecessary split-routing copies alone
    # would exceed the allowance. The entire batch needs no query reordering.
    case = build_case(
        qlens=(128, 128),
        context_lens=(257, 257),
        n_heads=24,
        n_kv_heads=4,
        head_dim=256,
        shared_prefix=shared_prefix,
    )
    assert_parity(case)
    assert len(recorded_ops) == 1
    assert not recorded_ops[0][1].get("use_turboquant", False)
    assert recorded_ops[0][0][7].tolist() == [257, 257]


def test_128k_history_does_not_block_smaller_candidate(recorded_ops):
    # Admission only: reject the large row without allocating its cache.
    case = build_case(qlens=(128, 128), context_lens=(257, 257))
    case.ctx.context_lens[0] = 131072
    meta = sdpa._kernel_metadata(case.ctx, None, [], case.ctx.block_tables, 16)
    plan = sdpa._turboquant_prefill_plan(
        case.ctx, meta, case.ctx.block_tables, 16, 24, 4, 256
    )
    assert plan is not None
    assert plan.prefill.seq_lens.tolist() == [257]
    assert plan.fallback.seq_lens.tolist() == [131072]
    assert plan.workspace_bytes <= 256 * 2**20


@pytest.mark.parametrize(
    "mode,ready,selected",
    [
        ("auto", False, False),
        ("auto", True, True),
        ("0", True, False),
        ("1", False, True),
    ],
)
def test_rollout_requires_nax_or_explicit_opt_in(
    recorded_ops, monkeypatch, request, mode, ready, selected
):
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL", mode)
    if not ready:
        request.getfixturevalue("force_tiled_prefill")
    monkeypatch.setattr(get_ops(), "nax_ready", lambda: ready)
    case = build_case()
    with (
        patch.object(
            turboquant_prefill, "nax_tile_work", wraps=turboquant_prefill.nax_tile_work
        ) as work,
        patch.object(
            sdpa, "nax_gpu_core_count", wraps=sdpa.nax_gpu_core_count
        ) as cores,
    ):
        assert_parity(case)
    assert work.call_count == int(selected and ready)
    assert cores.call_count == int(selected and ready)
    assert recorded_ops[0][1].get("use_turboquant", False) == (not selected)
    assert ("nax_split_hint" in recorded_ops[0][1]) == (selected and ready)


def test_workspace_plan_key_includes_kv_geometry():
    case = build_case(qlens=(128, 1), context_lens=(257, 16))
    meta = sdpa._kernel_metadata(case.ctx, None, [], case.ctx.block_tables, 16)
    plans = [
        sdpa._turboquant_prefill_plan(
            case.ctx, meta, case.ctx.block_tables, 16, queries, heads, dim
        )
        for queries, heads, dim in [(8, 2, 128), (8, 4, 256), (64, 2, 128)]
    ]
    assert all(plan is not None for plan in plans)
    assert plans[0] is not plans[1]
    assert plans[1].workspace_bytes > plans[0].workspace_bytes
    assert plans[2] is not plans[0]
    assert plans[2].workspace_bytes > plans[0].workspace_bytes
    # A runtime NAX toggle must not reuse a plan without its requested work.
    enabled = sdpa._turboquant_prefill_plan(
        case.ctx, meta, case.ctx.block_tables, 16, 8, 2, 128, compute_nax_work=True
    )
    assert enabled is not plans[0]
    assert enabled.prefill.nax_work is not None
    assert plans[0].prefill.nax_work is None
    assert enabled.fallback.nax_work is None


def test_prefill_plan_rejects_missing_query_lengths():
    case = build_case()
    meta = sdpa._kernel_metadata(case.ctx, None, [], case.ctx.block_tables, 16)
    case.ctx.cu_seqlens = None
    with pytest.raises(ValueError, match="requires cumulative query lengths"):
        sdpa._turboquant_prefill_plan(
            case.ctx, meta, case.ctx.block_tables, 16, 8, 2, 128
        )


@pytest.mark.parametrize("mode", ["", "false", "2", "AUTO"])
def test_prefill_rejects_invalid_mode(monkeypatch, mode):
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL", mode)
    with pytest.raises(
        ValueError, match="VLLM_METAL_TQ_PREFILL must be one of auto, 0, 1"
    ):
        prefill_workspace_bytes()


@pytest.mark.parametrize("mode", ["0", "auto", "1"])
@pytest.mark.parametrize("setting", ["", "-1", "1.5", "64MiB", "AUTO"])
def test_prefill_rejects_invalid_workspace_even_when_disabled(
    monkeypatch, mode, setting
):
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL", mode)
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", setting)
    with pytest.raises(
        ValueError,
        match="VLLM_METAL_TQ_PREFILL_MAX_MIB must be auto or a nonnegative integer",
    ):
        prefill_workspace_bytes()


@pytest.mark.parametrize(
    "working_mib,expected_mib",
    [(8192, 256), (32768, 704), (53088, 1088), (262144, 2048)],
)
def test_auto_workspace_scales_with_device_and_keeps_a_ceiling(
    monkeypatch, working_mib, expected_mib
):
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", "auto")
    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {"max_recommended_working_set_size": working_mib * 2**20},
    )
    assert prefill_workspace_bytes() == expected_mib * 2**20


@pytest.mark.parametrize(
    "setting,length,selected",
    [("256", 65264, True), ("256", 65280, False), ("auto", 262144, True)],
)
def test_long_context_admission_with_fused_workspace(
    monkeypatch, setting, length, selected
):
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", setting)
    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {"max_recommended_working_set_size": 53088 * 2**20},
    )
    case = build_case()
    # Inspect long-context admission without allocating a full KV pool.
    case.ctx.context_lens = [length]
    case.ctx.block_tables = [list(range((length + 15) // 16))]
    meta = sdpa._kernel_metadata(case.ctx, None, [], case.ctx.block_tables, 16)
    plan = sdpa._turboquant_prefill_plan(
        case.ctx, meta, case.ctx.block_tables, 16, 24, 4, 256
    )
    assert (plan is not None) == selected
    if plan is not None:
        assert plan.workspace_bytes <= prefill_workspace_bytes()


@pytest.mark.parametrize("block_size", [16, 32, 544])
@pytest.mark.parametrize("length", [4097, 131073])
def test_auto_cap_covers_independent_histories_and_split_routing(block_size, length):
    # Admission only: the new-query budget is small, but both prefills must
    # still read their entire independent histories, alongside a decode row.
    pages = (length + block_size - 1) // block_size
    tables = [list(range(i * pages, (i + 1) * pages)) for i in range(3)]
    allowance = workspace_upper_bound(
        max_model_len=length,
        max_num_seqs=3,
        max_num_batched_tokens=257,
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=128,
        block_size=block_size,
    )
    ctx = PagedAttentionContext(
        slot_mapping=[],
        block_tables=tables,
        context_lens=[length] * 3,
        cu_seqlens=[0, 128, 129, 257],
        tq_prefill_workspace_bytes=allowance,
    )
    meta = sdpa._kernel_metadata(ctx, 0, [], tables, block_size)
    plan = sdpa._turboquant_prefill_plan(ctx, meta, tables, block_size, 8, 2, 128)
    assert plan is not None
    assert plan.prefill.seq_lens.tolist() == [length, length]
    assert plan.fallback.seq_lens.tolist() == [length]
    assert plan.workspace_bytes <= allowance


def test_materialization_workspace_does_not_accumulate_across_layers(
    materialized_lengths, monkeypatch, prefill_backend
):
    monkeypatch.setenv("VLLM_METAL_TQ_PREFILL_MAX_MIB", "64")
    case = build_case(
        qlens=(128,), context_lens=(8192,), head_dim=256, n_heads=24, n_kv_heads=4
    )

    def chain():
        x = case.x
        for _ in range(8):
            x = sdpa.sdpa_forward(case.inner, x, case.ctx, case.cache, 0)[0][..., :32]
        return x

    def peak(reference):
        context = (
            patch.object(sdpa, "_turboquant_prefill_plan", return_value=None)
            if reference
            else nullcontext()
        )
        with context:
            mx.eval(chain())
            materialized_lengths.clear()
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
            before = mx.get_active_memory()
            mx.reset_peak_memory()
            result = chain()
            mx.eval(result)
            mx.synchronize()
            extra = mx.get_peak_memory() - before
            assert mx.all(mx.isfinite(result)).item()
            del result
            return extra

    reference_peak = peak(True)
    lane_peak = peak(False)
    assert materialized_lengths == [8192] * 8
    meta = next(iter(case.ctx.kernel_metadata_cache.values()))
    plan = next(iter(meta.tq_prefill_plans.values()))
    # Match the whole forward's ordinary allocations instead of charging
    # their platform-dependent peak against the materialization allowance.
    assert lane_peak - reference_peak < 1.25 * plan.workspace_bytes


@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "k_quant,v_quant",
    [
        ("q8_0", "q3_0"),
        ("q4_0", "q4_0"),
        ("q5_0", "q5_0"),
        ("int2", "q2_0"),
        ("uint8", "q8_0"),
    ],
)
def test_fused_materialization_matches_independent_decode(
    head_dim, dtype, k_quant, v_quant
):
    # Strided scheduler pages also exercise byte/half view offsets in each
    # packed cache field. The reference uses ordinary MLX operations.
    case = build_case(
        head_dim=head_dim,
        dtype=dtype,
        k_quant=k_quant,
        v_quant=v_quant,
        block_size=544,
        page_padding=512,
        qlens=(256,),
        context_lens=(1153,),
    )
    mx.eval(case.forward())
    meta = next(iter(case.ctx.kernel_metadata_cache.values()))
    plan = next(plan for plan in meta.tq_prefill_plans.values() if plan is not None)
    cache = case.cache
    arrays = [
        cache.key_caches[0],
        cache.value_caches[0],
        cache.key_scale_caches[0],
        cache.key_zero_caches[0],
        cache.value_scale_caches[0],
    ]
    actual = sdpa.materialize_turboquant_pages(
        *arrays,
        plan.pool_pages,
        plan.pool_offsets,
        get_v_centroids(cache.v_bits),
        head_dim=head_dim,
        key_quant_type=k_quant,
        value_bits=cache.v_bits,
        output_dtype=dtype,
    )

    def gather(array):
        return array[plan.pool_pages, plan.pool_offsets]

    expected = turbo_quant_decode(
        (gather(arrays[0]), gather(arrays[2]), gather(arrays[3])),
        (gather(arrays[1]), gather(arrays[4])),
        output_dtype=dtype,
        key_quant_type=k_quant,
        value_bits=cache.v_bits,
    )
    mx.eval(*actual, *expected)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            np.array(value.astype(mx.float32)),
            np.array(reference.astype(mx.float32)),
            atol=1e-5,
            rtol=2 * mx.finfo(dtype).eps,
        )
