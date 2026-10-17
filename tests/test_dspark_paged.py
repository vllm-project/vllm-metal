# SPDX-License-Identifier: Apache-2.0
"""DSpark's slot-zero proposals over real shared Metal KV and ragged pages."""

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from tests.test_dflash import _config, _torch_forward
from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.v1.dflash_paged import DFlashPagedCache
from vllm_metal.v1.dspark import DSparkConfig, DSparkModel
from vllm_metal.v1.dspark_paged import DSparkPagedCache


def make_cache(
    dtype=mx.float16,
    *,
    block_size=16,
    confidence=True,
    with_markov=True,
    quantized=False,
    context_windows=(),
    causal_blocks=False,
    max_model_len=64,
):
    cfg = DSparkConfig(
        backbone=replace(
            _config(),
            hidden_size=64,
            intermediate_size=64 if quantized else 48,
            head_dim=64,
            max_position_embeddings=max_model_len,
            block_size=7,
            target_layer_ids=(0, 2, 3),
        ),
        markov_rank=4,
        enable_confidence_head=confidence,
        confidence_head_with_markov=with_markov,
        context_windows=context_windows,
        causal_blocks=causal_blocks,
    )
    model = DSparkModel(cfg)
    # Keep exact-ID assertions well-conditioned across low-precision kernels.
    # The full logits/confidence still compare against independent attention;
    # real-checkpoint qualification separately rejects near-tie divergences.
    model.markov_head.markov_w1.weight = mx.eye(4)[mx.arange(64) % 4]
    correction = mx.zeros((64, 4))
    correction[mx.array([1, 2, 3, 0]), mx.arange(4)] = 8
    model.markov_head.markov_w2.weight = correction
    model.set_dtype(dtype)
    if quantized:
        model.quantize_draft_linears()
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16 if dtype == mx.float16 else torch.bfloat16,
    )
    names = ("target", "dspark_layers.0.self_attn", "dspark_layers.1.self_attn")
    num_blocks = max(12, 1 + 2 * ((max_model_len + block_size - 1) // block_size))
    size = num_blocks * spec.page_size_bytes
    storage = KVCacheStorage(
        KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=list(names), kv_cache_spec=spec)
            ],
            kv_cache_tensors=[
                KVCacheTensor(
                    size=len(names) * size,
                    layers=list(names),
                    layer_stride=size,
                    block_stride=spec.page_size_bytes,
                )
            ],
            kv_cache_layout="LBNHC",
        )
    )
    storage.tensors["target"].fill_(7)
    for name in names[1:]:
        storage.tensors[name].fill_(float("nan"))
    return model, DSparkPagedCache(
        model, storage, names[1:], max_model_len=max_model_len
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("block_size", [8, 16, 32])
@pytest.mark.parametrize("width", [1, 3, 7])
@pytest.mark.parametrize(
    "confidence,with_markov", [(False, False), (True, False), (True, True)]
)
def test_ragged_proposals_commit_verified_features_and_reuse_pages(
    dtype, block_size, width, confidence, with_markov
):
    _check_ragged_proposals(dtype, block_size, width, confidence, with_markov)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("width", [1, 7])
def test_candidate_limited_proposals_commit_features_and_reuse_pages(dtype, width):
    _check_ragged_proposals(dtype, 16, width, True, True, draft_topk=8)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("width", [3, 7])
@pytest.mark.parametrize("quantized", [False, True])
def test_single_request_with_head_major_projections(
    monkeypatch, dtype, width, quantized
):
    model, cache = make_cache(dtype, quantized=quantized)
    reference = model
    if quantized:
        from tests.test_dspark_quantization import dequantized_model

        reference = dequantized_model(model)

    def head_major(project):
        def forward(*args):
            # RoPE may return head-major storage; transposing back to token
            # order does not guarantee dense rows for the Metal kernels.
            return tuple(mx.contiguous(a) for a in project(*args))

        return forward

    for layer in model.backbone.layers:
        attn = layer.self_attn
        monkeypatch.setattr(attn, "project_block", head_major(attn.project_block))

    draft = cache.compile_draft(num_draft_tokens=width)
    table, anchor = [5, 2, 7, 1], mx.array([11])
    for length in (7, 17, 41):
        features = [mx.random.normal((1, length, 64)).astype(dtype) for _ in range(3)]
        cache.write_context([f[0] for f in features], [(table, 0, length)])
        actual = draft(anchor, [(table, length)])
        hidden = _torch_forward(
            reference.backbone, reference.block_embeddings(anchor, width), features
        )
        expected = reference.greedy_proposal(mx.array(hidden).astype(dtype), anchor)
        mx.eval(actual, expected, *cache.storage.buffers)
        np.testing.assert_array_equal(np.array(actual[0]), np.array(expected[0]))
        tolerance = 0.04 if dtype == mx.bfloat16 else 0.006
        for observed, wanted in zip(actual[1:], expected[1:], strict=True):
            np.testing.assert_allclose(
                np.array(observed.astype(mx.float32)),
                np.array(wanted.astype(mx.float32)),
                atol=tolerance,
                rtol=tolerance,
            )


def _check_ragged_proposals(
    dtype,
    block_size,
    width,
    confidence,
    with_markov,
    draft_topk=None,
    quantized=False,
    corrected_logits=True,
    context_windows=(),
    causal_blocks=False,
):
    model, cache = make_cache(
        dtype,
        block_size=block_size,
        confidence=confidence,
        with_markov=with_markov,
        quantized=quantized,
        context_windows=context_windows,
        causal_blocks=causal_blocks,
    )
    if draft_topk is not None and not quantized:
        # Separate candidate scores to avoid an unstable top-k boundary in
        # reduced precision. The independent attention comparison remains.
        model.lm_head.weight = mx.zeros((64, 64), dtype=dtype)
        model.lm_head.weight[:, 0] = (mx.arange(64).astype(dtype) - 32) / 32
    reference_model = model
    if quantized:
        from tests.test_dspark_quantization import dequantized_model

        reference_model = dequantized_model(model)
    draft = cache.compile_draft(
        num_draft_tokens=width,
        draft_topk=draft_topk,
        corrected_logits=corrected_logits,
    )
    tables = [[5, 2, 7, 1], [9, 3, 8, 4]]
    lengths = [block_size - 1, min(2 * block_size - 2, 47)]
    features = [
        [mx.random.normal((1, n, 64)).astype(dtype) for _ in range(3)] for n in lengths
    ]
    tolerance = 0.04 if dtype == mx.bfloat16 else 0.006
    for step in range(3):
        if step == 0 or step == 2:
            # A fresh request may reuse pages containing a prior draft/rejected tail.
            if step == 2:
                lengths = [1, 2]
                features = [
                    [mx.random.normal((1, n, 64)).astype(dtype) for _ in range(3)]
                    for n in lengths
                ]
            tails = features
            spans = [(t, 0, n) for t, n in zip(tables, lengths, strict=True)]
        else:
            # Only the anchor + accepted target rows become committed features.
            counts = [1, min(width + 1, 3)]
            tails = [
                [mx.random.normal((1, n, 64)).astype(dtype) for _ in range(3)]
                for n in counts
            ]
            spans = [
                (t, start, n)
                for t, start, n in zip(tables, lengths, counts, strict=True)
            ]
            features = [
                [
                    mx.concatenate([old, tail], axis=1)
                    for old, tail in zip(row, extra, strict=True)
                ]
                for row, extra in zip(features, tails, strict=True)
            ]
            lengths = [start + n for start, n in zip(lengths, counts, strict=True)]
        cache.write_context(
            [mx.concatenate([f[i][0] for f in tails]) for i in range(3)], spans
        )
        anchors = mx.array([11 + step, 21 + step])
        actual = draft(anchors, list(zip(tables, lengths, strict=True)))
        mx.eval(actual, *cache.storage.buffers)
        for row, feature in enumerate(features):
            anchor = anchors[row : row + 1]
            independent_hidden = _torch_forward(
                reference_model.backbone,
                reference_model.block_embeddings(anchor, width),
                feature,
                context_windows=context_windows or None,
                causal_blocks=causal_blocks,
            )
            expected = model.greedy_proposal(
                mx.array(independent_hidden).astype(dtype),
                anchor,
                draft_topk=draft_topk,
                corrected_logits=corrected_logits,
            )
            for observed, reference in zip(actual[1:], expected[1:], strict=True):
                if reference is None:
                    assert observed is None
                else:
                    np.testing.assert_allclose(
                        np.array(observed[row : row + 1].astype(mx.float32)),
                        np.array(reference.astype(mx.float32)),
                        atol=tolerance,
                        rtol=tolerance,
                    )
            np.testing.assert_array_equal(
                np.array(actual[0][row : row + 1]), np.array(expected[0])
            )
        assert torch.all(cache.storage.tensors["target"] == 7)
        for name in ("dspark_layers.0.self_attn", "dspark_layers.1.self_attn"):
            assert torch.all(torch.isnan(cache.storage.tensors[name][[0, 6, 10, 11]]))


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("width", [1, 7])
@pytest.mark.parametrize("windows", [(8, 8), (None, 16), (64, 64)])
@pytest.mark.parametrize("causal_blocks", [False, True])
def test_speculators_windows_preserve_ragged_context_and_page_reuse(
    dtype, width, windows, causal_blocks
):
    _check_ragged_proposals(
        dtype,
        16,
        width,
        True,
        True,
        context_windows=windows,
        causal_blocks=causal_blocks,
    )


@pytest.mark.parametrize("draft_topk", [None, 8])
def test_ragged_proposals_without_corrected_logits(draft_topk):
    # The serving proposer compiles with corrected_logits=False; skipping the
    # dense per-position -inf fill must not change ragged, multi-step or
    # page-reuse proposals, with or without a candidate limit.
    _check_ragged_proposals(
        mx.float16,
        16,
        7,
        True,
        True,
        draft_topk=draft_topk,
        corrected_logits=False,
    )


def test_one_token_proposal_uses_one_lookahead_slot_at_page_boundary():
    model, cache = make_cache()
    features = [mx.random.normal((1, 15, 64)).astype(mx.float16) for _ in range(3)]
    cache.write_context([f[0] for f in features], [([2], 0, 15)])
    actual = cache.compile_draft(num_draft_tokens=1)(mx.array([4]), [([2], 15)])
    mx.eval(actual, *cache.storage.buffers)
    expected = model.draft(mx.array([4]), features, num_draft_tokens=1)
    np.testing.assert_allclose(
        np.array(actual[1].astype(mx.float32)),
        np.array(expected[1].astype(mx.float32)),
        atol=0.006,
        rtol=0.006,
    )
    assert actual[0].shape == (1, 1)


@pytest.mark.parametrize(
    "width,blocks,length",
    [
        (7, [2], 10),
        (1, [2], 16),
        (3, [-1, 2], 15),
        (3, [12, 2], 15),
        (7, [2, 3, 4, 5], 58),
    ],
)
def test_invalid_lookahead_rejected_before_writing(width, blocks, length):
    _, cache = make_cache()
    with pytest.raises((ValueError, RuntimeError)):
        cache.compile_draft(num_draft_tokens=width)(mx.array([4]), [(blocks, length)])
    assert torch.all(cache.storage.tensors["target"] == 7)
    for name in ("dspark_layers.0.self_attn", "dspark_layers.1.self_attn"):
        assert torch.all(torch.isnan(cache.storage.tensors[name]))


@pytest.mark.parametrize("width", [0, 8, True, 1.5])
def test_invalid_width_rejected_before_compilation(width):
    _, cache = make_cache()
    with pytest.raises(ValueError, match="width"):
        cache.compile_draft(num_draft_tokens=width)


@pytest.mark.parametrize("draft_topk", [0, -1, 65, True, 1.5])
def test_invalid_candidate_limit_rejected_before_compilation(draft_topk):
    _, cache = make_cache()
    with pytest.raises(ValueError, match="draft_topk"):
        cache.compile_draft(num_draft_tokens=7, draft_topk=draft_topk)
    for name in ("dspark_layers.0.self_attn", "dspark_layers.1.self_attn"):
        assert torch.all(torch.isnan(cache.storage.tensors[name]))


def test_feature_precision_rejected_before_any_cache_write():
    _, cache = make_cache()
    with pytest.raises(ValueError, match="precision"):
        cache.write_context([mx.zeros((1, 64), dtype=mx.float32)] * 3, [([2], 0, 1)])
    for name in ("dspark_layers.0.self_attn", "dspark_layers.1.self_attn"):
        assert torch.all(torch.isnan(cache.storage.tensors[name]))


@pytest.mark.parametrize("windows", [(), (8, 8)])
def test_compiled_block_replays_without_retracing_as_context_grows(
    monkeypatch, windows
):
    model, cache = make_cache(context_windows=windows, causal_blocks=bool(windows))
    original = mx.compile
    traces = []

    def compile_counted(function):
        def counted(*args):
            traces.append(True)
            return function(*args)

        return original(counted)

    monkeypatch.setattr(mx, "compile", compile_counted)
    draft = cache.compile_draft(num_draft_tokens=7)
    tables = [5, 2, 7, 1]
    for step, length in enumerate((1, 7, 17, 33, 1)):
        features = [
            mx.random.normal((1, length, 64)).astype(mx.float16) for _ in range(3)
        ]
        cache.write_context([f[0] for f in features], [(tables, 0, length)])
        anchor = mx.array([step + 4])
        result = draft(anchor, [(tables, length)])
        mx.eval(result, *cache.storage.buffers)
        expected = model.draft(anchor, features, num_draft_tokens=7)
        np.testing.assert_array_equal(np.array(result[0]), np.array(expected[0]))
        np.testing.assert_allclose(
            np.array(result[1].astype(mx.float32)),
            np.array(expected[1].astype(mx.float32)),
            atol=0.006,
            rtol=0.006,
        )
    assert len(traces) == 1


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("width", [1, 7])
@pytest.mark.parametrize("windows", [(2048, 2048), (8, 2048), (None, 2048)])
@pytest.mark.parametrize("causal", [False, True])
def test_window_buckets_preserve_ragged_attention_during_growth_and_reuse(
    monkeypatch, dtype, width, windows, causal
):
    from vllm_metal.v1 import block_draft_paged

    model, cache = make_cache(
        dtype,
        context_windows=windows,
        causal_blocks=causal,
        max_model_len=2304,
    )
    tables = [list(range(1 + row, cache.cache.num_blocks, 2)) for row in range(2)]
    tables[1].reverse()
    gathered_rows = []
    gather = block_draft_paged._gather_window_kv

    def record_gather(*args, **kwargs):
        keys, values = gather(*args, **kwargs)
        gathered_rows.append(keys.shape[0] * keys.shape[1])
        return keys, values

    monkeypatch.setattr(block_draft_paged, "_gather_window_kv", record_gather)
    draft = cache.compile_draft(num_draft_tokens=width)
    # Cross shape/reduction boundaries, then rewind into reused, poisoned pages.
    for step, span in enumerate(
        (24, 256, 257, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 2112, 24)
    ):
        lengths = (span - width, 17)
        for name, tensor in cache.storage.tensors.items():
            if name != "target":
                tensor.fill_(float("nan"))
        features = [
            [mx.random.normal((1, length, 64)).astype(dtype) for _ in range(3)]
            for length in lengths
        ]
        cache.write_context(
            [mx.concatenate([row[i][0] for row in features]) for i in range(3)],
            [(table, 0, n) for table, n in zip(tables, lengths, strict=True)],
        )
        anchors = mx.array([11, 12])
        actual = draft(anchors, list(zip(tables, lengths, strict=True)))
        mx.eval(actual, *cache.storage.buffers)
        for row, feature in enumerate(features):
            hidden = _torch_forward(
                model.backbone,
                model.block_embeddings(anchors[row : row + 1], width),
                feature,
                context_windows=windows,
                causal_blocks=causal,
            )
            expected = model.greedy_proposal(
                mx.array(hidden).astype(dtype), anchors[row : row + 1]
            )
            mx.eval(expected)
            np.testing.assert_array_equal(
                np.array(actual[0][row : row + 1]), np.array(expected[0])
            )
            tolerance = 0.04 if dtype == mx.bfloat16 else 0.006
            for observed, wanted in zip(actual[1:], expected[1:], strict=True):
                np.testing.assert_allclose(
                    np.array(observed[row : row + 1].astype(mx.float32)),
                    np.array(wanted.astype(mx.float32)),
                    atol=tolerance,
                    rtol=tolerance,
                )
        if step == 0:
            # Short requests must not gather the entire trained 2K window.
            assert gathered_rows and max(gathered_rows) <= 2 * 256
        gathered_rows.clear()
        assert torch.all(cache.storage.tensors["target"] == 7)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block_size", [8, 16, 32])
@pytest.mark.parametrize("head_dim", [64, 96, 128, 256, 512])
def test_window_gather_preserves_strided_values_and_zeros_poisoned_padding(
    dtype, block_size, head_dim
):
    from vllm_metal.tensor_bridge import torch_to_mlx
    from vllm_metal.v1.block_draft_paged import _gather_window_kv

    width, window, span, heads = 3, 8, 14, 2
    lengths = [block_size + 3, 2]
    tables = [[6, 2, 4, 1], [3, 5, 1, 0]]
    # K and V have different offsets, padding and noncontiguous head strides.
    backing = torch.full(
        (7, block_size, heads, 2 * (head_dim + 16)), float("nan"), dtype=dtype
    )
    key = backing[..., :head_dim]
    value = backing[..., head_dim + 16 : 2 * head_dim + 16]
    expected = np.zeros((2, 2, span, heads, head_dim), dtype=np.float32)
    for batch, length in enumerate(lengths):
        start = max(length - window, 0)
        for row, position in enumerate(range(start, length + width)):
            block = tables[batch][position // block_size]
            offset = position % block_size
            data = torch.arange(heads * head_dim).reshape(heads, head_dim) % 251
            key[block, offset] = (data + row + 1).to(dtype)
            value[block, offset] = (-data - row - 1).to(dtype)
            expected[0, batch, row] = key[block, offset].float().numpy()
            expected[1, batch, row] = value[block, offset].float().numpy()
    actual = _gather_window_kv(
        torch_to_mlx(key),
        torch_to_mlx(value),
        mx.repeat(mx.array(tables, dtype=mx.int32), 2, axis=1)[:, ::2],
        mx.repeat(mx.array(lengths, dtype=mx.int32), 2)[::2],
        width=width,
        window=window,
        span=span,
        head_dim=head_dim,
        block_size=block_size,
    )
    for got, wanted in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(np.array(got.astype(mx.float32)), wanted)


def test_window_gather_reads_pooled_pages_past_32_bit_offsets():
    from vllm_metal.metal import get_ops
    from vllm_metal.tensor_bridge import torch_to_mlx
    from vllm_metal.v1.block_draft_paged import _gather_window_kv

    block_size, head_dim, stride = 16, 64, 1 << 16
    blocks = (1 << 15) + 2
    # Byte storage crosses the same INT32_MAX element offset without requiring
    # a >4 GiB shared buffer. Only the two written pages are resident.
    backing = torch.empty((blocks, stride), dtype=torch.uint8)
    cache = get_ops().as_strided(
        torch_to_mlx(backing),
        (blocks, block_size, 1, head_dim),
        (stride, stride // block_size, head_dim, 1),
        0,
    )
    backing[1] = 1
    backing[-1] = 2
    key, value = _gather_window_kv(
        cache,
        cache,
        mx.array([[1, blocks - 1]], dtype=mx.int32),
        mx.array([25], dtype=mx.int32),
        width=7,
        window=2048,
        span=32,
        head_dim=head_dim,
        block_size=block_size,
    )
    expected = np.broadcast_to(
        np.repeat([1, 2], block_size)[None, :, None, None], (1, 32, 1, head_dim)
    )
    for actual in (key, value):
        np.testing.assert_array_equal(np.array(actual), expected)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("block_size", [8, 16, 32])
@pytest.mark.parametrize("width", [1, 7])
@pytest.mark.parametrize("windows", [(), (8, 8)])
def test_q4_backbone_preserves_committed_features_and_page_reuse(
    dtype, block_size, width, windows
):
    _check_ragged_proposals(
        dtype,
        block_size,
        width,
        True,
        True,
        quantized=True,
        context_windows=windows,
        causal_blocks=bool(windows),
    )


@pytest.mark.parametrize("drafter", ["dflash", "dspark"])
@pytest.mark.parametrize("layout", ["duplicate", "separate_groups", "mixed_precision"])
def test_binding_rejects_layers_that_cannot_share_one_block_table(drafter, layout):
    model, valid = make_cache()
    names = ("dspark_layers.0.self_attn", "dspark_layers.1.self_attn")
    spec = valid.storage.specs[names[0]]
    storage = valid.storage
    if layout == "duplicate":
        names = (names[0], names[0])
    else:
        layer_bytes = 12 * spec.page_size_bytes
        if layout == "separate_groups":
            groups = [
                KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec)
                for name in names
            ]
        else:
            groups = [
                KVCacheGroupSpec(
                    layer_names=list(names),
                    kv_cache_spec=UniformTypeKVCacheSpecs(
                        block_size=16,
                        kv_cache_specs={
                            names[0]: spec,
                            names[1]: replace(spec, dtype=torch.bfloat16),
                        },
                    ),
                )
            ]
        storage = KVCacheStorage(
            KVCacheConfig(
                num_blocks=12,
                kv_cache_groups=groups,
                kv_cache_tensors=[
                    KVCacheTensor(
                        size=2 * layer_bytes,
                        layers=[name],
                        layer_stride=layer_bytes,
                        block_stride=spec.page_size_bytes,
                        # Different groups may overlay the same physical bytes;
                        # they require different scheduler block tables.
                        offset=0 if layout == "separate_groups" else i * layer_bytes,
                    )
                    for i, name in enumerate(names)
                ],
                kv_cache_layout="LBNHC",
            )
        )
        if layout == "separate_groups":
            assert (
                storage.tensors[names[0]].data_ptr()
                == storage.tensors[names[1]].data_ptr()
            )
    binding = DSparkPagedCache if drafter == "dspark" else DFlashPagedCache
    backbone = model if drafter == "dspark" else model.backbone
    with pytest.raises(ValueError):
        binding(backbone, storage, names, max_model_len=64)
