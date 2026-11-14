# SPDX-License-Identifier: Apache-2.0
"""Paged-attention and lifecycle regressions for Nanbeige weight-tied loops."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import nanbeige
from mlx_lm.models.cache import make_prompt_cache

from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.context import (
    OffsetCache,
    PagedAttentionContext,
    clear_context,
    prepare_unified,
    set_context,
)
from vllm_metal.attention.impls.sdpa_wrapper import patch_sdpa_attention
from vllm_metal.distributed.pipeline import apply_pipeline_split
from vllm_metal.v1.draft_model_proposer import resolve_draft_dims
from vllm_metal.v1.model_adapter import DefaultModelAdapter, effective_num_loops
from vllm_metal.v1.model_lifecycle import ModelLifecycle

_BLOCK_SIZE = 16


def _make_args(
    *,
    num_loops: int = 2,
    skip_loop_final_norm: bool = False,
    loop_loss_weights: list[float] | None = None,
) -> nanbeige.ModelArgs:
    return nanbeige.ModelArgs(
        model_type="nanbeige",
        hidden_size=128,
        num_hidden_layers=3,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        rms_norm_eps=1e-5,
        vocab_size=64,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        num_loops=num_loops,
        skip_loop_final_norm=skip_loop_final_norm,
        loop_loss_weights=loop_loss_weights,
    )


def _clone_model(
    ref_model: nanbeige.Model,
    args: nanbeige.ModelArgs,
    dtype: mx.Dtype,
) -> nanbeige.Model:
    paged_model = nanbeige.Model(args)
    paged_model.load_weights(list(tree_flatten(ref_model.parameters())), strict=True)
    paged_model.set_dtype(dtype)
    mx.eval(paged_model.parameters())
    return paged_model


@pytest.mark.parametrize(
    ("dtype", "tol"),
    [(mx.float32, 2e-5), (mx.bfloat16, 1e-2)],
    ids=["fp32", "bf16"],
)
@pytest.mark.parametrize(
    ("skip_loop_final_norm", "loop_loss_weights"),
    [
        (False, None),
        (True, None),
        (False, [0.3]),
    ],
    ids=["norm_each_loop", "norm_final_loop_only", "loop_loss_weights"],
)
def test_paged_looped_prefill_and_decode_matches_mlx_lm(
    dtype: mx.Dtype,
    tol: float,
    skip_loop_final_norm: bool,
    loop_loss_weights: list[float] | None,
) -> None:
    """Each weight-tied loop must write and read its own paged KV cache slice."""
    mx.random.seed(7)
    args = _make_args(
        num_loops=1 if loop_loss_weights is not None else 2,
        skip_loop_final_norm=skip_loop_final_norm,
        loop_loss_weights=loop_loss_weights,
    )
    ref_model = nanbeige.Model(args)
    ref_model.set_dtype(dtype)
    mx.eval(ref_model.parameters())
    paged_model = _clone_model(ref_model, args, dtype)

    total_kv_layers = args.effective_num_loops * args.num_hidden_layers
    assert total_kv_layers == 6
    kv_cache = MetalPagedKVCache(
        num_layers=total_kv_layers,
        num_kv_heads=args.num_key_value_heads,
        head_dim=args.head_dim,
        num_blocks=4,
        block_size=_BLOCK_SIZE,
        dtype=dtype,
    )
    assert patch_sdpa_attention(paged_model, kv_cache, _BLOCK_SIZE) == 3

    prompt = [3, 11, 7, 19, 23, 5, 17]
    ref_cache = make_prompt_cache(ref_model)
    assert len(ref_cache) == total_kv_layers
    expected_prefill = ref_model(mx.array([prompt], dtype=mx.int32), cache=ref_cache)
    mx.eval(expected_prefill)

    offset_caches = [OffsetCache(0) for _ in range(total_kv_layers)]
    prepare_unified(
        decode_requests=[],
        prefill_requests=[([0], len(prompt), 0)],
        block_size=_BLOCK_SIZE,
    )
    try:
        actual_prefill = paged_model(
            mx.array([prompt], dtype=mx.int32), cache=offset_caches
        )
        mx.eval(actual_prefill)
    finally:
        clear_context()

    assert mx.allclose(actual_prefill, expected_prefill, atol=tol, rtol=tol).item()

    seq_len = len(prompt)
    for tok in (29, 31, 41):
        expected_dec = ref_model(mx.array([[tok]], dtype=mx.int32), cache=ref_cache)
        mx.eval(expected_dec)

        step_caches = [OffsetCache(seq_len) for _ in range(total_kv_layers)]
        prepare_unified(
            decode_requests=[([0], seq_len)],
            prefill_requests=[],
            block_size=_BLOCK_SIZE,
        )
        try:
            actual_dec = paged_model(
                mx.array([[tok]], dtype=mx.int32), cache=step_caches
            )
            mx.eval(actual_dec)
        finally:
            clear_context()

        assert mx.allclose(actual_dec, expected_dec, atol=tol, rtol=tol).item()
        seq_len += 1


def test_chunked_prefill_and_batched_decode_matches_mlx_lm() -> None:
    """Loop call counters must advance once per packed forward and reset next step."""
    mx.random.seed(13)
    args = _make_args(num_loops=2)
    ref_model = nanbeige.Model(args)
    mx.eval(ref_model.parameters())
    paged_model = _clone_model(ref_model, args, mx.float32)

    total_kv_layers = args.effective_num_loops * args.num_hidden_layers
    kv_cache = MetalPagedKVCache(
        num_layers=total_kv_layers,
        num_kv_heads=args.num_key_value_heads,
        head_dim=args.head_dim,
        num_blocks=4,
        block_size=_BLOCK_SIZE,
        dtype=mx.float32,
    )
    patch_sdpa_attention(paged_model, kv_cache, _BLOCK_SIZE)

    seq_a = [2, 4, 6, 8, 10, 12, 14]
    seq_b = [1, 3, 5, 7, 9]
    ref_cache_a = make_prompt_cache(ref_model)
    ref_cache_b = make_prompt_cache(ref_model)
    exp_a = ref_model(mx.array([seq_a], dtype=mx.int32), cache=ref_cache_a)
    exp_b = ref_model(mx.array([seq_b], dtype=mx.int32), cache=ref_cache_b)
    mx.eval(exp_a, exp_b)

    # Chunk 1 of seq_a (first 4 tokens in block 0), then chunk 2 + full prefill of seq_b.
    prepare_unified(
        decode_requests=[],
        prefill_requests=[([0], 4, 0)],
        block_size=_BLOCK_SIZE,
    )
    try:
        mx.eval(
            paged_model(
                mx.array([seq_a[:4]], dtype=mx.int32),
                cache=[OffsetCache(0) for _ in range(total_kv_layers)],
            )
        )
    finally:
        clear_context()

    packed = seq_a[4:] + seq_b
    prepare_unified(
        decode_requests=[],
        prefill_requests=[([0], 3, 4), ([1], len(seq_b), 0)],
        block_size=_BLOCK_SIZE,
    )
    try:
        actual_packed = paged_model(
            mx.array([packed], dtype=mx.int32),
            cache=[OffsetCache(0) for _ in range(total_kv_layers)],
        )
        mx.eval(actual_packed)
    finally:
        clear_context()

    assert mx.allclose(actual_packed[:, :3, :], exp_a[:, 4:, :], atol=2e-5).item()
    assert mx.allclose(actual_packed[:, 3:, :], exp_b, atol=2e-5).item()

    # Batched decode for both sequences in one forward pass.
    exp_dec_a = ref_model(mx.array([[21]], dtype=mx.int32), cache=ref_cache_a)
    exp_dec_b = ref_model(mx.array([[22]], dtype=mx.int32), cache=ref_cache_b)
    mx.eval(exp_dec_a, exp_dec_b)

    max_len = max(len(seq_a), len(seq_b))
    prepare_unified(
        decode_requests=[([0], len(seq_a)), ([1], len(seq_b))],
        prefill_requests=[],
        block_size=_BLOCK_SIZE,
    )
    try:
        actual_dec = paged_model(
            mx.array([[21, 22]], dtype=mx.int32),
            cache=[OffsetCache(max_len) for _ in range(total_kv_layers)],
        )
        mx.eval(actual_dec)
    finally:
        clear_context()

    assert mx.allclose(actual_dec[:, 0:1, :], exp_dec_a, atol=2e-5).item()
    assert mx.allclose(actual_dec[:, 1:2, :], exp_dec_b, atol=2e-5).item()


def test_patched_model_preserves_unpaged_forward_and_selective_logits() -> None:
    """Unpaged fallback and split-head selective logits must preserve loop behavior."""
    mx.random.seed(19)
    args = _make_args(num_loops=2)
    model = nanbeige.Model(args)
    tokens = mx.array([[1, 5, 9, 13]], dtype=mx.int32)
    expected = model(tokens)
    mx.eval(expected)

    patch_sdpa_attention(model, object(), _BLOCK_SIZE)
    actual = model(tokens)
    mx.eval(actual)
    assert mx.array_equal(actual, expected).item()

    adapter = DefaultModelAdapter()
    assert adapter.supports_selective_logits(model) is True
    assert adapter.supports_intermediate_forward(model) is True

    indices = mx.array([1, 3], dtype=mx.int32)
    selective = adapter.target_forward(model, tokens, logits_indices=indices).logits
    mx.eval(selective)
    assert mx.allclose(selective, expected[0, indices, :], atol=1e-6).item()


def test_looped_patch_guardrails() -> None:
    """Mismatched cache depth, remapped caches, and excess loop calls fail loud."""
    args = _make_args(num_loops=2)
    model = nanbeige.Model(args)
    wrong_cache = MetalPagedKVCache(
        num_layers=args.num_hidden_layers,
        num_kv_heads=args.num_key_value_heads,
        head_dim=args.head_dim,
        num_blocks=1,
        block_size=_BLOCK_SIZE,
        dtype=mx.float32,
    )
    with pytest.raises(ValueError, match="expects 6 KV cache layers"):
        patch_sdpa_attention(model, wrong_cache, _BLOCK_SIZE)

    valid_cache = MetalPagedKVCache(
        num_layers=6,
        num_kv_heads=args.num_key_value_heads,
        head_dim=args.head_dim,
        num_blocks=1,
        block_size=_BLOCK_SIZE,
        dtype=mx.float32,
    )
    with pytest.raises(NotImplementedError, match="cache_idx_map / only_layers"):
        patch_sdpa_attention(
            model, valid_cache, _BLOCK_SIZE, cache_idx_map={0: 0, 1: 1, 2: 2}
        )
    with pytest.raises(NotImplementedError, match="cache_idx_map / only_layers"):
        patch_sdpa_attention(model, valid_cache, _BLOCK_SIZE, only_layers=[0])

    patch_sdpa_attention(model, valid_cache, _BLOCK_SIZE)
    wrapper = model.model.layers[0].self_attn
    ctx = PagedAttentionContext(
        slot_mapping=[0],
        block_tables=[[0]],
        context_lens=[1],
        offsets=[0],
        cu_seqlens=[0, 1],
    )
    set_context(ctx)
    try:
        assert wrapper._resolve_paged_cache_idx(ctx) == 0
        assert wrapper._resolve_paged_cache_idx(ctx) == 3
        with pytest.raises(RuntimeError, match="called 3 times"):
            wrapper._resolve_paged_cache_idx(ctx)
    finally:
        clear_context()


@pytest.mark.parametrize(
    "bad_override",
    [
        {"enable_double_loop_split": True},
        {"loop_share_kv": True},
        {"enable_depth_attention": True},
        {"enable_hyper_connection": True},
        {"enable_mhc": True},
        {"qk_layernorm": True},
        {"emb_neighbor_num": 2},
    ],
)
def test_unsupported_nanbeige_variants_fail_fast(
    bad_override: dict[str, object],
) -> None:
    base = {
        "model_type": "nanbeige",
        "num_hidden_layers": 22,
        "num_loops": 2,
        "hidden_size": 3072,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    with pytest.raises(NotImplementedError):
        effective_num_loops(base | bad_override)


def test_lifecycle_draft_and_pp_loop_contracts() -> None:
    """Lifecycle and draft sizing multiply physical layers by effective_num_loops; PP rejects loops."""
    runner = SimpleNamespace(
        model_args={
            "model_type": "nanbeige",
            "num_hidden_layers": 22,
            "num_loops": 2,
            "hidden_size": 3072,
            "num_attention_heads": 48,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        model=None,
        is_mla=False,
        is_hybrid=False,
        pp=None,
    )
    ModelLifecycle(runner, DefaultModelAdapter()).resolve_model_dims()
    assert runner.num_layers == 44
    assert runner.num_kv_cache_layers == 44
    assert runner.num_kv_heads == 8
    assert runner.head_dim == 128

    draft_cfg = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            num_loops=2,
            layer_types=None,
        ),
        get_sliding_window=lambda: None,
        get_total_num_hidden_layers=lambda: 22,
        get_num_kv_heads=lambda _pc: 8,
        get_head_size=lambda: 128,
    )
    dims = resolve_draft_dims(
        SimpleNamespace(draft_model_config=draft_cfg),
        parallel_config=SimpleNamespace(),
    )
    assert dims.num_layers == 44
    assert dims.num_kv_heads == 8
    assert dims.head_dim == 128

    model = nanbeige.Model(_make_args(num_loops=2))
    pp_two_stage = SimpleNamespace(size=2, rank=0, is_last=False)
    with pytest.raises(NotImplementedError, match="weight-tied looped models"):
        apply_pipeline_split(model, pp_two_stage)  # type: ignore[arg-type]
