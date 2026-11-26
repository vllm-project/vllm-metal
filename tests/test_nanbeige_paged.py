# SPDX-License-Identifier: Apache-2.0
"""Paged attention for Nanbeige 4.2's weight-tied loops."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import nanbeige
from mlx_lm.models.cache import make_prompt_cache

from tests.stub_runner import make_kv_cache
from vllm_metal.attention.context import OffsetCache, clear_context, prepare_unified
from vllm_metal.attention.impls.sdpa_wrapper import patch_sdpa_attention
from vllm_metal.v1.model_adapter import DefaultModelAdapter
from vllm_metal.v1.model_lifecycle import ModelLifecycle

_BLOCK_SIZE = 16


def _make_args(
    *, num_loops: int = 2, loop_loss_weights: list[float] | None = None
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
        loop_loss_weights=loop_loss_weights,
    )


def _clone_model(
    ref_model: nanbeige.Model, args: nanbeige.ModelArgs, dtype: mx.Dtype
) -> nanbeige.Model:
    paged_model = nanbeige.Model(args)
    paged_model.load_weights(list(tree_flatten(ref_model.parameters())), strict=True)
    paged_model.set_dtype(dtype)
    mx.eval(paged_model.parameters())
    return paged_model


def _kv_cache(
    args: nanbeige.ModelArgs, num_layers: int, dtype: mx.Dtype = mx.float32
) -> object:
    return make_kv_cache(
        num_layers=num_layers,
        num_kv_heads=args.num_key_value_heads,
        head_dim=args.head_dim,
        num_blocks=4,
        block_size=_BLOCK_SIZE,
        dtype=dtype,
    )


def _offset_caches(num_layers: int, offset: int = 0) -> list[OffsetCache]:
    """Indexed per-layer caches, as the model runner builds them."""
    return [OffsetCache(offset, index=i) for i in range(num_layers)]


def _paged_forward(
    model: nanbeige.Model,
    tokens: list[int],
    caches: list[OffsetCache],
    *,
    decode_requests: list,
    prefill_requests: list,
) -> mx.array:
    prepare_unified(
        decode_requests=decode_requests,
        prefill_requests=prefill_requests,
        block_size=_BLOCK_SIZE,
    )
    try:
        out = model(mx.array([tokens], dtype=mx.int32), cache=caches)
        mx.eval(out)
    finally:
        clear_context()
    return out


@pytest.mark.parametrize(
    ("dtype", "tol"),
    [(mx.float32, 2e-5), (mx.bfloat16, 2e-2)],
    ids=["fp32", "bf16"],
)
@pytest.mark.parametrize(
    ("num_loops", "loop_loss_weights"),
    [(2, None), (1, [0.3])],
    ids=["num_loops", "loop_loss_weights"],
)
def test_paged_looped_prefill_and_decode_matches_mlx_lm(
    dtype: mx.Dtype,
    tol: float,
    num_loops: int,
    loop_loss_weights: list[float] | None,
) -> None:
    """Each loop of a layer must write and read its own paged KV layer.

    ``loop_loss_weights`` sets the loop count inside mlx_lm, so the paged
    model must take its count from the loaded model, not ``num_loops``.
    """
    mx.random.seed(7)
    args = _make_args(num_loops=num_loops, loop_loss_weights=loop_loss_weights)
    ref_model = nanbeige.Model(args)
    ref_model.set_dtype(dtype)
    mx.eval(ref_model.parameters())
    paged_model = _clone_model(ref_model, args, dtype)

    kv_layers = 2 * args.num_hidden_layers
    kv_cache = _kv_cache(args, kv_layers, dtype)
    assert patch_sdpa_attention(paged_model, kv_cache, _BLOCK_SIZE) == 3

    prompt = [3, 11, 7, 19, 23, 5, 17]
    ref_cache = make_prompt_cache(ref_model)
    expected = ref_model(mx.array([prompt], dtype=mx.int32), cache=ref_cache)
    actual = _paged_forward(
        paged_model,
        prompt,
        _offset_caches(kv_layers),
        decode_requests=[],
        prefill_requests=[([0], len(prompt), 0)],
    )
    assert mx.allclose(actual, expected, atol=tol, rtol=tol).item()

    seq_len = len(prompt)
    for tok in (29, 31, 41):
        expected = ref_model(mx.array([[tok]], dtype=mx.int32), cache=ref_cache)
        actual = _paged_forward(
            paged_model,
            [tok],
            _offset_caches(kv_layers, seq_len),
            decode_requests=[([0], seq_len)],
            prefill_requests=[],
        )
        assert mx.allclose(actual, expected, atol=tol, rtol=tol).item()
        seq_len += 1


def test_chunked_prefill_and_batched_decode_matches_mlx_lm() -> None:
    """Packed chunked prefill and batched decode keep each request's loops apart."""
    mx.random.seed(13)
    args = _make_args()
    ref_model = nanbeige.Model(args)
    mx.eval(ref_model.parameters())
    paged_model = _clone_model(ref_model, args, mx.float32)

    kv_layers = 2 * args.num_hidden_layers
    patch_sdpa_attention(paged_model, _kv_cache(args, kv_layers), _BLOCK_SIZE)

    seq_a = [2, 4, 6, 8, 10, 12, 14]
    seq_b = [1, 3, 5, 7, 9]
    ref_cache_a = make_prompt_cache(ref_model)
    ref_cache_b = make_prompt_cache(ref_model)
    exp_a = ref_model(mx.array([seq_a], dtype=mx.int32), cache=ref_cache_a)
    exp_b = ref_model(mx.array([seq_b], dtype=mx.int32), cache=ref_cache_b)

    # Chunk 1 of seq_a, then chunk 2 of seq_a packed with all of seq_b.
    _paged_forward(
        paged_model,
        seq_a[:4],
        _offset_caches(kv_layers),
        decode_requests=[],
        prefill_requests=[([0], 4, 0)],
    )
    packed = _paged_forward(
        paged_model,
        seq_a[4:] + seq_b,
        _offset_caches(kv_layers),
        decode_requests=[],
        prefill_requests=[([0], 3, 4), ([1], len(seq_b), 0)],
    )
    assert mx.allclose(packed[:, :3, :], exp_a[:, 4:, :], atol=2e-5).item()
    assert mx.allclose(packed[:, 3:, :], exp_b, atol=2e-5).item()

    exp_dec_a = ref_model(mx.array([[21]], dtype=mx.int32), cache=ref_cache_a)
    exp_dec_b = ref_model(mx.array([[22]], dtype=mx.int32), cache=ref_cache_b)
    decoded = _paged_forward(
        paged_model,
        [21, 22],
        _offset_caches(kv_layers, max(len(seq_a), len(seq_b))),
        decode_requests=[([0], len(seq_a)), ([1], len(seq_b))],
        prefill_requests=[],
    )
    assert mx.allclose(decoded[:, 0:1, :], exp_dec_a, atol=2e-5).item()
    assert mx.allclose(decoded[:, 1:2, :], exp_dec_b, atol=2e-5).item()


def test_patched_model_preserves_unpaged_forward_and_selective_logits() -> None:
    """Paths without a paged context still run every loop of the model."""
    mx.random.seed(19)
    args = _make_args()
    model = nanbeige.Model(args)
    tokens = mx.array([[1, 5, 9, 13]], dtype=mx.int32)
    expected = model(tokens)
    mx.eval(expected)

    kv_cache = _kv_cache(args, 2 * args.num_hidden_layers)
    patch_sdpa_attention(model, kv_cache, _BLOCK_SIZE)
    actual = model(tokens)
    mx.eval(actual)
    assert mx.array_equal(actual, expected).item()

    indices = mx.array([1, 3], dtype=mx.int32)
    adapter = DefaultModelAdapter()
    selective = adapter.target_forward(model, tokens, logits_indices=indices).logits
    mx.eval(selective)
    assert mx.allclose(selective, expected[0, indices, :], atol=1e-6).item()


def test_looped_patch_requires_a_kv_layer_per_loop() -> None:
    """Every loop of every layer needs its own KV layer; fail before serving."""
    args = _make_args()
    model = nanbeige.Model(args)
    with pytest.raises(ValueError, match="needs 6 KV cache layers"):
        patch_sdpa_attention(
            model, _kv_cache(args, args.num_hidden_layers), _BLOCK_SIZE
        )


def test_lifecycle_sizes_one_kv_layer_per_loop() -> None:
    args = _make_args()
    runner = SimpleNamespace(
        model_args=asdict(args),
        model=nanbeige.Model(args),
        is_mla=False,
        is_hybrid=False,
        pp=None,
    )
    ModelLifecycle(runner, DefaultModelAdapter()).resolve_model_dims()
    assert runner.num_layers == 6
    assert runner.num_kv_cache_layers == 6
