# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from collections.abc import Generator
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import pytest
import torch
from mlx_lm.models.base import scaled_dot_product_attention

import vllm_metal.attention.runtime.mla as mla_runtime
from tests.stub_runner import (
    initialize_glm5_next_runtime,
    initialize_hybrid_runtime,
    initialize_mla_runtime,
    make_bailing_hybrid_plan,
    make_glm5_next_stub_runner,
)
from vllm_metal.attention import context as pac
from vllm_metal.attention.caches.mla_cache import MLAPagedLatentCache
from vllm_metal.attention.impls.dsa import DSAPagedAttentionWrapper, query_parts
from vllm_metal.attention.impls.mla import MLAPagedAttentionWrapper
from vllm_metal.attention.runtime.hybrid import HybridPagedAttentionRuntime
from vllm_metal.attention.runtime.mla import MLAPagedAttentionRuntime
from vllm_metal.attention.runtime.protocol import PagedAttentionRuntime

# Fixture dimensions matching GLM/DeepSeek-V2 defaults
_KV_LORA_RANK = 512
_QK_ROPE_HEAD_DIM = 64
_LATENT_DIM = _KV_LORA_RANK + _QK_ROPE_HEAD_DIM


class TestMLAPagedLatentCache:
    def test_per_layer_array_shape(self) -> None:
        cache = MLAPagedLatentCache(
            num_layers=3,
            latent_dim=288,
            num_blocks=8,
            block_size=16,
            dtype=mx.float16,
        )

        assert len(cache.latent_caches) == 3
        for arr in cache.latent_caches:
            assert arr.shape == (8, 16, 288)  # (num_blocks, block_size, latent_dim)
            assert arr.dtype == mx.float16

    def test_bfloat16_dtype_accepted(self) -> None:
        cache = MLAPagedLatentCache(
            num_layers=2,
            latent_dim=192,
            num_blocks=4,
            block_size=8,
            dtype=mx.bfloat16,
        )

        assert all(arr.dtype == mx.bfloat16 for arr in cache.latent_caches)

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported dtype"):
            MLAPagedLatentCache(
                num_layers=2,
                latent_dim=_LATENT_DIM,
                num_blocks=5,
                block_size=16,
                dtype=mx.int32,
            )


class TestMLAPagedAttentionRuntime:
    def _make_backend(self) -> MLAPagedAttentionRuntime:
        return MLAPagedAttentionRuntime(
            num_layers=4,
            latent_dim=_LATENT_DIM,
            block_size=16,
            dtype=mx.float16,
        )

    def test_implements_paged_attention_runtime_protocol(self) -> None:
        backend = self._make_backend()

        assert isinstance(backend, PagedAttentionRuntime)

    def test_num_blocks_raises_before_initialize(self) -> None:
        backend = self._make_backend()

        with pytest.raises(RuntimeError, match="called before initialize"):
            backend.num_blocks()

    def test_warm_up_raises_before_initialize(self) -> None:
        backend = self._make_backend()

        with pytest.raises(RuntimeError, match="called before initialize"):
            backend.warm_up()

    def test_patch_model_raises_before_initialize(self) -> None:
        backend = self._make_backend()

        with pytest.raises(RuntimeError, match="called before initialize"):
            backend.patch_model(object())

    def test_num_blocks_after_initialize(self) -> None:
        backend = self._make_backend()
        backend.initialize(50)

        assert backend.num_blocks() == 50

    def test_warm_up_after_initialize_does_not_raise(self) -> None:
        backend = self._make_backend()
        backend.initialize(10)

        backend.warm_up()

    def test_initialize_allocates_cache_with_correct_shape(self) -> None:
        backend = self._make_backend()

        backend.initialize(20)

        assert backend._cache is not None
        assert backend._cache.num_blocks == 20
        assert backend._cache.latent_dim == _LATENT_DIM
        assert backend._cache.num_layers == 4


class _FakeAttn(nn.Module):
    pass


class _FakeLayer:
    def __init__(self) -> None:
        self.self_attn = _FakeAttn()


class _FakeModel:
    """Minimal stand-in for a model with .model.layers."""

    def __init__(self, num_layers: int) -> None:
        self.model = SimpleNamespace(layers=[_FakeLayer() for _ in range(num_layers)])


class TestPatchModelAttentionMla:
    def _make_backend(self, num_layers: int) -> MLAPagedAttentionRuntime:
        backend = MLAPagedAttentionRuntime(
            num_layers=num_layers,
            latent_dim=_LATENT_DIM,
            block_size=16,
            dtype=mx.float16,
        )
        backend.initialize(5)
        return backend

    def test_replaces_all_attention_layers(self) -> None:
        model = _FakeModel(num_layers=3)

        n = self._make_backend(num_layers=3).patch_model(model)

        assert n == 3
        for layer in model.model.layers:
            assert isinstance(layer.self_attn, MLAPagedAttentionWrapper)

    def test_wrapped_layer_has_correct_index(self) -> None:
        model = _FakeModel(num_layers=2)

        self._make_backend(num_layers=2).patch_model(model)

        for idx, layer in enumerate(model.model.layers):
            assert layer.self_attn._mla_layer_idx == idx

    def test_already_patched_layers_update_cache_reference(self) -> None:
        model = _FakeModel(num_layers=1)
        backend_a = self._make_backend(num_layers=1)
        backend_b = self._make_backend(num_layers=1)
        backend_a.patch_model(model)

        n = backend_b.patch_model(model)

        assert n == 1
        assert model.model.layers[0].self_attn._mla_latent_cache is backend_b._cache

    def test_returns_correct_patch_count(self) -> None:
        for n_layers in (1, 4, 10):
            model = _FakeModel(num_layers=n_layers)

            count = self._make_backend(num_layers=n_layers).patch_model(model)

            assert count == n_layers


class TestMLAPagedAttentionWrapperFallback:
    def test_delegates_to_inner_when_no_paged_context(self) -> None:
        sentinel = object()
        inner = MagicMock(return_value=sentinel)
        latent_cache = MagicMock(spec=MLAPagedLatentCache)

        wrapper = MLAPagedAttentionWrapper(
            inner, layer_idx=0, latent_cache=latent_cache
        )

        x = mx.zeros((1, 3, 64))
        result = wrapper(x, mask=None, cache=None)

        inner.assert_called_once_with(x, mask=None, cache=None)
        assert result is sentinel

    def test_passes_mask_and_cache_to_inner(self) -> None:
        inner = MagicMock(return_value=mx.zeros((1, 2, 32)))
        latent_cache = MagicMock(spec=MLAPagedLatentCache)
        wrapper = MLAPagedAttentionWrapper(
            inner, layer_idx=1, latent_cache=latent_cache
        )
        x = mx.zeros((1, 2, 32))
        mask = object()
        cache = object()

        wrapper(x, mask=mask, cache=cache)

        inner.assert_called_once_with(x, mask=mask, cache=cache)


_HIDDEN = 32
_NUM_HEADS = 2
_NOPE_DIM = 8  # qk_nope_head_dim
_ROPE_DIM = 4  # qk_rope_head_dim
_KV_RANK = 16  # kv_lora_rank
_V_DIM = 8  # v_head_dim
_Q_LORA_RANK = 12  # q_lora_rank


class _MinimalMLAInner(nn.Module):
    """Minimal MLA attention stub with correct shapes for paged path tests."""

    def __init__(self) -> None:
        super().__init__()
        self.q_lora_rank = None
        self.num_heads = _NUM_HEADS
        self.q_head_dim = _NOPE_DIM + _ROPE_DIM
        self.qk_nope_head_dim = _NOPE_DIM
        self.qk_rope_head_dim = _ROPE_DIM
        self.kv_lora_rank = _KV_RANK
        self.scale = 1.0 / math.sqrt(_KV_RANK)

        self.q_proj = nn.Linear(_HIDDEN, _NUM_HEADS * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(_HIDDEN, _KV_RANK + _ROPE_DIM, bias=False)
        self.kv_a_layernorm = nn.LayerNorm(_KV_RANK)
        self.embed_q = nn.Linear(_NOPE_DIM, _KV_RANK, bias=False)
        self.unembed_out = nn.Linear(_KV_RANK, _V_DIM, bias=False)
        self.o_proj = nn.Linear(_NUM_HEADS * _V_DIM, _HIDDEN, bias=False)

    def rope(self, x: mx.array, offset: int = 0) -> mx.array:
        # Identity RoPE: preserves shape, sufficient for testing shape logic.
        return x


class _MiniCPM3StyleInner(nn.Module):
    """MLA stub shaped like MiniCPM3: softmax_scale and kv_b_proj only."""

    def __init__(self) -> None:
        super().__init__()
        self.q_lora_rank = _Q_LORA_RANK
        self.num_heads = _NUM_HEADS
        self.q_head_dim = _NOPE_DIM + _ROPE_DIM
        self.qk_nope_head_dim = _NOPE_DIM
        self.qk_rope_head_dim = _ROPE_DIM
        self.kv_lora_rank = _KV_RANK
        self.softmax_scale = 0.37

        self.q_a_proj = nn.Linear(_HIDDEN, _Q_LORA_RANK, bias=False)
        self.q_a_layernorm = nn.LayerNorm(_Q_LORA_RANK)
        self.q_b_proj = nn.Linear(
            _Q_LORA_RANK, _NUM_HEADS * self.q_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(_HIDDEN, _KV_RANK + _ROPE_DIM, bias=False)
        self.kv_a_layernorm = nn.LayerNorm(_KV_RANK)
        self.kv_b_proj = nn.Linear(
            _KV_RANK, _NUM_HEADS * (_NOPE_DIM + _V_DIM), bias=False
        )
        self.o_proj = nn.Linear(_NUM_HEADS * _V_DIM, _HIDDEN, bias=False)

    def rope(self, x: mx.array, offset: int = 0) -> mx.array:
        return x


def _minicpm3_dense_reference(
    inner: _MiniCPM3StyleInner,
    x: mx.array,
    *,
    cache_dtype: mx.Dtype,
) -> mx.array:
    _, seq_len, _ = x.shape

    q = inner.q_b_proj(inner.q_a_layernorm(inner.q_a_proj(x)))
    q = q.reshape(1, seq_len, inner.num_heads, inner.q_head_dim).transpose(0, 2, 1, 3)
    q_nope, q_pe = mx.split(q, [inner.qk_nope_head_dim], axis=-1)

    kv_out = inner.kv_a_proj_with_mqa(x)
    compressed_kv, k_pe = mx.split(kv_out, [inner.kv_lora_rank], axis=-1)
    kv_norm = inner.kv_a_layernorm(compressed_kv).astype(cache_dtype)
    k_pe = k_pe.reshape(1, seq_len, 1, inner.qk_rope_head_dim).transpose(0, 2, 1, 3)
    k_pe = k_pe.astype(cache_dtype)

    kv = inner.kv_b_proj(kv_norm)
    kv = kv.reshape(1, seq_len, inner.num_heads, -1).transpose(0, 2, 1, 3)
    k_nope, values = mx.split(kv, [inner.qk_nope_head_dim], axis=-1)
    k_pe = mx.broadcast_to(
        k_pe,
        (1, inner.num_heads, seq_len, inner.qk_rope_head_dim),
    )

    queries = mx.concatenate([q_nope, q_pe], axis=-1)
    keys = mx.concatenate([k_nope, k_pe], axis=-1)
    attn_mask = None
    if seq_len > 1:
        rows = mx.arange(seq_len).reshape(-1, 1)
        cols = mx.arange(seq_len).reshape(1, -1)
        valid = (cols <= rows).reshape(1, 1, seq_len, seq_len)
        fill = mx.array(mx.finfo(queries.dtype).min, queries.dtype)
        attn_mask = mx.where(valid, mx.array(0, queries.dtype), fill)

    out = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=inner.softmax_scale,
        mask=attn_mask,
    )
    out = out.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
    return inner.o_proj(out)


class TestMLAPagedAttentionWrapperPagedPath:
    """Exercises the paged attention computation path (PagedAttentionContext set)."""

    @pytest.fixture(autouse=True)
    def _clear_ctx(self) -> Generator[None, None, None]:
        pac.clear_context()
        yield
        pac.clear_context()

    def _make_cache(self) -> MLAPagedLatentCache:
        return MLAPagedLatentCache(
            num_layers=1,
            latent_dim=_KV_RANK + _ROPE_DIM,
            num_blocks=4,
            block_size=4,
            dtype=mx.float16,
        )

    def test_decode_output_shape(self) -> None:
        # 1 request, 3 cached tokens, 1 new decode token
        inner = _MinimalMLAInner()
        cache = self._make_cache()
        wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)

        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[3],
                block_tables=[[0]],
                context_lens=[4],
                cu_seqlens=[0, 1],
                offsets=[3],
            )
        )

        out = wrapper(
            mx.random.normal((1, 1, _HIDDEN)).astype(mx.float16), mask=None, cache=None
        )
        mx.eval(out)

        assert out.shape == (1, 1, _HIDDEN)

    def test_prefill_output_shape(self) -> None:
        # 1 request, 0 past tokens, 4 new prefill tokens
        inner = _MinimalMLAInner()
        cache = self._make_cache()
        wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)

        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[0, 1, 2, 3],
                block_tables=[[0]],
                context_lens=[4],
                cu_seqlens=[0, 4],
                offsets=[0],
            )
        )

        out = wrapper(
            mx.random.normal((1, 4, _HIDDEN)).astype(mx.float16), mask=None, cache=None
        )
        mx.eval(out)

        assert out.shape == (1, 4, _HIDDEN)

    def test_cache_written_at_correct_slot(self) -> None:
        # Scatter-write: only the assigned slot is non-zero after the call
        inner = _MinimalMLAInner()
        cache = self._make_cache()
        wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)

        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[2],
                block_tables=[[0]],
                context_lens=[3],
                cu_seqlens=[0, 1],
                offsets=[2],
            )
        )

        wrapper(
            mx.random.normal((1, 1, _HIDDEN)).astype(mx.float16), mask=None, cache=None
        )

        # block 0, position 2 should now hold the new latent
        written = cache.latent_caches[0][0, 2, :]
        untouched = cache.latent_caches[0][0, 0, :]

        assert bool(mx.any(written != 0))
        assert not bool(mx.any(untouched != 0))

    def test_two_decode_requests_combined_output_shape(self) -> None:
        # Two decode requests in one batch — outputs must be concatenated along seq axis.
        inner = _MinimalMLAInner()
        cache = self._make_cache()
        wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)

        # Request A: 2 past tokens, decode token at slot 2 in block 0
        # Request B: 1 past token,  decode token at slot 5 in block 1
        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[2, 5],
                block_tables=[[0], [1]],
                context_lens=[3, 2],
                cu_seqlens=[0, 1, 2],
                offsets=[2, 1],
            )
        )

        x = mx.random.normal((1, 2, _HIDDEN)).astype(mx.float16)
        out = wrapper(x, mask=None, cache=None)
        mx.eval(out)

        assert out.shape == (1, 2, _HIDDEN)

    def test_causal_mask_token0_output_independent_of_later_tokens(self) -> None:
        # Token 0 in a prefill can only attend to itself (causal mask).
        # Changing tokens 1-3 must not change token 0's output.
        inner = _MinimalMLAInner()

        # Run 1: prefill with input_a
        cache_a = self._make_cache()
        wrapper_a = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache_a)
        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[0, 1, 2, 3],
                block_tables=[[0]],
                context_lens=[4],
                cu_seqlens=[0, 4],
                offsets=[0],
            )
        )
        mx.random.seed(0)
        token0 = mx.random.normal((1, 1, _HIDDEN)).astype(mx.float16)
        other = mx.random.normal((1, 3, _HIDDEN)).astype(mx.float16)
        input_a = mx.concatenate([token0, other], axis=1)
        out_a = wrapper_a(input_a, mask=None, cache=None)
        mx.eval(out_a)

        pac.clear_context()

        # Run 2: same token 0, completely different tokens 1-3
        cache_b = self._make_cache()
        wrapper_b = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache_b)
        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[0, 1, 2, 3],
                block_tables=[[0]],
                context_lens=[4],
                cu_seqlens=[0, 4],
                offsets=[0],
            )
        )
        mx.random.seed(99)
        different_other = mx.random.normal((1, 3, _HIDDEN)).astype(mx.float16)
        input_b = mx.concatenate([token0, different_other], axis=1)
        out_b = wrapper_b(input_b, mask=None, cache=None)
        mx.eval(out_b)

        # Token 0 output must be identical — it attends only to position 0
        assert bool(mx.all(out_a[0, 0, :] == out_b[0, 0, :]))

    def test_minicpm3_style_prefill_matches_dense_reference(self) -> None:
        inner = _MiniCPM3StyleInner()
        cache = self._make_cache()
        wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)

        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[0, 1, 2, 3],
                block_tables=[[0]],
                context_lens=[4],
                cu_seqlens=[0, 4],
                offsets=[0],
            )
        )

        mx.random.seed(7)
        x = mx.random.normal((1, 4, _HIDDEN)).astype(mx.float16)
        out = wrapper(x, mask=None, cache=None)
        expected = _minicpm3_dense_reference(inner, x, cache_dtype=cache.dtype)
        mx.eval(out, expected)

        assert bool(mx.allclose(out, expected, rtol=1e-3, atol=1e-3))

    def test_minicpm3_style_decode_matches_dense_reference(self) -> None:
        inner = _MiniCPM3StyleInner()
        cache = self._make_cache()
        wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)

        mx.random.seed(11)
        past = mx.random.normal((1, 3, _HIDDEN)).astype(mx.float16)
        new = mx.random.normal((1, 1, _HIDDEN)).astype(mx.float16)

        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[0, 1, 2],
                block_tables=[[0]],
                context_lens=[3],
                cu_seqlens=[0, 3],
                offsets=[0],
            )
        )
        wrapper(past, mask=None, cache=None)
        pac.clear_context()

        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[3],
                block_tables=[[0]],
                context_lens=[4],
                cu_seqlens=[0, 1],
                offsets=[3],
            )
        )
        out = wrapper(new, mask=None, cache=None)
        dense = _minicpm3_dense_reference(
            inner,
            mx.concatenate([past, new], axis=1),
            cache_dtype=cache.dtype,
        )
        expected = dense[:, -1:, :]
        mx.eval(out, expected)

        assert bool(mx.allclose(out, expected, rtol=1e-3, atol=1e-3))

    def test_bailing_shared_padded_cache_matches_mlx_lm(self) -> None:
        from mlx_lm.models.bailing_moe_v3 import BailingMLA, ModelArgs
        from mlx_lm.models.base import create_causal_mask
        from mlx_lm.models.cache import KVCache

        mx.random.seed(29)
        args = ModelArgs(
            hidden_size=16,
            num_attention_heads=2,
            head_dim=4,
            q_lora_rank=8,
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            max_position_embeddings=32,
        )
        inner = BailingMLA(args)
        reference_cache = KVCache()
        runtime = HybridPagedAttentionRuntime(
            hybrid_plan=make_bailing_hybrid_plan(
                2, state_dtypes=(torch.float32, torch.float32)
            ),
            dtype=mx.float32,
        )
        initialize_hybrid_runtime(
            runtime,
            3,
            block_size=4,
            head_dim=args.kv_lora_rank + args.qk_rope_head_dim,
            mla=True,
        )
        runtime.zero_blocks([0, 1, 2])
        mx.eval(*runtime.storage.buffers)
        paged_cache = runtime.kv_cache
        assert not paged_cache.has_dense_pages
        wrapper = MLAPagedAttentionWrapper(inner, 0, paged_cache)

        def run_paged(tokens: mx.array, *, offset: int, context_len: int) -> mx.array:
            num_tokens = tokens.shape[1]
            pac.set_context(
                pac.PagedAttentionContext(
                    slot_mapping=list(range(offset, offset + num_tokens)),
                    block_tables=[[0, 1]],
                    context_lens=[context_len],
                    offsets=[offset],
                    cu_seqlens=[0, num_tokens],
                )
            )
            try:
                return wrapper(tokens)
            finally:
                pac.clear_context()

        prefill = mx.random.normal((1, 5, args.hidden_size)).astype(mx.float32)
        expected_prefill = inner(
            prefill,
            mask=create_causal_mask(prefill.shape[1]),
            cache=reference_cache,
        )
        actual_prefill = run_paged(prefill, offset=0, context_len=5)

        decode = mx.random.normal((1, 1, args.hidden_size)).astype(mx.float32)
        expected_decode = inner(decode, cache=reference_cache)
        actual_decode = run_paged(decode, offset=5, context_len=6)
        mx.eval(actual_prefill, expected_prefill, actual_decode, expected_decode)

        assert bool(mx.allclose(actual_prefill, expected_prefill, rtol=1e-5, atol=1e-5))
        assert bool(mx.allclose(actual_decode, expected_decode, rtol=1e-5, atol=1e-5))
        rebound = MLAPagedLatentCache.from_upstream(
            runtime.storage, ["layers.1.self_attn"]
        )
        mx.eval(rebound.latent_caches[0])
        assert bool(
            mx.array_equal(rebound.latent_caches[0], paged_cache.latent_caches[0])
        )


# Single-pass kernel dimensions (matches mla.metal instantiation).
_KERNEL_KV_RANK = 512
_KERNEL_ROPE_DIM = 64
_KERNEL_NOPE_DIM = 128
_KERNEL_V_DIM = 128


class _KernelDimsAbsorbedInner(nn.Module):
    """Absorbed-MLA inner stub shaped to match the single-pass kernel
    instantiation (kv_lora_rank=512, qk_rope_head_dim=64). Used by the
    routing admission tests so ``_can_use_kernel`` actually exercises
    the accept path."""

    def __init__(self) -> None:
        super().__init__()
        self.q_lora_rank = None
        self.num_heads = 8  # multiple of 2 so _pick_heads_per_tg returns 2
        self.q_head_dim = _KERNEL_NOPE_DIM + _KERNEL_ROPE_DIM
        self.qk_nope_head_dim = _KERNEL_NOPE_DIM
        self.qk_rope_head_dim = _KERNEL_ROPE_DIM
        self.kv_lora_rank = _KERNEL_KV_RANK
        self.scale = 1.0 / math.sqrt(_KERNEL_KV_RANK)
        # embed_q / unembed_out presence is what flags the inner as absorbed.
        self.embed_q = nn.Linear(_KERNEL_NOPE_DIM, _KERNEL_KV_RANK, bias=False)
        self.unembed_out = nn.Linear(_KERNEL_KV_RANK, _KERNEL_V_DIM, bias=False)


def _make_kernel_dims_wrapper(
    *, block_size: int = 16, dtype: mx.Dtype = mx.float16
) -> MLAPagedAttentionWrapper:
    cache = MLAPagedLatentCache(
        num_layers=1,
        latent_dim=_KERNEL_KV_RANK + _KERNEL_ROPE_DIM,
        num_blocks=4,
        block_size=block_size,
        dtype=dtype,
    )
    return MLAPagedAttentionWrapper(
        inner=_KernelDimsAbsorbedInner(), layer_idx=0, latent_cache=cache
    )


def _make_decode_ctx(num_seqs: int = 1) -> SimpleNamespace:
    """Single-token-per-seq decode context, just enough for
    ``_can_use_kernel`` to inspect."""
    return SimpleNamespace(
        context_lens=[16] * num_seqs,
        cu_seqlens=list(range(num_seqs + 1)),
        block_tables=[[0]] * num_seqs,
    )


class TestDecodePathLog:
    """At warm-up the MLA runtime says whether decode takes the single-pass
    kernel, and why not when it doesn't."""

    def _warm_up(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        kernel_env: bool,
        kv_lora_rank: int = _KERNEL_KV_RANK,
        patches: int = 1,
    ) -> list[str]:
        monkeypatch.setattr("vllm_metal.envs.VLLM_METAL_MLA_KERNEL", kernel_env)
        inners = [_KernelDimsAbsorbedInner() for _ in range(2)]
        for inner in inners:
            inner.kv_lora_rank = kv_lora_rank
        layers = [SimpleNamespace(self_attn=inner) for inner in inners]
        model = SimpleNamespace(model=SimpleNamespace(layers=layers))
        backend = MLAPagedAttentionRuntime(
            num_layers=2,
            latent_dim=_KERNEL_KV_RANK + _KERNEL_ROPE_DIM,
            block_size=16,
            dtype=mx.float16,
        )
        backend.initialize(4)
        for _ in range(patches):
            backend.patch_model(model)
        info = MagicMock()
        monkeypatch.setattr(mla_runtime.logger, "info", info)
        backend.warm_up()
        return [call.args[0] % call.args[1:] for call in info.call_args_list]

    def test_logs_the_kernel(self, monkeypatch) -> None:
        lines = self._warm_up(monkeypatch, kernel_env=True)

        assert "Metal: MLA decode-only batches take the single-pass Metal kernel" in (
            lines
        )

    def test_logs_the_env_switch_when_off(self, monkeypatch) -> None:
        lines = self._warm_up(monkeypatch, kernel_env=False)

        assert (
            "Metal: MLA decode-only batches take the MLX SDPA path on 2 of 2 layers "
            "(VLLM_METAL_MLA_KERNEL is off)"
        ) in lines

    def test_a_repatch_counts_each_layer_once(self, monkeypatch) -> None:
        lines = self._warm_up(monkeypatch, kernel_env=False, patches=2)

        assert (
            "Metal: MLA decode-only batches take the MLX SDPA path on 2 of 2 layers "
            "(VLLM_METAL_MLA_KERNEL is off)"
        ) in lines

    def test_logs_the_shape_the_kernel_lacks(self, monkeypatch) -> None:
        lines = self._warm_up(monkeypatch, kernel_env=True, kv_lora_rank=256)

        assert (
            "Metal: MLA decode-only batches take the MLX SDPA path on 2 of 2 layers "
            "(kv_lora_rank 256, the kernel takes 512)"
        ) in lines


class TestSinglePassRouting:
    """Focused tests for the env-gated single-pass kernel fast path.
    Each test exercises one admission rule of ``_can_use_kernel`` —
    no exhaustive cell sweeps. Per the alignment with reviewers on
    ``Ship one kernel, prove it wins'', the kernel routing is a
    single boolean gate, not a multi-variant priority chain."""

    def test_env_off_rejects(self, monkeypatch) -> None:
        monkeypatch.setattr("vllm_metal.envs.VLLM_METAL_MLA_KERNEL", False)
        wrapper = _make_kernel_dims_wrapper()
        assert (
            wrapper._can_use_kernel(
                wrapper._inner, wrapper._mla_latent_cache, _make_decode_ctx()
            )
            is False
        )

    def test_env_on_accepts(self, monkeypatch) -> None:
        monkeypatch.setattr("vllm_metal.envs.VLLM_METAL_MLA_KERNEL", True)
        wrapper = _make_kernel_dims_wrapper()
        assert (
            wrapper._can_use_kernel(
                wrapper._inner, wrapper._mla_latent_cache, _make_decode_ctx()
            )
            is True
        )

    @pytest.mark.parametrize("padded", [False, True])
    def test_shared_cache_requires_dense_pages(self, monkeypatch, padded) -> None:
        monkeypatch.setattr("vllm_metal.envs.VLLM_METAL_MLA_KERNEL", True)
        runtime = HybridPagedAttentionRuntime(
            hybrid_plan=make_bailing_hybrid_plan(2, head_dim=128 if padded else 4),
            dtype=mx.float16,
        )
        initialize_hybrid_runtime(
            runtime, 2, block_size=16, head_dim=_LATENT_DIM, mla=True
        )
        cache = runtime.kv_cache
        wrapper = MLAPagedAttentionWrapper(_KernelDimsAbsorbedInner(), 0, cache)

        assert (
            wrapper._can_use_kernel(wrapper._inner, cache, _make_decode_ctx())
            is not padded
        )

    def test_non_absorbed_rejects(self, monkeypatch) -> None:
        """Inner without embed_q / unembed_out (kv_b_proj-style models
        like MiniCPM3) is not routed through the absorbed-MLA kernel."""
        monkeypatch.setattr("vllm_metal.envs.VLLM_METAL_MLA_KERNEL", True)
        cache = MLAPagedLatentCache(
            num_layers=1,
            latent_dim=_KERNEL_KV_RANK + _KERNEL_ROPE_DIM,
            num_blocks=4,
            block_size=16,
            dtype=mx.float16,
        )
        wrapper = MLAPagedAttentionWrapper(
            inner=_MiniCPM3StyleInner(), layer_idx=0, latent_cache=cache
        )
        assert (
            wrapper._can_use_kernel(
                wrapper._inner, wrapper._mla_latent_cache, _make_decode_ctx()
            )
            is False
        )

    def test_wrong_kv_lora_rank_rejects(self, monkeypatch) -> None:
        monkeypatch.setattr("vllm_metal.envs.VLLM_METAL_MLA_KERNEL", True)
        wrapper = _make_kernel_dims_wrapper()
        wrapper._inner.kv_lora_rank = 256  # not 512
        assert (
            wrapper._can_use_kernel(
                wrapper._inner, wrapper._mla_latent_cache, _make_decode_ctx()
            )
            is False
        )

    def test_wrong_qk_rope_head_dim_rejects(self, monkeypatch) -> None:
        monkeypatch.setattr("vllm_metal.envs.VLLM_METAL_MLA_KERNEL", True)
        wrapper = _make_kernel_dims_wrapper()
        wrapper._inner.qk_rope_head_dim = 32  # not 64
        assert (
            wrapper._can_use_kernel(
                wrapper._inner, wrapper._mla_latent_cache, _make_decode_ctx()
            )
            is False
        )

    def test_unsupported_block_size_rejects(self, monkeypatch) -> None:
        monkeypatch.setattr("vllm_metal.envs.VLLM_METAL_MLA_KERNEL", True)
        wrapper = _make_kernel_dims_wrapper(block_size=64)  # not in {16, 32}
        assert (
            wrapper._can_use_kernel(
                wrapper._inner, wrapper._mla_latent_cache, _make_decode_ctx()
            )
            is False
        )

    def test_unsupported_dtype_rejects(self, monkeypatch) -> None:
        monkeypatch.setattr("vllm_metal.envs.VLLM_METAL_MLA_KERNEL", True)
        wrapper = _make_kernel_dims_wrapper(dtype=mx.float32)
        assert (
            wrapper._can_use_kernel(
                wrapper._inner, wrapper._mla_latent_cache, _make_decode_ctx()
            )
            is False
        )

    def test_multi_token_query_rejects(self, monkeypatch) -> None:
        """Decode-only kernel — any request with >1 query token must
        fall back to MLX (varlen prefill is a follow-up)."""
        monkeypatch.setattr("vllm_metal.envs.VLLM_METAL_MLA_KERNEL", True)
        wrapper = _make_kernel_dims_wrapper()
        prefill_ctx = SimpleNamespace(
            context_lens=[16],
            cu_seqlens=[0, 4],  # 4 query tokens for seq 0
            block_tables=[[0]],
        )
        assert (
            wrapper._can_use_kernel(
                wrapper._inner, wrapper._mla_latent_cache, prefill_ctx
            )
            is False
        )

    @pytest.mark.parametrize(
        "num_heads,batch_size,expected_g",
        [
            (1, 1, 1),  # odd num_heads → fall back to G=1
            (16, 1, 1),  # B=1 small H → G=1 (better single-TG utilization)
            (16, 8, 2),  # B*H ≳ 30 → G=2 wins
            (32, 1, 2),  # B=1 but H≥32 → G=2 saturates GPU
        ],
    )
    def test_pick_heads_per_tg(
        self, num_heads: int, batch_size: int, expected_g: int
    ) -> None:
        assert (
            MLAPagedAttentionWrapper._pick_heads_per_tg(num_heads, batch_size)
            == expected_g
        )


class TestSharedStorageRuntime:
    """Standalone MLA binds latent views of vLLM's shared KV allocation."""

    def _make_backend(self) -> MLAPagedAttentionRuntime:
        return initialize_mla_runtime(6, num_layers=2)

    def test_binds_dense_views_the_kernel_admits(self, monkeypatch) -> None:
        monkeypatch.setattr("vllm_metal.envs.VLLM_METAL_MLA_KERNEL", True)
        backend = self._make_backend()
        cache = backend._cache

        assert backend.num_blocks() == 6
        assert backend.kv_scheduler_group_indices() == (0,)
        assert backend.kv_group_block_sizes() == (16,)
        assert cache.latent_caches[1].shape == (6, 16, _LATENT_DIM)
        wrapper = MLAPagedAttentionWrapper(_KernelDimsAbsorbedInner(), 0, cache)
        assert wrapper._can_use_kernel(wrapper._inner, cache, _make_decode_ctx())

    def test_copy_blocks_reach_the_shared_backing(self) -> None:
        backend = self._make_backend()
        cache = backend._cache
        rows = mx.random.normal((16, _LATENT_DIM)).astype(mx.float16)
        for layer in range(2):
            cache.write_slots(layer, mx.arange(16, 32), rows + layer)  # block 1

        backend.copy_blocks([(1, 4)])
        outputs: list[mx.array] = []
        backend.extend_forward_eval_outputs(outputs)
        mx.eval(*outputs)

        names = ["layers.0.self_attn", "layers.1.self_attn"]
        rebound = MLAPagedLatentCache.from_upstream(backend.storage, names)
        for layer in range(2):
            assert bool(mx.array_equal(rebound.latent_caches[layer][4], rows + layer))


class TestMLAPerForwardMetadata:
    """The typed ``ctx.mla_metadata`` memo converts each per-forward input
    once; every MLA layer in the pass must reuse the same mx arrays."""

    @pytest.fixture(autouse=True)
    def _clear_ctx(self) -> Generator[None, None, None]:
        pac.clear_context()
        yield
        pac.clear_context()

    def test_two_layers_share_one_sdpa_conversion(self) -> None:
        inner = _MinimalMLAInner()
        cache = MLAPagedLatentCache(
            num_layers=2,
            latent_dim=_KV_RANK + _ROPE_DIM,
            num_blocks=4,
            block_size=4,
            dtype=mx.float16,
        )
        wrapper0 = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)
        wrapper1 = MLAPagedAttentionWrapper(inner, layer_idx=1, latent_cache=cache)

        # Request A: 2 past tokens, decode token at slot 2 in block 0
        # Request B: 1 past token,  decode token at slot 5 in block 1
        ctx = pac.PagedAttentionContext(
            slot_mapping=[2, 5],
            block_tables=[[0], [1]],
            context_lens=[3, 2],
            cu_seqlens=[0, 1, 2],
            offsets=[2, 1],
        )
        pac.set_context(ctx)

        x = mx.random.normal((1, 2, _HIDDEN)).astype(mx.float16)
        mx.eval(wrapper0(x, mask=None, cache=None))

        meta = ctx.mla_metadata
        assert meta is not None
        slot = meta.slot_mapping
        rows = meta.block_table_rows

        mx.eval(wrapper1(x, mask=None, cache=None))

        assert ctx.mla_metadata is meta
        assert meta.slot_mapping is slot
        assert meta.block_table_rows is rows
        assert meta.kernel is None

        assert slot.dtype == mx.int64
        assert slot.tolist() == [2, 5]
        assert rows is not None
        assert [r.tolist() for r in rows] == [[0], [1]]
        assert all(r.dtype == mx.int32 for r in rows)

        # Both layers scatter-wrote slots 2 (block 0 pos 2) and 5 (block 1 pos 1)
        for layer_idx in (0, 1):
            layer_cache = cache.latent_caches[layer_idx]
            assert bool(mx.any(layer_cache[0, 2, :] != 0))
            assert bool(mx.any(layer_cache[1, 1, :] != 0))

    def test_two_layers_share_one_kernel_conversion(self, monkeypatch) -> None:
        calls: list[dict] = []

        def fake_kernel(**kwargs):
            calls.append(kwargs)
            return mx.zeros(kwargs["q_nope"].shape, dtype=kwargs["q_nope"].dtype)

        monkeypatch.setattr("vllm_metal.metal.metal_mla_paged_attention", fake_kernel)

        inner = _KernelDimsAbsorbedInner()
        cache = MLAPagedLatentCache(
            num_layers=2,
            latent_dim=_KERNEL_KV_RANK + _KERNEL_ROPE_DIM,
            num_blocks=4,
            block_size=16,
            dtype=mx.float16,
        )
        wrapper0 = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)
        wrapper1 = MLAPagedAttentionWrapper(inner, layer_idx=1, latent_cache=cache)

        # Unequal block-table lengths force zero-padding in the kernel input.
        ctx = pac.PagedAttentionContext(
            slot_mapping=[0, 32],
            block_tables=[[0], [1, 2]],
            context_lens=[5, 20],
            cu_seqlens=[0, 1, 2],
            offsets=[4, 19],
        )
        pac.set_context(ctx)

        q_nope = mx.random.normal((1, 8, 2, _KERNEL_NOPE_DIM)).astype(mx.float16)
        q_pe = mx.random.normal((1, 8, 2, _KERNEL_ROPE_DIM)).astype(mx.float16)

        mx.eval(
            wrapper0._kernel_fast_path_single_pass(
                inner, cache, 0, q_nope, q_pe, ctx, seq_len=2
            )
        )
        mx.eval(
            wrapper1._kernel_fast_path_single_pass(
                inner, cache, 1, q_nope, q_pe, ctx, seq_len=2
            )
        )

        assert len(calls) == 2
        first, second = calls
        for key in ("block_tables", "context_lens", "cu_seqlens_q"):
            assert second[key] is first[key]

        block_tables = first["block_tables"]
        assert block_tables.dtype == mx.int32
        assert block_tables.tolist() == [[0, 0], [1, 2]]
        assert first["context_lens"].dtype == mx.uint32
        assert first["context_lens"].tolist() == [5, 20]
        assert first["cu_seqlens_q"].dtype == mx.int32
        assert first["cu_seqlens_q"].tolist() == [0, 1, 2]

    def test_fresh_context_rebuilds(self) -> None:
        inner = _MinimalMLAInner()
        cache = MLAPagedLatentCache(
            num_layers=1,
            latent_dim=_KV_RANK + _ROPE_DIM,
            num_blocks=4,
            block_size=4,
            dtype=mx.float16,
        )
        wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)
        x = mx.random.normal((1, 1, _HIDDEN)).astype(mx.float16)

        ctx_a = pac.PagedAttentionContext(
            slot_mapping=[2],
            block_tables=[[0]],
            context_lens=[3],
            cu_seqlens=[0, 1],
            offsets=[2],
        )
        pac.set_context(ctx_a)
        mx.eval(wrapper(x, mask=None, cache=None))
        meta_a = ctx_a.mla_metadata
        assert meta_a is not None

        ctx_b = pac.PagedAttentionContext(
            slot_mapping=[7],
            block_tables=[[1]],
            context_lens=[2],
            cu_seqlens=[0, 1],
            offsets=[1],
        )
        pac.set_context(ctx_b)
        mx.eval(wrapper(x, mask=None, cache=None))
        meta_b = ctx_b.mla_metadata

        assert meta_b is not None
        assert meta_b is not meta_a
        assert meta_b.slot_mapping is not meta_a.slot_mapping
        assert meta_b.slot_mapping.tolist() == [7]


# ---------------------------------------------------------------------------
# GLM-5-Next sparse MLA on vLLM's paged latent and pooled-key caches.
# The layer cases run against mlx-vlm's own layer, indexer and caches.
# ---------------------------------------------------------------------------

_GLM_HIDDEN = 64
_GLM_LATENT = 32
_GLM_INDEX_DIM = 16


def _glm5_next_layer(*, kpool: int = 4, ties: bool = False) -> nn.Module:
    """A tiny bf16 GLM-5-Next sparse attention layer, in inference mode as
    mlx-vlm's loader returns it. Its widths are multiples of 32, so mlx-vlm's
    prefill takes its indexed kernel. ``ties`` zeroes the head weights, so
    every pool scores exactly 0."""
    from mlx_vlm.models.glm5_next.config import TextConfig
    from mlx_vlm.models.glm5_next.language import Glm5NextAttention

    mx.random.seed(0)
    config = TextConfig(
        hidden_size=_GLM_HIDDEN,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=32,
        kv_lora_rank=_GLM_LATENT,
        qk_nope_head_dim=32,
        qk_head_dim=32,
        v_head_dim=32,
        index_n_heads=2,
        index_head_dim=_GLM_INDEX_DIM,
        index_topk=2 * kpool,
        index_kpool=kpool,
        num_hidden_layers=4,
    )
    # In training mode, mlx-vlm sends decode steps through its prefill path.
    inner = Glm5NextAttention(config, layer_idx=3).eval()
    inner.set_dtype(mx.bfloat16)
    # Set after set_dtype: the checkpoint keeps the pool compression float32.
    inner.indexer.index_kpool_compress_ape = mx.random.normal((kpool, _GLM_INDEX_DIM))
    inner.indexer.index_kpool_compress_gate = mx.random.normal(
        (_GLM_INDEX_DIM, _GLM_HIDDEN)
    )
    if ties:
        weights = inner.indexer.weights_proj.weight
        inner.indexer.weights_proj.weight = mx.zeros_like(weights)
    return inner


def _glm5_next_paged(inner, *, num_blocks: int = 12, block_size: int = 32):
    """The paged layer over pages that all start with stale random rows, as
    reused blocks do, so every step must read only its own rows."""
    runner = make_glm5_next_stub_runner(
        block_size=block_size,
        kv_lora_rank=_GLM_LATENT,
        index_head_dim=_GLM_INDEX_DIM,
        index_kpool=inner.indexer.index_kpool,
    )
    runtime = initialize_glm5_next_runtime(runner, num_blocks)
    caches = runtime.indexer_caches
    for cache in (runtime.kv_cache, caches.pooled, caches.tail):
        rows = cache.num_blocks * cache.block_size
        cache.write_slots(
            0, mx.arange(rows), mx.random.normal((rows, cache.latent_dim))
        )
    return runtime, DSAPagedAttentionWrapper(inner, 0, runtime.kv_cache, caches)


class _GlmRequest:
    """One request's latent and pooled-key blocks and mlx-vlm's own caches for
    it."""

    def __init__(
        self, blocks: list[int], pooled_blocks: list[int], kpool: int = 4
    ) -> None:
        from mlx_vlm.models.cache import CacheList, KVCache, PoolingCache

        self.blocks = blocks
        self.pooled_blocks = pooled_blocks
        self.kpool = kpool
        self.offset = 0
        # Every step so far ran this request alone, so its rows are mlx-vlm's.
        self.alone = True
        self.reference = CacheList(KVCache(), KVCache(), PoolingCache(kpool), KVCache())


def _glm_step(inner, runtime, wrapper, step):
    """Run one packed step (decode rows first) through the paged layer, and
    each request alone through mlx-vlm's layer on its contiguous caches.

    Returns per request ``(paged output, mlx-vlm output, mlx-vlm top-k, x)``.
    """
    dtype = inner.qkv_a_proj.weight.dtype
    decode, prefill, xs = [], [], []
    for request, num_tokens in step:
        ids = [list(request.blocks), list(request.pooled_blocks)]
        if num_tokens == 1 and request.offset:
            assert not prefill, "decode rows come first"
            decode.append((ids, request.offset))
        else:
            prefill.append((ids, num_tokens, request.offset))
        xs.append(mx.random.normal((1, num_tokens, _GLM_HIDDEN)).astype(dtype))
    pac.prepare_grouped(decode, prefill, runtime.kv_group_block_sizes())
    try:
        actual, _ = wrapper(mx.concatenate(xs, axis=1))
    finally:
        pac.clear_context()
    results = []
    start = 0
    for (request, num_tokens), x in zip(step, xs, strict=True):
        expected, topk = inner(x, cache=request.reference)
        results.append((actual[:, start : start + num_tokens], expected, topk, x))
        start += num_tokens
        request.offset += num_tokens
        request.alone = request.alone and len(step) == 1
    return results


def _relative_error(actual: mx.array, expected: mx.array) -> float:
    actual, expected = actual.astype(mx.float32), expected.astype(mx.float32)
    return float(mx.linalg.norm(actual - expected) / mx.linalg.norm(expected))


def _assert_output_matches(actual, expected, num_tokens: int, alone: bool) -> None:
    """Decode rows of a request served alone must be bitwise mlx-vlm's: the
    same rows, selection and attention. Prefill rows, and the rows of a packed
    step, are held to bf16 tolerance."""
    if num_tokens == 1 and alone:
        assert bool(mx.array_equal(actual, expected))
    else:
        assert _relative_error(actual, expected) < 3e-2


def _assert_pages_match(runtime, request: _GlmRequest) -> None:
    """The paged pools and tail hold bitwise what mlx-vlm's PoolingCache holds."""
    caches = runtime.indexer_caches
    tail_keys, tail_gates, pooled = request.reference[2].state
    num_pools, remainder = divmod(request.offset, request.kpool)
    if num_pools:
        pages = caches.pooled.latent_caches[0][mx.array(request.pooled_blocks)]
        rows = pages.reshape(-1, pages.shape[-1])[:num_pools]
        assert bool(mx.array_equal(rows, pooled[0]))
    if remainder:
        block = request.blocks[(request.offset - 1) // runtime.kv_cache.block_size]
        tail = caches.tail.latent_caches[0][block, :remainder]
        assert bool(
            mx.array_equal(tail, mx.concatenate([tail_keys[0], tail_gates[0]], -1))
        )


class TestGlm5NextSparseAttention:
    @pytest.mark.parametrize(
        ("kpool", "block_size", "chunks"),
        [
            # A chunk ending exactly on a block, then decode into the next.
            (4, 32, (32, 1, 1, 3, 1)),
            # Chunks start and end mid-pool, reading and overwriting the tail.
            # One-token steps take mlx-vlm's decode path, the 5- and 9-token
            # chunks the indexed kernel.
            (4, 64, (1, 1, 11, 5, 9, 1, 1)),
            (16, 512, (35, 13, 1, 1)),
        ],
        ids=["block_boundary", "short_chunks_block_64", "kpool_16"],
    )
    def test_layer_matches_mlx_vlm_across_steps(
        self, kpool, block_size, chunks
    ) -> None:
        """One request through chunked prefill and decode past index_topk, on
        permuted, non-contiguous blocks: each output against mlx-vlm's layer,
        and the paged pools and tail against its PoolingCache, every step."""
        inner = _glm5_next_layer(kpool=kpool)
        runtime, wrapper = _glm5_next_paged(inner, block_size=block_size)
        request = _GlmRequest([5, 2, 7], [10], kpool=kpool)

        for num_tokens in chunks:
            ((actual, expected, _, _),) = _glm_step(
                inner, runtime, wrapper, [(request, num_tokens)]
            )
            _assert_output_matches(actual, expected, num_tokens, request.alone)
            _assert_pages_match(runtime, request)

    def test_packed_steps_match_each_request_alone(self) -> None:
        """Decode rows packed with chunks that start and end mid-pool (what two
        streams get at a 1024-token budget: 1023-token chunks), and prefills
        with different block tables. Pools and tails stay bitwise each
        request's own; outputs within bf16 tolerance."""
        inner = _glm5_next_layer()
        runtime, wrapper = _glm5_next_paged(inner)
        a, b, c = (
            _GlmRequest([3, 9], [10]),
            _GlmRequest([6, 4, 11], [7]),
            _GlmRequest([8], [5]),
        )
        steps = [
            [(a, 13)],
            [(a, 1), (b, 7)],
            [(a, 1), (b, 7)],
            [(a, 1), (b, 1), (c, 9)],
            [(b, 1), (a, 5), (c, 3)],
            [(a, 1), (b, 1), (c, 1)],
        ]

        for step in steps:
            results = _glm_step(inner, runtime, wrapper, step)
            for (request, num_tokens), (actual, expected, _, _) in zip(
                step, results, strict=True
            ):
                _assert_output_matches(actual, expected, num_tokens, request.alone)
                _assert_pages_match(runtime, request)

    def test_prefix_hit_starts_on_the_shared_blocks(self) -> None:
        """A prefix-cache hit: the second request starts on the first
        request's whole first latent block and, as vLLM's partial hits do, on
        a copy of its partly filled pooled-key block. It reads that block's
        latent rows and pools, and matches mlx-vlm fed the same prefix rows."""
        inner = _glm5_next_layer()
        runtime, wrapper = _glm5_next_paged(inner)
        first = _GlmRequest([4, 1], [10])
        ((_, _, _, prefix),) = _glm_step(inner, runtime, wrapper, [(first, 32)])
        # The first request goes on, into its own second block.
        _glm_step(inner, runtime, wrapper, [(first, 5)])

        second = _GlmRequest([4, 6], [7])
        runtime.copy_blocks([(10, 7)])
        inner(prefix, cache=second.reference)
        second.offset = 32
        for num_tokens in (7, 1, 1):
            ((actual, expected, _, _),) = _glm_step(
                inner, runtime, wrapper, [(second, num_tokens)]
            )
            _assert_output_matches(actual, expected, num_tokens, second.alone)
            _assert_pages_match(runtime, second)

    @pytest.mark.parametrize(
        ("cap_mib", "ties", "block_size", "history", "chunk"),
        [
            (512, False, 32, 21, 19),
            (0, True, 32, 21, 19),
            # At 2 MiB, 1400 rows over 350 pools split 748 + 652, and 600 rows
            # over 500 pools 524 + 76.
            (2, False, 64, 1400, 600),
        ],
        ids=["one_part", "row_parts_ties", "multi_row_parts"],
    )
    def test_prefill_matches_mlx_vlms_kernel(
        self, monkeypatch, cap_mib, ties, block_size, history, chunk
    ) -> None:
        """A bf16 prefill chunk over paged history, its indexer scored in one
        part or in query parts (a cap of 0 makes every row a part), is bitwise
        mlx-vlm's indexed kernel on the contiguous latent with the selection
        of mlx-vlm's single indexer call, and close to mlx-vlm's projected
        prefill."""
        # mlx-vlm's own kernel call, private in mlx-vlm 0.7.6, is the reference.
        from mlx_vlm.models.glm5_next.language import _sparse_prefill_attention
        from mlx_vlm.models.linear import linear

        monkeypatch.setattr(
            "vllm_metal.attention.impls.dsa._MAX_SCORES_BYTES", cap_mib * 2**20
        )
        inner = _glm5_next_layer(ties=ties)
        runtime, wrapper = _glm5_next_paged(inner, num_blocks=40, block_size=block_size)
        blocks = list(range(math.ceil((history + chunk) / block_size), 0, -1))
        num_pooled = math.ceil((history + chunk) / runtime.kv_group_block_sizes()[1])
        pooled = list(range(len(blocks) + 1, len(blocks) + 1 + num_pooled))
        request = _GlmRequest(blocks, pooled)
        _glm_step(inner, runtime, wrapper, [(request, history)])

        ((actual, projected, selected, x),) = _glm_step(
            inner, runtime, wrapper, [(request, chunk)]
        )

        q_a, _ = mx.split(linear(inner.qkv_a_proj, x), (inner.q_lora_rank,), axis=-1)
        q = (
            linear(inner.q_b_proj, inner.q_a_layernorm(q_a))
            .reshape(1, chunk, inner.num_heads, inner.q_head_dim)
            .transpose(0, 2, 1, 3)
        )
        latent = request.reference[0].state[0]
        out = _sparse_prefill_attention(
            inner.embed_q(q), latent, latent, selected, inner.scale
        )
        absorbed = linear(
            inner.o_proj,
            inner.unembed_out(out).transpose(0, 2, 1, 3).reshape(1, chunk, -1),
        )
        assert bool(mx.array_equal(actual, absorbed))
        assert _relative_error(actual, projected) < 3e-2
        _assert_pages_match(runtime, request)

    def test_query_parts_stay_within_the_cap(self) -> None:
        """At the default cap and 32 index heads, a 1024-row chunk after 17k
        tokens (4352 pools) splits into parts of 512 MiB // (4 * 32 * 4352)
        rows, the part the reserve covers."""
        assert query_parts(1024, 32, 4352) == [0, 963, 1024]

    def test_decode_build_submits_no_gpu_work(self, monkeypatch) -> None:
        """A later pipeline stage builds a decode-only forward before its
        receive wait, so building one must evaluate nothing."""
        inner = _glm5_next_layer()
        runtime, wrapper = _glm5_next_paged(inner)
        request = _GlmRequest([1, 2], [3])
        _glm_step(inner, runtime, wrapper, [(request, 21)])
        calls: list[str] = []
        for name in ("eval", "async_eval", "synchronize"):
            monkeypatch.setattr(mx, name, lambda *_, _n=name, **__: calls.append(_n))
        pac.prepare_grouped(
            [([request.blocks, request.pooled_blocks], 21)],
            [],
            runtime.kv_group_block_sizes(),
        )
        try:
            wrapper(mx.random.normal((1, 1, _GLM_HIDDEN)).astype(mx.bfloat16))
        finally:
            pac.clear_context()

        assert calls == []
