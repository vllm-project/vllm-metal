# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from collections.abc import Generator
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import pytest

import vllm_metal.paged_attention_common as pac
from vllm_metal.mlx_backend.mla_cache import MLAPagedLatentCache
from vllm_metal.paged_attention_backend.mla import (
    MLAPagedAttentionBackend,
    MLAPagedAttentionWrapper,
)
from vllm_metal.paged_attention_backend.protocol import PagedAttentionBackend

_np_ctx = pac.PagedAttentionContext.from_lists


# Fixture dimensions matching GLM/DeepSeek-V2 defaults
_KV_LORA_RANK = 512
_QK_ROPE_HEAD_DIM = 64
_LATENT_DIM = _KV_LORA_RANK + _QK_ROPE_HEAD_DIM


class TestMLAPagedLatentCache:
    def test_latent_dim_stored_correctly(self) -> None:
        cache = MLAPagedLatentCache(
            num_layers=4,
            latent_dim=_LATENT_DIM,
            num_blocks=10,
            block_size=16,
            dtype=mx.float16,
        )

        assert cache.latent_dim == _LATENT_DIM

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

        assert cache.dtype == mx.bfloat16

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported dtype"):
            MLAPagedLatentCache(
                num_layers=2,
                latent_dim=_LATENT_DIM,
                num_blocks=5,
                block_size=16,
                dtype=mx.int32,
            )


class TestMLAPagedAttentionBackend:
    def _make_backend(self) -> MLAPagedAttentionBackend:
        return MLAPagedAttentionBackend(
            num_layers=4,
            latent_dim=_LATENT_DIM,
            block_size=16,
            dtype=mx.float16,
        )

    def test_implements_paged_attention_backend_protocol(self) -> None:
        backend = self._make_backend()

        assert isinstance(backend, PagedAttentionBackend)

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
    def _make_backend(self, num_layers: int) -> MLAPagedAttentionBackend:
        backend = MLAPagedAttentionBackend(
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
            _np_ctx(
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
            _np_ctx(
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
            _np_ctx(
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
            _np_ctx(
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
            _np_ctx(
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
            _np_ctx(
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
