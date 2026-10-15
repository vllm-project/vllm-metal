# SPDX-License-Identifier: Apache-2.0
"""Unit tests for paged SDPA attention contracts.

Covers:
- ``pad_qkv_to_cache_head_dim`` / ``truncate_padded_output`` pure helpers.
- ``prepare_sdpa_qkv`` branches (K-eq-V fallback, v_norm, YOCO shared_kv).
- ``sdpa_forward`` propagation of per-layer metadata to the kernel.

The ``prepare_sdpa_qkv`` tests use minimal fake attention modules rather
than real mlx_lm Attention modules so they stay fast and deterministic.
Contiguous-prefill tests also exercise real cache writes and compare both
attention implementations against an independent causal reference.
"""

import logging
import os
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest
import torch
from mlx.utils import tree_flatten
from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)

import vllm_metal.attention.impls.sdpa as sdpa_mod
from vllm_metal.attention.attention_contracts import (
    AttentionContract,
    attention_contract_for,
)
from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.attention.context import (
    PagedAttentionContext,
    clear_context,
    get_context,
    prepare_grouped,
)
from vllm_metal.attention.impls.sdpa import (
    apply_g_proj_gate,
    pad_qkv_to_cache_head_dim,
    prepare_sdpa_qkv,
    sdpa_forward,
    truncate_padded_output,
)

# === Test fixtures (shared shapes) ===

_BATCH = 1
_SEQ_LEN = 4
_HIDDEN = 8
_N_HEADS = 2
_N_KV_HEADS = 2
_HEAD_DIM = 4  # small enough for fast unit tests
_CACHE_HEAD_DIM = 8  # Gemma4-style: cache wider than layer head_dim


@pytest.fixture(autouse=True)
def _clear_attention_context() -> Iterator[None]:
    yield
    clear_context()


@pytest.fixture(autouse=True)
def _force_paged_prefill_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests capture paged-kernel calls; the contiguous prefill path
    legitimately bypasses them, so pin it off."""
    monkeypatch.setenv("VLLM_METAL_SDPA_CONTIGUOUS_PREFILL", "0")


class _FakeLinear:
    """Minimal linear-like callable with a ``.weight`` attribute.

    Matches what ``prepare_sdpa_qkv`` needs from ``inner.{q,k,v}_proj``:
    a callable that projects ``x`` and exposes ``.weight.shape`` for
    head_dim resolution.
    """

    def __init__(self, weight: mx.array) -> None:
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.T


class _RaisingLinear:
    """Linear stub that fails if called — used to prove a branch is skipped."""

    def __init__(self, weight: mx.array, message: str) -> None:
        self.weight = weight
        self._message = message

    def __call__(self, x: mx.array) -> mx.array:
        raise AssertionError(self._message)


class _ExplodingPackedLinear:
    """Packed projection stub that must not be used on split fallback paths."""

    def __init__(self, message: str) -> None:
        self._message = message

    def __call__(self, x: mx.array) -> mx.array:
        raise AssertionError(self._message)


class _FakeQuantLinear:
    """Quantized-wrapper-style projection (e.g. GGUFLinear).

    Callable and exposes ``out_features``, but has NO dense ``.weight`` — the
    packed weight is hidden behind the quant tensor. head_dim resolution must
    read ``out_features``; reaching for ``.weight`` would ``AttributeError``
    (the GGUF serve crash this exercises).
    """

    def __init__(self, weight: mx.array) -> None:
        self._weight = weight  # private: never exposed as ``.weight``
        self.out_features = weight.shape[0]

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self._weight.T


def _make_ctx(seq_len: int) -> PagedAttentionContext:
    """Return a minimal paged context sufficient for apply_packed_rope."""
    return PagedAttentionContext(
        slot_mapping=list(range(seq_len)),
        block_tables=[[0]],
        context_lens=[seq_len],
        offsets=[],
        cu_seqlens=[0, seq_len],
    )


def _identity_rope(x: mx.array, offset: int = 0) -> mx.array:
    """Stand-in RoPE that leaves inputs unchanged (tests call-site plumbing,
    not the rotation math itself)."""
    return x


def _make_inner(
    *,
    with_v_proj: bool = True,
    with_v_norm: bool = False,
) -> SimpleNamespace:
    """Build a fake Attention module matching sdpa_forward's contract."""
    q_weight = mx.ones((_N_HEADS * _HEAD_DIM, _HIDDEN))
    k_weight = mx.ones((_N_KV_HEADS * _HEAD_DIM, _HIDDEN))
    v_weight = mx.ones((_N_KV_HEADS * _HEAD_DIM, _HIDDEN))

    inner = SimpleNamespace(
        n_heads=_N_HEADS,
        n_kv_heads=_N_KV_HEADS,
        scale=_HEAD_DIM**-0.5,
        q_proj=_FakeLinear(q_weight),
        k_proj=_FakeLinear(k_weight),
        rope=_identity_rope,
    )
    if with_v_proj:
        inner.v_proj = _FakeLinear(v_weight)
    if with_v_norm:
        # Track invocation so the test can assert the norm was actually
        # applied rather than silently skipped.
        inner._v_norm_calls = 0

        def v_norm(v: mx.array) -> mx.array:
            inner._v_norm_calls += 1
            return v * 2.0

        inner.v_norm = v_norm
    return inner


def _make_qkv_inner(
    *,
    n_heads: int = _N_HEADS,
    n_kv_heads: int = _N_KV_HEADS,
    head_dim: int = _HEAD_DIM,
) -> SimpleNamespace:
    """Build a Phi3/Phi4-like Attention module with packed qkv_proj."""
    q_weight = mx.ones((n_heads * head_dim, _HIDDEN))
    k_weight = mx.full((n_kv_heads * head_dim, _HIDDEN), 2.0)
    v_weight = mx.full((n_kv_heads * head_dim, _HIDDEN), 3.0)
    qkv_weight = mx.concatenate([q_weight, k_weight, v_weight], axis=0)

    return SimpleNamespace(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        scale=head_dim**-0.5,
        qkv_proj=_FakeLinear(qkv_weight),
        o_proj=lambda out: out,
        rope=_identity_rope,
    )


# === pad_qkv_to_cache_head_dim ===


class TestPadQKVToCacheHeadDim:
    """Tests for ``pad_qkv_to_cache_head_dim``."""

    def test_noop_when_head_dim_matches(self) -> None:
        # Arrange
        shape = (_BATCH, _N_HEADS, _SEQ_LEN, _HEAD_DIM)
        q = mx.ones(shape)
        k = mx.ones(shape)
        v = mx.ones(shape)

        # Act
        qp, kp, vp = pad_qkv_to_cache_head_dim(q, k, v, _HEAD_DIM, _HEAD_DIM)

        # Assert — no change, same objects returned
        assert qp is q
        assert kp is k
        assert vp is v

    def test_pads_last_axis_with_zeros(self) -> None:
        # Arrange — Gemma4 sliding layer: head_dim=4, cache_head_dim=8
        shape = (_BATCH, _N_HEADS, _SEQ_LEN, _HEAD_DIM)
        q = mx.ones(shape)
        k = mx.ones(shape)
        v = mx.ones(shape)

        # Act
        qp, kp, vp = pad_qkv_to_cache_head_dim(q, k, v, _HEAD_DIM, _CACHE_HEAD_DIM)

        # Assert — shape padded to cache_head_dim
        expected_shape = (_BATCH, _N_HEADS, _SEQ_LEN, _CACHE_HEAD_DIM)
        assert qp.shape == expected_shape
        assert kp.shape == expected_shape
        assert vp.shape == expected_shape
        # Leading values preserved, trailing positions zeroed
        mx.eval(qp, kp, vp)
        trailing = qp[..., _HEAD_DIM:]
        assert bool(mx.all(trailing == 0).item())

    def test_rejects_head_dim_larger_than_cache(self) -> None:
        # Arrange
        shape = (_BATCH, _N_HEADS, _SEQ_LEN, _CACHE_HEAD_DIM)
        q = mx.ones(shape)

        # Act / Assert
        with pytest.raises(ValueError, match="exceeds cache_head_dim"):
            pad_qkv_to_cache_head_dim(q, q, q, _CACHE_HEAD_DIM, _HEAD_DIM)

    def test_rejects_mismatched_qkv_last_dim(self) -> None:
        # Arrange — caller passes Q at one head_dim but K/V at another
        q = mx.ones((_BATCH, _N_HEADS, _SEQ_LEN, _HEAD_DIM))
        k = mx.ones((_BATCH, _N_HEADS, _SEQ_LEN, _CACHE_HEAD_DIM))
        v = mx.ones((_BATCH, _N_HEADS, _SEQ_LEN, _CACHE_HEAD_DIM))

        # Act / Assert
        with pytest.raises(ValueError, match="last-dim mismatch"):
            pad_qkv_to_cache_head_dim(q, k, v, _HEAD_DIM, _CACHE_HEAD_DIM)


# === truncate_padded_output ===


class TestTruncatePaddedOutput:
    """Tests for ``truncate_padded_output``."""

    def test_noop_when_not_padded(self) -> None:
        # Arrange — same head_dim, no padding to strip
        out = mx.ones((_SEQ_LEN, _N_HEADS, _CACHE_HEAD_DIM))

        # Act
        flat = truncate_padded_output(
            out,
            _BATCH,
            _SEQ_LEN,
            _N_HEADS,
            _CACHE_HEAD_DIM,
            _CACHE_HEAD_DIM,
        )

        # Assert
        assert flat.shape == (_BATCH, _SEQ_LEN, _N_HEADS * _CACHE_HEAD_DIM)

    def test_strips_trailing_padded_dims(self) -> None:
        # Arrange — kernel output at cache width, actual layer is narrower
        out = mx.ones((_SEQ_LEN, _N_HEADS, _CACHE_HEAD_DIM))

        # Act
        flat = truncate_padded_output(
            out,
            _BATCH,
            _SEQ_LEN,
            _N_HEADS,
            _CACHE_HEAD_DIM,
            _HEAD_DIM,
        )

        # Assert — trailing per-head dims dropped before flatten
        assert flat.shape == (_BATCH, _SEQ_LEN, _N_HEADS * _HEAD_DIM)


# === prepare_sdpa_qkv ===


class TestPrepareSDPAQKV:
    """Tests for ``prepare_sdpa_qkv`` Gemma4 branches."""

    def test_standard_path_projects_independent_kv(self) -> None:
        # Arrange — Qwen3/Llama-style with v_proj present
        inner = _make_inner(with_v_proj=True)
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))

        # Act
        queries, keys, values, gate, kv_for_sharing = prepare_sdpa_qkv(
            inner, x, ctx, _N_HEADS, _N_KV_HEADS, shared_kv=None
        )

        # Assert — canonical (B, H, L, D) shapes
        assert queries.shape == (_BATCH, _N_HEADS, _SEQ_LEN, _HEAD_DIM)
        assert keys.shape == (_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM)
        assert values.shape == (_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM)
        assert gate is None
        assert kv_for_sharing == (keys, values)

    def test_resolves_head_dim_from_out_features_for_weightless_projection(
        self,
    ) -> None:
        # A quantized wrapper (GGUFLinear) hides its packed weight and exposes
        # out_features instead of a dense .weight. With no self.head_dim attr
        # (the qwen3 path), head_dim must resolve from k_proj.out_features;
        # reaching for k_proj.weight here would AttributeError (the crash this
        # fixes). Mirrors test_standard_path_projects_independent_kv, which
        # already pins the dense .weight path to the same head_dim.
        inner = SimpleNamespace(
            n_heads=_N_HEADS,
            n_kv_heads=_N_KV_HEADS,
            scale=_HEAD_DIM**-0.5,
            q_proj=_FakeQuantLinear(mx.ones((_N_HEADS * _HEAD_DIM, _HIDDEN))),
            k_proj=_FakeQuantLinear(mx.ones((_N_KV_HEADS * _HEAD_DIM, _HIDDEN))),
            v_proj=_FakeQuantLinear(mx.ones((_N_KV_HEADS * _HEAD_DIM, _HIDDEN))),
            rope=_identity_rope,
        )
        assert not hasattr(inner.k_proj, "weight")  # the weightless contract
        assert not hasattr(inner, "head_dim")  # forces the k_proj derivation
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))

        queries, keys, values, _, _ = prepare_sdpa_qkv(
            inner, x, ctx, _N_HEADS, _N_KV_HEADS, shared_kv=None
        )

        # head_dim resolved from out_features (== _HEAD_DIM): canonical shapes.
        assert queries.shape == (_BATCH, _N_HEADS, _SEQ_LEN, _HEAD_DIM)
        assert keys.shape == (_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM)
        assert values.shape == (_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM)

    def test_k_eq_v_fallback_when_v_proj_missing(self) -> None:
        # Arrange — Gemma4 26B/31B style: no v_proj
        inner = _make_inner(with_v_proj=False)
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))

        # Act
        _, keys, values, _, _ = prepare_sdpa_qkv(
            inner, x, ctx, _N_HEADS, _N_KV_HEADS, shared_kv=None
        )

        # Assert — without v_proj we expect values to share K's projection.
        # After transpose they must be element-equal (same weights, same x).
        mx.eval(keys, values)
        assert bool(mx.all(keys == values).item())

    def test_k_eq_v_fallback_when_v_proj_is_none(self) -> None:
        # Arrange — DiffusionGemma full-attention layers keep v_proj = None
        inner = _make_inner(with_v_proj=False)
        inner.v_proj = None
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))

        # Act
        _, keys, values, _, _ = prepare_sdpa_qkv(
            inner, x, ctx, _N_HEADS, _N_KV_HEADS, shared_kv=None
        )

        # Assert — a None v_proj takes the K-eq-V path instead of calling None
        mx.eval(keys, values)
        assert bool(mx.all(keys == values).item())

    def test_qkv_proj_path_splits_phi_style_packed_projection(self) -> None:
        # Arrange — Phi3/Phi4-style attention uses a single qkv_proj linear.
        inner = _make_qkv_inner()
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))

        # Act
        queries, keys, values, gate, kv_for_sharing = prepare_sdpa_qkv(
            inner, x, ctx, _N_HEADS, _N_KV_HEADS, shared_kv=None
        )

        # Assert — packed projection splits into canonical (B, H, L, D).
        mx.eval(queries, keys, values)
        assert queries.shape == (_BATCH, _N_HEADS, _SEQ_LEN, _HEAD_DIM)
        assert keys.shape == (_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM)
        assert values.shape == (_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM)
        assert bool(mx.all(queries == 8.0).item())
        assert bool(mx.all(keys == 16.0).item())
        assert bool(mx.all(values == 24.0).item())
        assert gate is None
        assert kv_for_sharing == (keys, values)

    def test_qkv_proj_path_supports_phi_style_gqa_head_ratio(self) -> None:
        # Arrange — real Phi checkpoints use more query heads than KV heads.
        n_heads = 4
        n_kv_heads = 2
        head_dim = _HEAD_DIM
        inner = _make_qkv_inner(
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))

        # Act
        queries, keys, values, gate, _ = prepare_sdpa_qkv(
            inner, x, ctx, n_heads, n_kv_heads, shared_kv=None
        )

        # Assert — packed qkv must preserve the GQA head ratio after split.
        mx.eval(queries, keys, values)
        assert queries.shape == (_BATCH, n_heads, _SEQ_LEN, head_dim)
        assert keys.shape == (_BATCH, n_kv_heads, _SEQ_LEN, head_dim)
        assert values.shape == (_BATCH, n_kv_heads, _SEQ_LEN, head_dim)
        assert bool(mx.all(queries == 8.0).item())
        assert bool(mx.all(keys == 16.0).item())
        assert bool(mx.all(values == 24.0).item())
        assert gate is None

    def test_mixed_module_falls_back_to_split_projection_path(self) -> None:
        # Arrange — dispatch may accept mixed modules via the split contract.
        inner = _make_inner(with_v_proj=True)
        inner.qkv_proj = _ExplodingPackedLinear(
            "qkv_proj must not be used when packed metadata is incomplete"
        )
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))

        # Act
        queries, keys, values, gate, _ = prepare_sdpa_qkv(
            inner, x, ctx, _N_HEADS, _N_KV_HEADS, shared_kv=None
        )

        # Assert — split q_proj/k_proj/v_proj path still works.
        mx.eval(queries, keys, values)
        assert queries.shape == (_BATCH, _N_HEADS, _SEQ_LEN, _HEAD_DIM)
        assert keys.shape == (_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM)
        assert values.shape == (_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM)
        assert gate is None

    def test_v_norm_is_applied_when_present(self) -> None:
        # Arrange
        inner = _make_inner(with_v_proj=True, with_v_norm=True)
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))

        # Act
        prepare_sdpa_qkv(inner, x, ctx, _N_HEADS, _N_KV_HEADS, shared_kv=None)

        # Assert — v_norm callback was invoked exactly once on values
        assert inner._v_norm_calls == 1

    def test_yoco_reuses_shared_kv_without_reprojection(self) -> None:
        # Arrange — Gemma4 YOCO: shared_kv comes from a prior layer and
        # should be returned untouched (no reprojection, no re-norm).
        inner = _make_inner(with_v_proj=True)
        # Swap in a linear that explodes if called — proves the YOCO path
        # skips v_proj rather than silently falling back to projection.
        inner.v_proj = _RaisingLinear(
            inner.v_proj.weight, "v_proj must not be called on YOCO path"
        )
        inner.k_proj = _RaisingLinear(
            inner.k_proj.weight, "k_proj must not be called on YOCO path"
        )
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))

        shared_k = mx.full((_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM), 7.0)
        shared_v = mx.full((_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM), 9.0)

        # Act
        _, keys, values, _, kv_for_sharing = prepare_sdpa_qkv(
            inner,
            x,
            ctx,
            _N_HEADS,
            _N_KV_HEADS,
            shared_kv=(shared_k, shared_v),
        )

        # Assert — shared tensors flow through unchanged
        assert keys is shared_k
        assert values is shared_v
        assert kv_for_sharing == (shared_k, shared_v)

    def test_read_existing_kv_prepares_q_without_kv_projection(self) -> None:
        # Arrange — MTP assistant layers read target K/V from paged cache, so
        # K/V projection must not be used as a control path.
        inner = _make_inner(with_v_proj=True)
        inner.k_proj = _RaisingLinear(
            inner.k_proj.weight,
            "k_proj must not be called when reading existing KV",
        )
        inner.v_proj = _RaisingLinear(
            inner.v_proj.weight,
            "v_proj must not be called when reading existing KV",
        )
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))

        # Act
        queries, keys, values, gate, kv_for_sharing = prepare_sdpa_qkv(
            inner,
            x,
            ctx,
            _N_HEADS,
            _N_KV_HEADS,
            read_existing_kv=True,
        )

        # Assert — Q is prepared normally, while K/V are local shape holders;
        # sdpa_forward uses the explicit flag to read authoritative cache K/V.
        mx.eval(queries, keys, values)
        assert queries.shape == (_BATCH, _N_HEADS, _SEQ_LEN, _HEAD_DIM)
        assert keys.shape == (_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM)
        assert values.shape == (_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM)
        assert bool(mx.all(keys == 0).item())
        assert bool(mx.all(values == 0).item())
        assert gate is None
        assert kv_for_sharing == (keys, values)

    def test_read_existing_kv_rejects_shared_kv_pair(self) -> None:
        inner = _make_inner(with_v_proj=True)
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))
        shared_k = mx.full((_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM), 1.0)
        shared_v = mx.full((_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM), 1.0)

        with pytest.raises(ValueError, match="mutually exclusive"):
            prepare_sdpa_qkv(
                inner,
                x,
                ctx,
                _N_HEADS,
                _N_KV_HEADS,
                shared_kv=(shared_k, shared_v),
                read_existing_kv=True,
            )

    def test_yoco_requires_rope_attribute(self) -> None:
        # Arrange — YOCO path must still reject non-RoPE models; without a
        # guard we would silently return un-rotated queries.
        inner = _make_inner(with_v_proj=True)
        del inner.rope
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))
        shared_k = mx.full((_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM), 1.0)
        shared_v = mx.full((_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM), 1.0)

        # Act / Assert
        with pytest.raises(NotImplementedError, match="rope"):
            prepare_sdpa_qkv(
                inner,
                x,
                ctx,
                _N_HEADS,
                _N_KV_HEADS,
                shared_kv=(shared_k, shared_v),
            )

    def test_contract_without_rope_keeps_projections_unrotated(self) -> None:
        inner = _make_inner(with_v_proj=True)
        del inner.rope
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))
        expected_q = mx.full((_BATCH, _N_HEADS, _SEQ_LEN, _HEAD_DIM), float(_HIDDEN))
        expected_k = mx.full((_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM), float(_HIDDEN))

        queries, keys, _, _, _ = prepare_sdpa_qkv(
            inner,
            x,
            ctx,
            _N_HEADS,
            _N_KV_HEADS,
            attention_contract=AttentionContract(use_rope=False),
        )

        assert mx.array_equal(queries, expected_q)
        assert mx.array_equal(keys, expected_k)

    def test_contract_without_rope_covers_the_shared_kv_path(self) -> None:
        inner = _make_inner(with_v_proj=True)
        del inner.rope
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))
        shared_k = mx.full((_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM), 1.0)
        shared_v = mx.full((_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM), 1.0)
        expected_q = mx.full((_BATCH, _N_HEADS, _SEQ_LEN, _HEAD_DIM), float(_HIDDEN))

        queries, keys, _, _, _ = prepare_sdpa_qkv(
            inner,
            x,
            ctx,
            _N_HEADS,
            _N_KV_HEADS,
            shared_kv=(shared_k, shared_v),
            attention_contract=AttentionContract(use_rope=False),
        )

        assert mx.array_equal(queries, expected_q)
        assert mx.array_equal(keys, shared_k)

    def test_precomputed_rope_path_does_not_require_rope_attribute(self) -> None:
        # Arrange — some VLMs precompute ``(cos, sin)`` at the model level
        # and pass those embeddings into ``self_attn`` positionally.
        inner = _make_inner(with_v_proj=True)
        del inner.rope
        inner.rope_parameters = {"mrope_section": [1, 1, 1]}
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))
        cos = mx.ones((3, _BATCH, _SEQ_LEN, _HEAD_DIM), dtype=mx.float32)
        sin = mx.zeros((3, _BATCH, _SEQ_LEN, _HEAD_DIM), dtype=mx.float32)
        expected_q = mx.full((_BATCH, _N_HEADS, _SEQ_LEN, _HEAD_DIM), 5.0)
        expected_k = mx.full((_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM), 7.0)

        with patch.object(
            sdpa_mod,
            "apply_attention_rope",
            return_value=(expected_q, expected_k),
        ) as apply_rope:
            queries, keys, values, gate, kv_for_sharing = prepare_sdpa_qkv(
                inner,
                x,
                ctx,
                _N_HEADS,
                _N_KV_HEADS,
                position_embeddings=(cos, sin),
            )

        assert apply_rope.call_count == 1
        position_embeddings_arg = apply_rope.call_args.kwargs["position_embeddings"]
        assert position_embeddings_arg[0] is cos
        assert position_embeddings_arg[1] is sin
        assert queries is expected_q
        assert keys is expected_k
        assert values.shape == (_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM)
        assert gate is None
        assert kv_for_sharing == (expected_k, values)


class _PagedOpsCapsOff:
    """Baseline capability answer: a current build reporting no GQA features."""

    def paged_attention_capabilities(self) -> dict[str, bool]:
        return {
            "gqa_decode": False,
            "gqa_disable": False,
            "decode_routing_metadata": False,
        }


class _PagedRoutingOpsSpy:
    def __init__(self, *, supports_mm_prefix: bool = True) -> None:
        self.calls: list[SimpleNamespace] = []
        self._supports_mm_prefix = supports_mm_prefix
        self.length_plans: list[tuple[int, ...]] = []

    def supports_mm_prefix(self) -> bool:
        return self._supports_mm_prefix

    def paged_attention_capabilities(self) -> dict[str, bool]:
        return {
            "gqa_decode": True,
            "gqa_disable": True,
            "decode_routing_metadata": True,
            "gqa_batch_context_lens": True,
            "gqa_length_plan": True,
        }

    def gqa_decode_length_plan(self, lengths):
        plan = tuple(lengths)
        self.length_plans.append(plan)
        return plan

    def reshape_and_cache(
        self,
        _key,
        _value,
        key_cache,
        value_cache,
        slot_mapping,
    ) -> tuple[mx.array, mx.array]:
        self.calls.append(
            SimpleNamespace(
                slot_mapping=slot_mapping.tolist(),
                cache_block_size=key_cache.shape[1],
            )
        )
        return key_cache, value_cache

    def paged_attention_primitive(
        self,
        _query: mx.array,
        _key_cache: mx.array,
        _value_cache: mx.array,
        _num_kv_heads: int,
        _scale: float,
        _softcap: float,
        block_tables: mx.array,
        _seq_lens: mx.array,
        _cu_seqlens: mx.array,
        block_size: int,
        _max_seq_len: int,
        _sliding_window: int,
        _out: mx.array,
        window_seqlen_q: int = 1,
        sinks: mx.array | None = None,
        mm_prefix_ranges: mx.array | None = None,
        num_decode_requests: int = -1,
        num_decode_tokens: int = 0,
        max_decode_context_len: int = 0,
        gqa_disabled: bool = False,
        gqa_context_lens: list[int] | None = None,
        gqa_length_plan: object = None,
    ) -> None:
        del window_seqlen_q, sinks
        self.calls[-1].block_tables = block_tables.tolist()
        self.calls[-1].block_size = block_size
        self.calls[-1].num_decode_requests = num_decode_requests
        self.calls[-1].gqa_disabled = gqa_disabled
        self.calls[-1].mm_prefix_ranges = mm_prefix_ranges
        self.calls[-1].num_decode_tokens = num_decode_tokens
        self.calls[-1].max_decode_context_len = max_decode_context_len
        self.calls[-1].gqa_context_lens = gqa_context_lens
        self.calls[-1].gqa_length_plan = gqa_length_plan


class _PreMmPrefixOps(_PagedOpsCapsOff):
    """Old native signature: reports no new capability and rejects new keywords."""

    def __init__(self) -> None:
        self._spy = _PagedRoutingOpsSpy()
        self.calls = self._spy.calls

    def reshape_and_cache(self, *args):
        return self._spy.reshape_and_cache(*args)

    def paged_attention_primitive(
        self,
        *args,
        window_seqlen_q=1,
        sinks=None,
        num_decode_requests=-1,
    ):
        self._spy.paged_attention_primitive(
            *args,
            window_seqlen_q=window_seqlen_q,
            sinks=sinks,
            num_decode_requests=num_decode_requests,
        )


class TestSDPAForward:
    """Tests for ``sdpa_forward`` runtime argument propagation."""

    def test_deepseek_num_kv_heads_naming_reaches_kernel(self) -> None:
        """DeepSeek exposes ``num_kv_heads``, not ``num_key_value_heads``."""
        from mlx_lm.models.deepseek import DeepseekAttention, ModelArgs

        args = ModelArgs(
            model_type="deepseek",
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        inner = DeepseekAttention(args)
        x = mx.zeros((1, 2, 16), dtype=mx.float16)
        cache = MetalPagedKVCache(
            num_layers=1,
            num_kv_heads=2,
            head_dim=4,
            num_blocks=1,
            block_size=8,
            dtype=mx.float16,
        )
        captured: dict[str, int | float] = {}

        class _FakeOps(_PagedOpsCapsOff):
            def reshape_and_cache(
                self,
                _keys,
                _values,
                key_cache,
                value_cache,
                _slot_mapping,
            ) -> tuple[mx.array, mx.array]:
                return key_cache, value_cache

            def paged_attention_primitive(
                self,
                _query,
                _key_cache,
                _value_cache,
                num_kv_heads,
                scale,
                *_args,
                **_kwargs,
            ) -> None:
                captured["num_kv_heads"] = num_kv_heads
                captured["scale"] = scale

        with (
            patch.object(sdpa_mod, "get_ops", return_value=_FakeOps()),
            patch.object(
                sdpa_mod,
                "truncate_padded_output",
                return_value=mx.zeros((1, 2, 16), dtype=mx.float16),
            ),
        ):
            output, (keys, values) = sdpa_forward(
                inner,
                x,
                _make_ctx(2),
                cache,
                layer_idx=0,
            )

        mx.eval(output, keys, values)
        assert output.shape == (1, 2, 16)
        assert keys.shape == (1, 2, 2, 4)
        assert values.shape == (1, 2, 2, 4)
        assert captured["num_kv_heads"] == 2
        assert captured["scale"] == 0.5

    @pytest.mark.parametrize("disabled", [False, True])
    def test_gqa_policy_reaches_primitive(
        self, monkeypatch: pytest.MonkeyPatch, disabled: bool
    ) -> None:
        """The environment escape hatch and scheduler decode count reach native code."""
        from vllm_metal import envs

        monkeypatch.setenv("VLLM_METAL_DISABLE_GQA_DECODE", str(int(disabled)))
        read_env = MagicMock(
            wraps=envs.environment_variables["VLLM_METAL_DISABLE_GQA_DECODE"]
        )
        monkeypatch.setitem(
            envs.environment_variables, "VLLM_METAL_DISABLE_GQA_DECODE", read_env
        )
        spy = _PagedRoutingOpsSpy()
        capability_query = MagicMock(wraps=spy.paged_attention_capabilities)
        monkeypatch.setattr(spy, "paged_attention_capabilities", capability_query)
        inner = _make_inner()
        inner.o_proj = lambda out: out
        cache = MetalPagedKVCache(
            num_layers=1,
            num_kv_heads=_N_KV_HEADS,
            head_dim=_HEAD_DIM,
            num_blocks=1,
            block_size=8,
            dtype=mx.float16,
        )
        ctx = _make_ctx(_SEQ_LEN)
        ctx.num_decode_requests = 1
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN), dtype=mx.float16)
        zeros = mx.zeros((_BATCH, _SEQ_LEN, _N_HEADS * _HEAD_DIM), dtype=mx.float16)
        with (
            patch.object(sdpa_mod, "get_ops", return_value=spy),
            patch.object(sdpa_mod, "truncate_padded_output", return_value=zeros),
        ):
            sdpa_forward(inner, x, ctx, cache, layer_idx=0)
            monkeypatch.setenv("VLLM_METAL_DISABLE_GQA_DECODE", str(int(not disabled)))
            sdpa_forward(inner, x, ctx, cache, layer_idx=0)
            assert spy.calls[-1].gqa_disabled is disabled
            assert read_env.call_count == 1
            assert capability_query.call_count == 1
            next_ctx = _make_ctx(_SEQ_LEN)
            next_ctx.num_decode_requests = 1
            sdpa_forward(inner, x, next_ctx, cache, layer_idx=0)
        assert read_env.call_count == 2
        assert capability_query.call_count == 2
        assert spy.calls[-1].gqa_disabled is not disabled
        assert spy.calls[-1].num_decode_requests == 1

    @pytest.mark.parametrize("disabled", [False, True])
    @pytest.mark.parametrize("capability", ["none", "plan"])
    @pytest.mark.parametrize("kind", ["decode", "mixed", "expanded_verify"])
    def test_gqa_batch_metadata_is_only_forwarded_for_ordinary_decode(
        self, monkeypatch, disabled, capability, kind
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_DISABLE_GQA_DECODE", str(int(disabled)))
        if kind == "decode":
            decode, prefill = [([[0]], 1), ([[1]], 3)], []
        elif kind == "mixed":
            decode = [([[0]], 1), ([[1]], 3)]
            prefill = [([[2]], 1, 0)]
        else:
            decode, prefill = [([[0]], 1, 2), ([[1]], 3, 2)], []
        prepare_grouped(decode, prefill, (8,))
        ctx = get_context()
        assert ctx is not None and ctx.num_decode_requests == 2
        if kind == "mixed":
            assert ctx.num_decode_tokens == 2 and len(ctx.context_lens) == 3
        elif kind == "expanded_verify":
            assert ctx.num_decode_tokens == 4
        spy = _PagedRoutingOpsSpy()
        query = MagicMock(
            return_value={
                "gqa_decode": True,
                "gqa_disable": True,
                "decode_routing_metadata": True,
                "gqa_length_plan": capability == "plan",
            }
        )
        monkeypatch.setattr(spy, "paged_attention_capabilities", query)
        inner = _make_inner()
        inner.o_proj = lambda out: out
        cache = MetalPagedKVCache(
            num_layers=1,
            num_kv_heads=_N_KV_HEADS,
            head_dim=_HEAD_DIM,
            num_blocks=3,
            block_size=8,
            dtype=mx.float16,
        )
        total = len(ctx.slot_mapping)
        x = mx.ones((1, total, _HIDDEN), mx.float16)
        zeros = mx.zeros((1, total, _N_HEADS * _HEAD_DIM), mx.float16)
        eligible = kind == "decode" and not disabled
        with (
            patch.object(sdpa_mod, "get_ops", return_value=spy),
            patch.object(sdpa_mod, "truncate_padded_output", return_value=zeros),
        ):
            for _layer in range(3):
                sdpa_forward(inner, x, ctx, cache, layer_idx=0)
            plan = spy.calls[-1].gqa_length_plan
            assert plan == ((2, 4) if eligible and capability == "plan" else None)
            assert len(spy.length_plans) == int(eligible and capability == "plan")
            query.assert_called_once_with()
            if eligible and capability == "plan":
                # A new forward has new lengths and must not reuse the old plan.
                prepare_grouped([([[1]], 4), ([[0]], 2)], [], (8,))
                next_ctx = get_context()
                sdpa_forward(inner, x, next_ctx, cache, layer_idx=0)
                assert spy.length_plans == [(2, 4), (5, 3)]
                assert spy.calls[-1].gqa_length_plan is not plan

    def test_mixed_batch_routes_slots_and_page_tables_by_layer_group(self) -> None:
        """Full and sliding layers consume their scheduler-group metadata."""
        full = FullAttentionSpec(
            block_size=64,
            num_kv_heads=_N_KV_HEADS,
            head_size=_HEAD_DIM,
            dtype=torch.float16,
        )
        sliding = SlidingWindowSpec(
            block_size=16,
            num_kv_heads=_N_KV_HEADS,
            head_size=_HEAD_DIM,
            dtype=torch.float16,
            sliding_window=1024,
            page_size_padded=full.page_size_bytes,
        )
        groups = [
            KVCacheGroupSpec(layer_names=["full"], kv_cache_spec=full),
            KVCacheGroupSpec(layer_names=["sliding"], kv_cache_spec=sliding),
        ]
        vllm_config = VllmConfig()
        vllm_config.cache_config.kv_cache_layout = "LBNHC"
        config = get_kv_cache_config_from_groups(
            vllm_config, groups, 11 * full.page_size_bytes
        )
        config.kv_cache_layout = "LBNHC"
        cache = MetalPagedKVCache.from_upstream(
            KVCacheStorage(config), ["full", "sliding"]
        )
        prepare_grouped(
            [([[3], [8, 9]], 17, 1)],
            [([[4], [10]], 2, 0)],
            (64, 16),
        )
        ctx = get_context()
        assert ctx is not None

        inner = SimpleNamespace(
            n_heads=_N_HEADS,
            n_kv_heads=_N_KV_HEADS,
            scale=_HEAD_DIM**-0.5,
            o_proj=lambda out: out,
        )
        x = mx.ones((_BATCH, 3, _HIDDEN))
        queries = mx.ones((_BATCH, _N_HEADS, 3, _HEAD_DIM))
        keys = mx.ones((_BATCH, _N_KV_HEADS, 3, _HEAD_DIM))
        values = mx.ones((_BATCH, _N_KV_HEADS, 3, _HEAD_DIM))
        ops = _PagedRoutingOpsSpy()

        with (
            patch.object(
                sdpa_mod,
                "prepare_sdpa_qkv",
                return_value=(queries, keys, values, None, (keys, values)),
            ),
            patch.object(sdpa_mod, "get_ops", return_value=ops),
            patch.object(
                sdpa_mod,
                "truncate_padded_output",
                return_value=mx.zeros((_BATCH, 3, _N_HEADS * _HEAD_DIM)),
            ),
        ):
            sdpa_forward(inner, x, ctx, cache, layer_idx=0)
            sdpa_forward(inner, x, ctx, cache, layer_idx=1)

        full_call, sliding_call = ops.calls
        assert full_call.slot_mapping == [3 * 64 + 17, 4 * 64, 4 * 64 + 1]
        assert full_call.block_tables == [[6, 7], [8, 9]]
        assert full_call.cache_block_size == 64
        assert full_call.block_size == 32
        assert full_call.num_decode_requests == 1
        assert full_call.num_decode_tokens == 1
        assert full_call.max_decode_context_len == 18
        assert sliding_call.slot_mapping == [9 * 16 + 1, 10 * 16, 10 * 16 + 1]
        assert sliding_call.block_tables == [[8, 9], [10, 0]]
        assert sliding_call.cache_block_size == 16
        assert sliding_call.block_size == 16
        assert sliding_call.num_decode_requests == 1
        assert sliding_call.num_decode_tokens == 1
        assert sliding_call.max_decode_context_len == 18

    def test_kernel_uses_layer_heads_and_registered_default_scale(self) -> None:
        """Kernel metadata comes from the layer, not padded cache allocation.

        StableLM exposes no scale attribute. Mixed layouts may allocate a wider
        cache, but its registered default must use the projected query width.
        Unregistered scale-less modules must continue to fail before dispatch.
        """
        actual_kv_heads = 1
        cache = MetalPagedKVCache(
            num_layers=2,
            num_kv_heads=_N_KV_HEADS,
            head_dim=_HEAD_DIM,
            num_blocks=1,
            block_size=8,
            dtype=mx.float16,
            kv_heads_per_layer=[_N_KV_HEADS, actual_kv_heads],
            head_dim_per_layer=[_HEAD_DIM, _CACHE_HEAD_DIM],
        )

        from mlx_lm.models.stablelm import Attention, ModelArgs

        inner = Attention(
            ModelArgs(
                model_type="stablelm",
                vocab_size=32,
                hidden_size=_HIDDEN,
                num_attention_heads=_N_HEADS,
                num_hidden_layers=1,
                num_key_value_heads=actual_kv_heads,
                intermediate_size=16,
                rope_theta=10_000.0,
                use_qkv_bias=False,
                partial_rotary_factor=0.25,
                layer_norm_eps=1e-5,
            )
        )
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))

        queries = mx.ones((_BATCH, _N_HEADS, _SEQ_LEN, _HEAD_DIM))
        keys = mx.ones((_BATCH, actual_kv_heads, _SEQ_LEN, _HEAD_DIM))
        values = mx.ones((_BATCH, actual_kv_heads, _SEQ_LEN, _HEAD_DIM))
        kv_for_sharing = (keys, values)

        captured: dict[str, int | float] = {}

        class _FakeOps(_PagedOpsCapsOff):
            def reshape_and_cache(
                self,
                _key,
                _value,
                key_cache,
                value_cache,
                _slot_mapping,
            ) -> tuple[mx.array, mx.array]:
                return key_cache, value_cache

            def paged_attention_primitive(
                self,
                _query,
                _key_cache,
                _value_cache,
                num_kv_heads,
                scale,
                *_args,
                **_kwargs,
            ) -> None:
                captured["num_kv_heads"] = num_kv_heads
                captured["scale"] = scale

        with (
            patch.object(
                sdpa_mod,
                "prepare_sdpa_qkv",
                return_value=(queries, keys, values, None, kv_for_sharing),
            ),
            patch.object(sdpa_mod, "get_ops", return_value=_FakeOps()),
            patch.object(
                sdpa_mod,
                "truncate_padded_output",
                return_value=mx.zeros((_BATCH, _SEQ_LEN, _N_HEADS * _HEAD_DIM)),
            ),
        ):
            unregistered = SimpleNamespace(
                n_heads=_N_HEADS,
                n_kv_heads=actual_kv_heads,
                o_proj=lambda out: out,
            )
            with pytest.raises(AttributeError, match="neither 'scale' nor 'sm_scale'"):
                sdpa_forward(unregistered, x, ctx, cache, layer_idx=1)

            sdpa_forward(
                inner,
                x,
                ctx,
                cache,
                layer_idx=1,
                attention_contract=attention_contract_for(inner),
            )

        assert captured["num_kv_heads"] == actual_kv_heads
        assert captured["scale"] == pytest.approx(_HEAD_DIM**-0.5)

    def test_gpt_oss_scale_and_sinks_reach_kernel(self) -> None:
        """GPT-OSS exposes ``sm_scale`` and per-head attention sinks."""
        cache = MetalPagedKVCache(
            num_layers=1,
            num_kv_heads=_N_KV_HEADS,
            head_dim=_HEAD_DIM,
            num_blocks=1,
            block_size=8,
            dtype=mx.float16,
        )
        expected_scale = 0.125
        fp16_sinks = mx.arange(_N_HEADS, dtype=mx.float16)
        inner = SimpleNamespace(
            num_attention_heads=_N_HEADS,
            num_key_value_heads=_N_KV_HEADS,
            sm_scale=expected_scale,
            sinks=fp16_sinks,
            o_proj=lambda out: out,
        )
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))
        queries = mx.ones((_BATCH, _N_HEADS, _SEQ_LEN, _HEAD_DIM))
        keys = mx.ones((_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM))
        values = mx.ones((_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM))
        captured: dict[str, object] = {}

        class _FakeOps(_PagedOpsCapsOff):
            def reshape_and_cache(
                self,
                _key,
                _value,
                key_cache,
                value_cache,
                _slot_mapping,
            ) -> tuple[mx.array, mx.array]:
                return key_cache, value_cache

            def paged_attention_primitive(
                self,
                _query,
                _key_cache,
                _value_cache,
                _num_kv_heads,
                scale,
                *_args,
                sinks=None,
                **_kwargs,
            ) -> None:
                captured["scale"] = scale
                captured["sinks"] = sinks

        with (
            patch.object(
                sdpa_mod,
                "prepare_sdpa_qkv",
                return_value=(queries, keys, values, None, (keys, values)),
            ),
            patch.object(sdpa_mod, "get_ops", return_value=_FakeOps()),
            patch.object(
                sdpa_mod,
                "truncate_padded_output",
                return_value=mx.zeros((_BATCH, _SEQ_LEN, _N_HEADS * _HEAD_DIM)),
            ),
        ):
            sdpa_forward(inner, x, ctx, cache, layer_idx=0)

        sinks = captured["sinks"]
        assert captured["scale"] == expected_scale
        assert isinstance(sinks, mx.array)
        assert sinks.dtype == mx.float32

        # The widened tensor is memoized off the module, so a later forward
        # hands the kernel the same float32 array while the parameter keeps
        # its checkpoint dtype.
        first_sinks = sinks
        with (
            patch.object(
                sdpa_mod,
                "prepare_sdpa_qkv",
                return_value=(queries, keys, values, None, (keys, values)),
            ),
            patch.object(sdpa_mod, "get_ops", return_value=_FakeOps()),
            patch.object(
                sdpa_mod,
                "truncate_padded_output",
                return_value=mx.zeros((_BATCH, _SEQ_LEN, _N_HEADS * _HEAD_DIM)),
            ),
        ):
            sdpa_forward(inner, x, _make_ctx(_SEQ_LEN), cache, layer_idx=0)

        assert inner.sinks is fp16_sinks
        assert captured["sinks"] is first_sinks

    def test_float32_sinks_cache_stays_off_the_parameter_tree(self) -> None:
        class _Attention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sinks = mx.arange(_N_HEADS, dtype=mx.bfloat16)

        inner = _Attention()
        first = sdpa_mod._float32_sinks(inner)

        assert first is not None and first.dtype == mx.float32
        assert sdpa_mod._float32_sinks(inner) is first
        parameters = dict(tree_flatten(inner.parameters()))
        assert parameters.keys() == {"sinks"}
        assert parameters["sinks"].dtype == mx.bfloat16

        inner.sinks = mx.zeros((_N_HEADS,), dtype=mx.bfloat16)
        assert sdpa_mod._float32_sinks(inner) is not first

    def _run_capturing_softcap(self, inner: SimpleNamespace) -> float:
        """Drive ``sdpa_forward`` and return the softcap the kernel received."""
        cache = MetalPagedKVCache(
            num_layers=1,
            num_kv_heads=_N_KV_HEADS,
            head_dim=_HEAD_DIM,
            num_blocks=1,
            block_size=8,
            dtype=mx.float16,
        )
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))
        queries = mx.ones((_BATCH, _N_HEADS, _SEQ_LEN, _HEAD_DIM))
        keys = mx.ones((_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM))
        values = mx.ones((_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM))
        captured: dict[str, float] = {}

        class _FakeOps(_PagedOpsCapsOff):
            def reshape_and_cache(
                self,
                _key,
                _value,
                key_cache,
                value_cache,
                _slot_mapping,
            ) -> tuple[mx.array, mx.array]:
                return key_cache, value_cache

            def paged_attention_primitive(
                self,
                _query,
                _key_cache,
                _value_cache,
                _num_kv_heads,
                _scale,
                softcap,
                *_args,
                **_kwargs,
            ) -> None:
                captured["softcap"] = softcap

        with (
            patch.object(
                sdpa_mod,
                "prepare_sdpa_qkv",
                return_value=(queries, keys, values, None, (keys, values)),
            ),
            patch.object(sdpa_mod, "get_ops", return_value=_FakeOps()),
            patch.object(
                sdpa_mod,
                "truncate_padded_output",
                return_value=mx.zeros((_BATCH, _SEQ_LEN, _N_HEADS * _HEAD_DIM)),
            ),
        ):
            sdpa_forward(inner, x, ctx, cache, layer_idx=0)

        return captured["softcap"]

    def test_attn_logit_softcapping_reaches_kernel(self) -> None:
        """Gemma 2 caps pre-softmax scores; the kernel must receive the value."""
        inner = SimpleNamespace(
            n_heads=_N_HEADS,
            n_kv_heads=_N_KV_HEADS,
            scale=_HEAD_DIM**-0.5,
            attn_logit_softcapping=50.0,
            o_proj=lambda out: out,
        )
        assert self._run_capturing_softcap(inner) == 50.0

    def test_softcap_defaults_to_disabled_without_the_attribute(self) -> None:
        """Architectures without the attribute keep the uncapped kernel path.

        The kernel treats any value <= 0 as disabled, so this is the
        regression guard that plumbing the value changed nothing for the
        models that were already working.
        """
        inner = SimpleNamespace(
            n_heads=_N_HEADS,
            n_kv_heads=_N_KV_HEADS,
            scale=_HEAD_DIM**-0.5,
            o_proj=lambda out: out,
        )
        assert self._run_capturing_softcap(inner) == 0.0

    def test_shared_kv_path_does_not_rebind_cache_arrays(self) -> None:
        """Shared-KV attention must read the existing cache without writing."""
        cache = MetalPagedKVCache(
            num_layers=1,
            num_kv_heads=_N_KV_HEADS,
            head_dim=_HEAD_DIM,
            num_blocks=1,
            block_size=8,
            dtype=mx.float16,
        )
        original_k_cache = cache.key_caches[0]
        original_v_cache = cache.value_caches[0]

        inner = SimpleNamespace(
            n_heads=_N_HEADS,
            n_kv_heads=_N_KV_HEADS,
            scale=_HEAD_DIM**-0.5,
            o_proj=lambda out: out,
        )
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))
        queries = mx.ones((_BATCH, _N_HEADS, _SEQ_LEN, _HEAD_DIM))
        shared_k = mx.ones((_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM))
        shared_v = mx.ones((_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM))
        captured: dict[str, mx.array] = {}

        class _FakeOps(_PagedOpsCapsOff):
            def paged_attention_primitive(
                self,
                _query,
                key_cache,
                value_cache,
                *_args,
                **_kwargs,
            ) -> None:
                captured["key_cache"] = key_cache
                captured["value_cache"] = value_cache

        with (
            patch.object(
                sdpa_mod,
                "prepare_sdpa_qkv",
                return_value=(queries, shared_k, shared_v, None, (shared_k, shared_v)),
            ),
            patch.object(sdpa_mod, "get_ops", return_value=_FakeOps()),
            patch.object(
                sdpa_mod,
                "truncate_padded_output",
                return_value=mx.zeros((_BATCH, _SEQ_LEN, _N_HEADS * _HEAD_DIM)),
            ),
        ):
            sdpa_forward(
                inner, x, ctx, cache, layer_idx=0, shared_kv=(shared_k, shared_v)
            )

        assert cache.key_caches[0] is original_k_cache
        assert cache.value_caches[0] is original_v_cache
        assert captured["key_cache"] is original_k_cache
        assert captured["value_cache"] is original_v_cache

    def test_read_existing_kv_path_does_not_rebind_cache_arrays(self) -> None:
        """MTP cache-read mode must not write new K/V into the target cache."""
        cache = MetalPagedKVCache(
            num_layers=1,
            num_kv_heads=_N_KV_HEADS,
            head_dim=_HEAD_DIM,
            num_blocks=1,
            block_size=8,
            dtype=mx.float16,
        )
        original_k_cache = cache.key_caches[0]
        original_v_cache = cache.value_caches[0]

        inner = SimpleNamespace(
            n_heads=_N_HEADS,
            n_kv_heads=_N_KV_HEADS,
            scale=_HEAD_DIM**-0.5,
            o_proj=lambda out: out,
        )
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))
        queries = mx.ones((_BATCH, _N_HEADS, _SEQ_LEN, _HEAD_DIM))
        keys = mx.zeros((_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM))
        values = mx.zeros((_BATCH, _N_KV_HEADS, _SEQ_LEN, _HEAD_DIM))
        captured: dict[str, mx.array] = {}

        class _FakeOps(_PagedOpsCapsOff):
            def paged_attention_primitive(
                self,
                _query,
                key_cache,
                value_cache,
                *_args,
                **_kwargs,
            ) -> None:
                captured["key_cache"] = key_cache
                captured["value_cache"] = value_cache

        with (
            patch.object(
                sdpa_mod,
                "prepare_sdpa_qkv",
                return_value=(queries, keys, values, None, (keys, values)),
            ) as prepare_mock,
            patch.object(sdpa_mod, "get_ops", return_value=_FakeOps()),
            patch.object(
                sdpa_mod,
                "truncate_padded_output",
                return_value=mx.zeros((_BATCH, _SEQ_LEN, _N_HEADS * _HEAD_DIM)),
            ),
        ):
            sdpa_forward(inner, x, ctx, cache, layer_idx=0, read_existing_kv=True)

        assert prepare_mock.call_args.kwargs["read_existing_kv"] is True
        assert cache.key_caches[0] is original_k_cache
        assert cache.value_caches[0] is original_v_cache
        assert captured["key_cache"] is original_k_cache
        assert captured["value_cache"] is original_v_cache


class TestBidirectionalDispatch:
    """Image-block rows take the kernel path or the recompute per env."""

    @staticmethod
    def _cache(dtype: mx.Dtype = mx.float16) -> MetalPagedKVCache:
        """Gemma 4's layer mix over upstream storage: one full-attention layer
        at block 32 beside one sliding layer (window 1024) at block 16."""
        torch_dtype = torch.float32 if dtype == mx.float32 else torch.float16
        full = FullAttentionSpec(
            block_size=32,
            num_kv_heads=_N_KV_HEADS,
            head_size=_HEAD_DIM,
            dtype=torch_dtype,
        )
        sliding = SlidingWindowSpec(
            block_size=16,
            num_kv_heads=_N_KV_HEADS,
            head_size=_HEAD_DIM,
            dtype=torch_dtype,
            sliding_window=1024,
            page_size_padded=full.page_size_bytes,
        )
        vllm_config = VllmConfig()
        vllm_config.cache_config.kv_cache_layout = "LBNHC"
        config = get_kv_cache_config_from_groups(
            vllm_config,
            [
                KVCacheGroupSpec(layer_names=["full"], kv_cache_spec=full),
                KVCacheGroupSpec(layer_names=["sliding"], kv_cache_spec=sliding),
            ],
            11 * full.page_size_bytes,
        )
        config.kv_cache_layout = "LBNHC"
        return MetalPagedKVCache.from_upstream(
            KVCacheStorage(config), ["full", "sliding"]
        )

    def _run(
        self,
        kinds: frozenset[str],
        ranges,
        layer_idx: int,
        *,
        path: str | None = None,
        supports: bool = True,
        repeat: int = 1,
        dtype: mx.Dtype = mx.float16,
        ops: object | None = None,
        window_at_block_start: bool = False,
    ):
        cache = self._cache(dtype)
        # One decode row (17 cached tokens) and a 2-row prefill at positions 0..1.
        prepare_grouped([([[3], [8, 9]], 17, 1)], [([[4], [10]], 2, 0)], (32, 16))
        ctx = get_context()
        assert ctx is not None
        ctx.segment_bidi_ranges = ranges
        ctx.bidi_layer_kinds = kinds
        ctx.bidi_window_at_block_start = window_at_block_start
        inner = SimpleNamespace(
            n_heads=_N_HEADS,
            n_kv_heads=_N_KV_HEADS,
            scale=_HEAD_DIM**-0.5,
            o_proj=lambda out: out,
        )
        x = mx.ones((_BATCH, 3, _HIDDEN))
        queries = mx.ones((_BATCH, _N_HEADS, 3, _HEAD_DIM))
        keys = mx.ones((_BATCH, _N_KV_HEADS, 3, _HEAD_DIM))
        values = mx.ones((_BATCH, _N_KV_HEADS, 3, _HEAD_DIM))
        bidi = MagicMock(side_effect=lambda out, *a, **k: out)
        spy = ops or _PagedRoutingOpsSpy(supports_mm_prefix=supports)
        env = {k: v for k, v in os.environ.items() if k != "VLLM_METAL_MM_PREFIX_PATH"}
        if path is not None:
            env["VLLM_METAL_MM_PREFIX_PATH"] = path
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(
                sdpa_mod,
                "prepare_sdpa_qkv",
                return_value=(queries, keys, values, None, (keys, values)),
            ),
            patch.object(sdpa_mod, "get_ops", return_value=spy),
            patch.object(
                sdpa_mod,
                "truncate_padded_output",
                return_value=mx.zeros((_BATCH, 3, _N_HEADS * _HEAD_DIM)),
            ),
            patch.object(sdpa_mod, "apply_bidirectional_segments", bidi),
        ):
            for _ in range(repeat):
                sdpa_forward(inner, x, ctx, cache, layer_idx=layer_idx)
        return bidi, spy, ctx

    def test_recompute_path_keeps_the_splice(self) -> None:
        ranges = [None, [(0, 2)]]
        bidi, spy, _ = self._run(frozenset({"sliding"}), ranges, 0, path="recompute")
        assert bidi.call_count == 0
        bidi, spy, ctx = self._run(frozenset({"sliding"}), ranges, 1, path="recompute")
        assert bidi.call_count == 1
        kwargs = bidi.call_args.kwargs
        assert kwargs["window"] == 1024
        assert kwargs["head_dim"] == _HEAD_DIM
        assert kwargs["block_size"] == 16
        assert kwargs["cu_seqlens"] == [0, 1, 3]
        assert kwargs["turboquant"] is False
        assert spy.calls[-1].mm_prefix_ranges is None
        assert ctx.mm_prefix_rows_built is False

    def test_kernel_path_passes_rows_only_on_configured_layers(self) -> None:
        ranges = [None, [(0, 2)]]
        bidi, spy, ctx = self._run(frozenset({"sliding"}), ranges, 0)
        assert bidi.call_count == 0
        assert spy.calls[-1].mm_prefix_ranges is None
        assert ctx.mm_prefix_rows_built is False
        bidi, spy, ctx = self._run(frozenset({"sliding"}), ranges, 1)
        assert bidi.call_count == 0
        rows = spy.calls[-1].mm_prefix_ranges
        assert rows is not None and rows.dtype == mx.int32
        assert rows.tolist() == [[-1, -1], [0, 1], [0, 1]]
        assert ctx.mm_prefix_row_count == 2
        assert ctx.bidi_logged is True

    def test_full_kind_takes_the_kernel_path_on_full_layers(self) -> None:
        ranges = [None, [(0, 2)]]
        _, spy, _ = self._run(frozenset({"full"}), ranges, 0)
        assert spy.calls[-1].mm_prefix_ranges.tolist() == [[-1, -1], [0, 1], [0, 1]]
        _, spy, _ = self._run(frozenset({"full"}), ranges, 1)
        assert spy.calls[-1].mm_prefix_ranges is None

    def test_block_anchored_window_recomputes_sliding_layers_only(self) -> None:
        """DiffusionGemma's canvas: the kernel windows per row, so sliding
        layers recompute; full layers have no window and keep the kernel."""
        kinds, ranges = frozenset({"sliding", "full"}), [None, [(0, 2)]]
        bidi, spy, ctx = self._run(kinds, ranges, 1, window_at_block_start=True)
        assert bidi.call_count == 1
        assert bidi.call_args.kwargs["window"] == 1024
        assert spy.calls[-1].mm_prefix_ranges is None
        assert ctx.mm_prefix_rows_built is False
        bidi, spy, _ = self._run(kinds, ranges, 0, window_at_block_start=True)
        assert bidi.call_count == 0
        assert spy.calls[-1].mm_prefix_ranges.tolist() == [[-1, -1], [0, 1], [0, 1]]

    def test_float32_cache_falls_back_to_recompute(self) -> None:
        """The tiled kernel has no float32 instantiation (the recompute path)."""
        bidi, spy, ctx = self._run(
            frozenset({"sliding"}), [None, [(0, 2)]], 1, dtype=mx.float32
        )
        assert bidi.call_count == 1
        assert spy.calls[-1].mm_prefix_ranges is None
        assert ctx.mm_prefix_rows_built is False

    def test_unsupported_ops_fall_back_to_recompute(self) -> None:
        bidi, spy, _ = self._run(
            frozenset({"sliding"}), [None, [(0, 2)]], 1, supports=False
        )
        assert bidi.call_count == 1
        assert spy.calls[-1].mm_prefix_ranges is None

    @pytest.mark.parametrize("disable_env", [None, "0", "1"])
    def test_ops_predating_mm_prefix_serve_the_recompute(
        self, monkeypatch, disable_env
    ) -> None:
        # No probe and no keyword: the image rows still reach the recompute.
        if disable_env is None:
            monkeypatch.delenv("VLLM_METAL_DISABLE_GQA_DECODE", raising=False)
        else:
            monkeypatch.setenv("VLLM_METAL_DISABLE_GQA_DECODE", disable_env)
        bidi, spy, _ = self._run(
            frozenset({"sliding"}), [None, [(0, 2)]], 1, ops=_PreMmPrefixOps()
        )
        assert bidi.call_count == 1
        assert spy.calls[-1].gqa_disabled is False

    def test_unknown_gqa_native_cannot_silently_ignore_disable(self, monkeypatch):
        monkeypatch.setenv("VLLM_METAL_DISABLE_GQA_DECODE", "1")
        ops = _PreMmPrefixOps()
        # A GQA build without disable support must fail, not drop the switch.
        monkeypatch.setattr(
            ops,
            "paged_attention_capabilities",
            lambda: {
                "gqa_decode": True,
                "gqa_disable": False,
                "decode_routing_metadata": False,
            },
        )
        with pytest.raises(RuntimeError, match="rebuild.*native extension"):
            self._run(frozenset({"sliding"}), [None, [(0, 2)]], 1, ops=ops)

    def test_unknown_path_value_raises(self) -> None:
        with pytest.raises(ValueError, match="VLLM_METAL_MM_PREFIX_PATH"):
            self._run(frozenset({"sliding"}), [None, [(0, 2)]], 1, path="kernle")

    def test_rows_are_built_once_per_forward(self) -> None:
        with patch.object(
            sdpa_mod, "build_mm_prefix_rows", wraps=sdpa_mod.build_mm_prefix_rows
        ) as build:
            _, spy, ctx = self._run(
                frozenset({"sliding"}), [None, [(0, 2)]], 1, repeat=3
            )
        assert build.call_count == 1
        first = spy.calls[0].mm_prefix_ranges
        assert all(c.mm_prefix_ranges is first for c in spy.calls)
        assert ctx.mm_prefix_rows is first

    def test_no_rows_inside_a_block_passes_none(self) -> None:
        _, spy, ctx = self._run(frozenset({"sliding"}), [None, [(10, 12)]], 1)
        assert spy.calls[-1].mm_prefix_ranges is None
        assert ctx.mm_prefix_rows_built is True
        assert ctx.bidi_logged is False

    def test_no_ranges_never_enters(self) -> None:
        bidi, spy, _ = self._run(frozenset({"sliding"}), None, 1)
        assert bidi.call_count == 0
        assert spy.calls[-1].mm_prefix_ranges is None

    def test_kernel_path_logs_the_row_count_once(self) -> None:
        records: list[logging.LogRecord] = []

        class _Sink(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        logger = logging.getLogger("vllm_metal.attention.impls.sdpa")
        sink = _Sink(level=logging.INFO)
        logger.addHandler(sink)
        previous = logger.level
        logger.setLevel(logging.INFO)
        try:
            self._run(frozenset({"sliding"}), [None, [(0, 2)]], 1, repeat=2)
        finally:
            logger.setLevel(previous)
            logger.removeHandler(sink)
        lines = [
            r.getMessage() for r in records if "mm_prefix ranges" in r.getMessage()
        ]
        assert lines == ["Metal: mm_prefix ranges on 2 row(s)"]


# === Laguna g_proj per-head gating ===


class TestApplyGProjGate:
    """Cover Laguna-style ``g_proj`` + softplus per-head output gating."""

    def test_noop_without_g_proj(self) -> None:
        """Modules without ``g_proj`` (every non-Laguna SDPA model) pass through."""
        inner = SimpleNamespace()  # no g_proj, no gating
        out = mx.ones((_BATCH, _SEQ_LEN, _N_HEADS * _HEAD_DIM))
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))
        result = apply_g_proj_gate(inner, out, x, _N_HEADS, _HEAD_DIM)
        assert result is out

    def test_noop_when_gating_disabled(self) -> None:
        """A ``g_proj`` with ``gating=False`` is ignored (defensive guard)."""
        inner = SimpleNamespace(gating=False, g_proj=nn.Linear(_HIDDEN, _N_HEADS))
        out = mx.ones((_BATCH, _SEQ_LEN, _N_HEADS * _HEAD_DIM))
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))
        result = apply_g_proj_gate(inner, out, x, _N_HEADS, _HEAD_DIM)
        assert result is out

    def test_matches_reference_gating(self) -> None:
        """Output matches the mlx_lm Laguna reference gating computation."""
        g_proj = nn.Linear(_HIDDEN, _N_HEADS, bias=False)
        inner = SimpleNamespace(gating=True, g_proj=g_proj)
        x = mx.random.normal((_BATCH, _SEQ_LEN, _HIDDEN))
        out = mx.random.normal((_BATCH, _SEQ_LEN, _N_HEADS * _HEAD_DIM))

        result = apply_g_proj_gate(inner, out, x, _N_HEADS, _HEAD_DIM)

        # Reference: gate = softplus(g_proj(x)); out *= gate per head.
        gate = nn.softplus(g_proj(x).astype(mx.float32)).astype(out.dtype)
        expected = (
            out.reshape(_BATCH, _SEQ_LEN, _N_HEADS, _HEAD_DIM) * gate[..., None]
        ).reshape(_BATCH, _SEQ_LEN, -1)

        assert result.shape == out.shape
        assert mx.allclose(result, expected, atol=1e-6).item()


class TestPhiAttention:
    """Phi-1/1.5 (``mlx_lm.models.phi``) reaches the paged kernel.

    The real class spells its output projection ``dense`` (not ``o_proj``)
    and computes the standard softmax scale inline, so it needs both the
    dispatch alias and the query-derived-scale contract to serve on the
    paged path.
    """

    def test_phi_dispatch_and_derived_scale_reach_kernel(self) -> None:
        """is_sdpa accepts ``dense`` and the kernel sees head_dim**-0.5."""
        from mlx_lm.models.phi import ModelArgs, PhiAttention

        n_heads, n_kv_heads, head_dim = 4, 2, 16
        hidden = n_heads * head_dim
        inner = PhiAttention(
            ModelArgs(
                model_type="phi",
                vocab_size=32,
                hidden_size=hidden,
                num_attention_heads=n_heads,
                num_hidden_layers=1,
                num_key_value_heads=n_kv_heads,
                partial_rotary_factor=0.5,
                intermediate_size=16,
            )
        )

        assert sdpa_mod.is_sdpa(inner)

        cache = MetalPagedKVCache(
            num_layers=1,
            num_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_blocks=1,
            block_size=8,
            dtype=mx.float16,
        )
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, hidden))

        queries = mx.ones((_BATCH, n_heads, _SEQ_LEN, head_dim))
        keys = mx.ones((_BATCH, n_kv_heads, _SEQ_LEN, head_dim))
        values = mx.ones((_BATCH, n_kv_heads, _SEQ_LEN, head_dim))
        kv_for_sharing = (keys, values)

        captured: dict[str, float] = {}

        class _FakeOps(_PagedOpsCapsOff):
            def reshape_and_cache(
                self, _key, _value, key_cache, value_cache, _slot_mapping
            ):
                return key_cache, value_cache

            def paged_attention_primitive(
                self, _query, _key_cache, _value_cache, _num_kv_heads, scale, *_a, **_k
            ):
                captured["scale"] = scale

        with (
            patch.object(
                sdpa_mod,
                "prepare_sdpa_qkv",
                return_value=(queries, keys, values, None, kv_for_sharing),
            ),
            patch.object(sdpa_mod, "get_ops", return_value=_FakeOps()),
            patch.object(
                sdpa_mod,
                "truncate_padded_output",
                return_value=mx.zeros((_BATCH, _SEQ_LEN, hidden)),
            ),
        ):
            out = sdpa_forward(
                inner,
                x,
                ctx,
                cache,
                layer_idx=0,
                attention_contract=attention_contract_for(inner),
            )

        assert captured["scale"] == pytest.approx(head_dim**-0.5)
        # The output ran through PhiAttention.dense, not a missing o_proj.
        assert out is not None


class TestContiguousPrefillPath:
    """Routing and numerics for the contiguous flash-SDPA prefill read path."""

    _NH = 2
    _NKH = 2
    _HD = 64
    _HIDDEN = _NH * _HD
    _SEQ = 64
    _BS = 8
    _NB = _SEQ // _BS

    def _inner(self) -> SimpleNamespace:
        def w(rows: int) -> mx.array:
            return mx.random.normal((rows, self._HIDDEN)) * 0.05

        return SimpleNamespace(
            n_heads=self._NH,
            n_kv_heads=self._NKH,
            scale=self._HD**-0.5,
            q_proj=_FakeLinear(w(self._NH * self._HD)),
            k_proj=_FakeLinear(w(self._NKH * self._HD)),
            v_proj=_FakeLinear(w(self._NKH * self._HD)),
            rope=_identity_rope,
            o_proj=lambda t: t,
        )

    def _ctx(self) -> PagedAttentionContext:
        return PagedAttentionContext(
            slot_mapping=list(range(self._SEQ)),
            block_tables=[list(range(self._NB))],
            context_lens=[self._SEQ],
            offsets=[],
            cu_seqlens=[0, self._SEQ],
        )

    def _cache(self) -> MetalPagedKVCache:
        return MetalPagedKVCache(
            num_layers=1,
            num_kv_heads=self._NKH,
            head_dim=self._HD,
            num_blocks=self._NB,
            block_size=self._BS,
            dtype=mx.float16,
        )

    def _fake_ops(self, calls: list) -> type:
        class _FakeOps(_PagedOpsCapsOff):
            def reshape_and_cache(self, _k, _v, key_cache, value_cache, _sm):
                return key_cache, value_cache

            def paged_attention_primitive(self, *args, **kwargs) -> None:
                calls.append(args)

        return _FakeOps

    def test_eligible_prefill_skips_paged_kernel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_SDPA_CONTIGUOUS_PREFILL", "1")
        monkeypatch.setattr(sdpa_mod, "_MIN_FAST_PATH_FLOPS", 0.0)
        calls: list = []
        x = mx.random.normal((1, self._SEQ, self._HIDDEN))
        with patch.object(sdpa_mod, "get_ops", return_value=self._fake_ops(calls)()):
            out, _ = sdpa_forward(
                self._inner(), x, self._ctx(), self._cache(), layer_idx=0
            )
        mx.eval(out)
        assert out.shape == (1, self._SEQ, self._HIDDEN)
        assert not calls

    def test_flops_floor_falls_back_to_paged_kernel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_SDPA_CONTIGUOUS_PREFILL", "1")
        monkeypatch.setattr(sdpa_mod, "_MIN_FAST_PATH_FLOPS", float("inf"))
        calls: list = []
        x = mx.random.normal((1, self._SEQ, self._HIDDEN))
        with (
            patch.object(sdpa_mod, "get_ops", return_value=self._fake_ops(calls)()),
            patch.object(
                sdpa_mod,
                "truncate_padded_output",
                return_value=mx.zeros((1, self._SEQ, self._HIDDEN)),
            ),
        ):
            sdpa_forward(self._inner(), x, self._ctx(), self._cache(), layer_idx=0)
        assert calls

    def test_matches_paged_kernel_numerically(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mx.random.seed(0)
        inner = self._inner()
        x = mx.random.normal((1, self._SEQ, self._HIDDEN))
        cache = self._cache()
        monkeypatch.setenv("VLLM_METAL_SDPA_CONTIGUOUS_PREFILL", "0")
        out_paged, _ = sdpa_forward(inner, x, self._ctx(), cache, layer_idx=0)
        mx.eval(out_paged)
        monkeypatch.setenv("VLLM_METAL_SDPA_CONTIGUOUS_PREFILL", "1")
        monkeypatch.setattr(sdpa_mod, "_MIN_FAST_PATH_FLOPS", 0.0)
        out_contig, _ = sdpa_forward(inner, x, self._ctx(), cache, layer_idx=0)
        mx.eval(out_contig)
        assert mx.allclose(out_paged, out_contig, atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize(
        ("head_dim", "query_len", "nax_supported"),
        [
            pytest.param(256, 512, True, id="wide-short-query"),
            pytest.param(256, 1024, False, id="wide-without-nax"),
            pytest.param(192, 1024, True, id="unfused-head192"),
            pytest.param(512, 1024, True, id="unsupported-head512"),
            pytest.param(64, 8, True, id="vector-query"),
        ],
    )
    def test_unfused_geometries_stay_paged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        head_dim: int,
        query_len: int,
        nax_supported: bool,
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_SDPA_CONTIGUOUS_PREFILL", "1")
        monkeypatch.setattr(sdpa_mod, "_MIN_FAST_PATH_FLOPS", 0.0)
        seq_len = 2048
        query = mx.zeros((query_len, 8, head_dim), dtype=mx.bfloat16)
        cache = mx.zeros((64, 32, 2, head_dim), dtype=mx.bfloat16)
        ctx = PagedAttentionContext(
            slot_mapping=[],
            context_lens=[seq_len],
            cu_seqlens=[0, query_len],
        )
        with (
            patch.object(
                sdpa_mod,
                "get_ops",
                return_value=SimpleNamespace(nax_supported=lambda: nax_supported),
            ),
            patch.object(mx.fast, "scaled_dot_product_attention") as fused,
        ):
            out = sdpa_mod._contiguous_prefill_attention(
                query,
                cache,
                cache,
                mx.arange(64, dtype=mx.int32)[None],
                mx.array(ctx.context_lens, dtype=mx.int32),
                mx.array(ctx.cu_seqlens, dtype=mx.int32),
                32,
                2,
                head_dim**-0.5,
                -1,
                None,
                0.0,
                ctx,
            )
        assert out is None
        fused.assert_not_called()

    @pytest.mark.parametrize("head_dim", [64, 72, 80, 96, 128, 256])
    def test_eligible_geometry_requires_fused_attention(
        self, monkeypatch: pytest.MonkeyPatch, head_dim: int
    ) -> None:
        monkeypatch.setenv("VLLM_METAL_SDPA_CONTIGUOUS_PREFILL", "1")
        monkeypatch.setattr(sdpa_mod, "_MIN_FAST_PATH_FLOPS", 0.0)
        query_len = 1024 if head_dim == 256 else 16
        num_blocks = (query_len + 31) // 32
        query = mx.zeros((query_len, 8, head_dim), dtype=mx.float16)
        cache = mx.zeros((num_blocks, 32, 2, head_dim), dtype=mx.float16)
        ctx = PagedAttentionContext(
            slot_mapping=[],
            context_lens=[query_len],
            cu_seqlens=[0, query_len],
        )
        with (
            patch.object(
                sdpa_mod,
                "get_ops",
                return_value=SimpleNamespace(nax_supported=lambda: True),
            ),
            patch.object(
                mx.fast,
                "scaled_dot_product_attention",
                side_effect=lambda q, *_args, **_kwargs: q,
            ) as fused,
        ):
            out = sdpa_mod._contiguous_prefill_attention(
                query,
                cache,
                cache,
                mx.arange(num_blocks, dtype=mx.int32)[None],
                mx.array(ctx.context_lens, dtype=mx.int32),
                mx.array(ctx.cu_seqlens, dtype=mx.int32),
                32,
                2,
                head_dim**-0.5,
                -1,
                None,
                0.0,
                ctx,
            )
        assert out is not None
        assert out.shape == query.shape
        fused.assert_called_once()
        assert fused.call_args.kwargs["force_fused"] is True
        assert fused.call_args.kwargs["mask"] == "causal"

    @pytest.mark.parametrize(
        "fallback",
        [
            "disabled",
            "nax_disabled",
            "float32",
            "sliding_window",
            "sinks",
            "softcap",
            "bidirectional",
            "verification",
            "decode_prefix",
            "one_token_segment",
        ],
    )
    def test_ineligible_attention_contract_stays_paged(
        self, monkeypatch: pytest.MonkeyPatch, fallback: str
    ) -> None:
        monkeypatch.setenv(
            "VLLM_METAL_SDPA_CONTIGUOUS_PREFILL",
            "0" if fallback == "disabled" else "1",
        )
        monkeypatch.setenv(
            "VLLM_METAL_DISABLE_NAX", "1" if fallback == "nax_disabled" else "0"
        )
        monkeypatch.setattr(sdpa_mod, "_MIN_FAST_PATH_FLOPS", 0.0)
        dtype = mx.float32 if fallback == "float32" else mx.float16
        query = mx.zeros((32, 8, 64), dtype=dtype)
        cache = mx.zeros((2, 32, 2, 64), dtype=dtype)
        ctx = PagedAttentionContext(
            slot_mapping=[], context_lens=[64], cu_seqlens=[0, 32]
        )
        block_tables = mx.array([[0, 1]], dtype=mx.int32)
        if fallback == "bidirectional":
            ctx.segment_bidi_ranges = [[(32, 64)]]
        elif fallback == "verification":
            ctx.verify_window_q = 32
        elif fallback == "decode_prefix":
            ctx.num_decode_requests = 1
        elif fallback == "one_token_segment":
            ctx.context_lens = [1, 63]
            ctx.cu_seqlens = [0, 1, 32]
            block_tables = mx.array([[0, 0], [0, 1]], dtype=mx.int32)
        with patch.object(mx.fast, "scaled_dot_product_attention") as fused:
            out = sdpa_mod._contiguous_prefill_attention(
                query,
                cache,
                cache,
                block_tables,
                mx.array(ctx.context_lens, dtype=mx.int32),
                mx.array(ctx.cu_seqlens, dtype=mx.int32),
                32,
                2,
                64**-0.5,
                32 if fallback == "sliding_window" else -1,
                mx.zeros((8,)) if fallback == "sinks" else None,
                1.0 if fallback == "softcap" else 0.0,
                ctx,
            )
        assert out is None
        fused.assert_not_called()

    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
    @pytest.mark.parametrize(
        ("head_dim", "block_size", "request_lengths"),
        [
            pytest.param(64, 32, ((17, 67), (33, 95)), id="prefix-small-pages"),
            pytest.param(128, 544, ((35, 559), (17, 97)), id="chunk-large-pages"),
            pytest.param(256, 784, ((1024, 1573), (1568, 2387)), id="nax-d256-prefix"),
        ],
    )
    def test_upstream_gqa_prefix_matches_independent_causal_reference(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dtype: mx.Dtype,
        head_dim: int,
        block_size: int,
        request_lengths: tuple[tuple[int, int], ...],
    ) -> None:
        """Keep prefixes, page translation, tail masking and row order observable."""
        if head_dim == 256 and not sdpa_mod.get_ops().nax_supported():
            pytest.skip("D256 fused prefill requires NAX hardware")
        mx.random.seed(11)
        monkeypatch.setattr(sdpa_mod, "_MIN_FAST_PATH_FLOPS", 0.0)
        n_heads, n_kv_heads = 8, 2
        context_lens = [length for _, length in request_lengths]
        block_counts = [
            (length + block_size - 1) // block_size for length in context_lens
        ]
        num_blocks = sum(block_counts) + 2
        spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=n_kv_heads,
            head_size=head_dim,
            dtype=torch.float16 if dtype == mx.float16 else torch.bfloat16,
        )
        vllm_config = VllmConfig()
        vllm_config.cache_config.kv_cache_layout = "LBNHC"
        config = get_kv_cache_config_from_groups(
            vllm_config,
            [KVCacheGroupSpec(layer_names=["attention"], kv_cache_spec=spec)],
            num_blocks * spec.page_size_bytes,
        )
        config.kv_cache_layout = "LBNHC"
        storage = KVCacheStorage(config)
        cache = MetalPagedKVCache.from_upstream(storage, ["attention"])
        # Unused physical pages and the last page's unwritten rows must never
        # contribute. Retain the upstream interleaved K/V views and strides.
        storage.buffers = [
            mx.full(buffer.view(dtype).shape, float("nan"), dtype=dtype).view(
                buffer.dtype
            )
            for buffer in storage.buffers
        ]
        physical_blocks = list(reversed(range(num_blocks)))
        tables = []
        cursor = 0
        for count in block_counts:
            tables.append(physical_blocks[cursor : cursor + count])
            cursor += count

        prefix_slots = []
        prefix_indices = []
        current_indices = []
        current_slots = []
        cu_seqlens = [0]
        offsets = []
        base = 0
        for table, (query_len, context_len) in zip(
            tables, request_lengths, strict=True
        ):
            slots = [
                table[pos // block_size] * block_size + pos % block_size
                for pos in range(context_len)
            ]
            prefix_len = context_len - query_len
            prefix_slots.extend(slots[:prefix_len])
            prefix_indices.extend(range(base, base + prefix_len))
            current_indices.extend(range(base + prefix_len, base + context_len))
            current_slots.extend(slots[prefix_len:])
            offsets.append(prefix_len)
            cu_seqlens.append(cu_seqlens[-1] + query_len)
            base += context_len

        total_q = cu_seqlens[-1]
        q = mx.random.normal((total_q, n_heads, head_dim)).astype(dtype)
        k = mx.random.normal((base, n_kv_heads, head_dim)).astype(dtype)
        v = mx.random.normal((base, n_kv_heads, head_dim)).astype(dtype)
        cached = mx.array(prefix_indices, dtype=mx.int32)
        key_cache, value_cache = sdpa_mod.get_ops().reshape_and_cache(
            k[cached],
            v[cached],
            cache.key_caches[0],
            cache.value_caches[0],
            mx.array(prefix_slots, dtype=mx.int64),
        )
        cache.replace_layer_cache(0, key_cache, value_cache)
        mx.eval(*storage.buffers)

        reference_parts = []
        base = 0
        for i, (query_len, context_len) in enumerate(request_lengths):
            qi = q[cu_seqlens[i] : cu_seqlens[i + 1]].transpose(1, 0, 2)
            ki = k[base : base + context_len].transpose(1, 0, 2)
            vi = v[base : base + context_len].transpose(1, 0, 2)
            ki = mx.repeat(ki.astype(mx.float32), n_heads // n_kv_heads, axis=0)
            vi = mx.repeat(vi.astype(mx.float32), n_heads // n_kv_heads, axis=0)
            scores = (qi.astype(mx.float32) * head_dim**-0.5) @ ki.transpose(0, 2, 1)
            visible = (
                mx.arange(context_len)[None, :]
                <= mx.arange(context_len - query_len, context_len)[:, None]
            )
            scores = mx.where(visible, scores, -float("inf"))
            reference_parts.append(
                (mx.softmax(scores, axis=-1) @ vi).transpose(1, 0, 2)
            )
            base += context_len
        reference = mx.concatenate(reference_parts).reshape(
            1, total_q, n_heads * head_dim
        )
        selected = mx.array(current_indices, dtype=mx.int32)
        queries = q.transpose(1, 0, 2)[None]
        keys = k[selected].transpose(1, 0, 2)[None]
        values = v[selected].transpose(1, 0, 2)[None]
        ctx = PagedAttentionContext(
            slot_mapping=current_slots,
            block_tables=tables,
            context_lens=context_lens,
            offsets=offsets,
            cu_seqlens=cu_seqlens,
        )
        inner = SimpleNamespace(
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            scale=head_dim**-0.5,
            o_proj=lambda t: t,
        )
        x = mx.zeros((1, total_q, n_heads * head_dim), dtype=dtype)
        with patch.object(
            sdpa_mod,
            "prepare_sdpa_qkv",
            return_value=(queries, keys, values, None, (keys, values)),
        ):
            monkeypatch.setenv("VLLM_METAL_SDPA_CONTIGUOUS_PREFILL", "1")
            with patch.object(
                mx.fast,
                "scaled_dot_product_attention",
                wraps=mx.fast.scaled_dot_product_attention,
            ) as fused:
                contiguous, _ = sdpa_forward(inner, x, ctx, cache, layer_idx=0)
                mx.eval(contiguous, reference)
            monkeypatch.setenv("VLLM_METAL_SDPA_CONTIGUOUS_PREFILL", "0")
            paged, _ = sdpa_forward(inner, x, ctx, cache, layer_idx=0)
            mx.eval(paged)
        assert fused.call_count == len(request_lengths)
        assert all(call.kwargs["force_fused"] for call in fused.call_args_list)
        assert bool(mx.all(mx.isfinite(contiguous)))
        assert mx.allclose(contiguous, reference, atol=2e-2, rtol=2e-2)
        assert mx.allclose(paged, reference, atol=2e-2, rtol=2e-2)
