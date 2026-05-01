# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Gemma4-specific branches in attention_sdpa.

Covers:
- ``pad_qkv_to_cache_head_dim`` / ``truncate_padded_output`` pure helpers.
- ``prepare_sdpa_qkv`` branches (K-eq-V fallback, v_norm, YOCO shared_kv).
- ``sdpa_forward`` propagation of per-layer ``num_kv_heads`` to the kernel.

The ``prepare_sdpa_qkv`` tests use minimal fake attention modules rather
than real mlx_lm Attention modules so they stay fast and deterministic.
Metal kernel dispatch itself is covered by the end-to-end smoke tests,
not here.
"""

from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

import vllm_metal.metal_kernel_backend.attention_sdpa as sdpa_mod
from vllm_metal.metal_kernel_backend.attention_sdpa import (
    pad_qkv_to_cache_head_dim,
    prepare_sdpa_qkv,
    sdpa_forward,
    truncate_padded_output,
)
from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache
from vllm_metal.paged_attention_common import PagedAttentionContext

# === Test fixtures (shared shapes) ===

_BATCH = 1
_SEQ_LEN = 4
_HIDDEN = 8
_N_HEADS = 2
_N_KV_HEADS = 2
_HEAD_DIM = 4  # small enough for fast unit tests
_CACHE_HEAD_DIM = 8  # Gemma4-style: cache wider than layer head_dim


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


class TestSDPAForward:
    """Tests for ``sdpa_forward`` runtime argument propagation."""

    def test_kernel_receives_per_layer_kv_heads(self) -> None:
        """Heterogeneous layers must pass their concrete KV-head count.

        Regression guard for Gemma4-style mixed KV layouts: ``sdpa_forward``
        resolves the layer's actual cache shape from ``kv_heads_per_layer``
        and must pass that same count to ``paged_attention_primitive``.
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
            head_dim_per_layer=[_HEAD_DIM, _HEAD_DIM],
        )

        inner = SimpleNamespace(
            n_heads=_N_HEADS,
            n_kv_heads=actual_kv_heads,
            scale=_HEAD_DIM**-0.5,
            o_proj=lambda out: out,
        )
        ctx = _make_ctx(_SEQ_LEN)
        x = mx.ones((_BATCH, _SEQ_LEN, _HIDDEN))

        queries = mx.ones((_BATCH, _N_HEADS, _SEQ_LEN, _HEAD_DIM))
        keys = mx.ones((_BATCH, actual_kv_heads, _SEQ_LEN, _HEAD_DIM))
        values = mx.ones((_BATCH, actual_kv_heads, _SEQ_LEN, _HEAD_DIM))
        kv_for_sharing = (keys, values)

        captured: dict[str, int] = {}

        class _FakeOps:
            def paged_attention_primitive(
                self,
                _query,
                _key_cache,
                _value_cache,
                num_kv_heads,
                *_args,
                **_kwargs,
            ) -> None:
                captured["num_kv_heads"] = num_kv_heads

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
            sdpa_forward(inner, x, ctx, cache, layer_idx=1)

        assert captured["num_kv_heads"] == actual_kv_heads
