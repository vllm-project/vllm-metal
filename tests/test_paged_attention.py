# SPDX-License-Identifier: Apache-2.0
"""Tests for paged attention shared utilities — OffsetCache, prepare functions.

Run with:
    python -m pytest tests/test_paged_attention.py -v -s
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.rope_utils import (
    Llama3RoPE,
    ProportionalRoPE,
    SuScaledRoPE,
    YarnRoPE,
)

from vllm_metal.attention.context import (
    OffsetCache,
    clear_context,
    get_context,
    prepare_unified,
)
from vllm_metal.attention.impls.sdpa_wrapper import SDPAPagedAttentionWrapper


class TestOffsetCache:
    def test_offset_property(self):
        c = OffsetCache(42)
        assert c.offset == 42

    def test_make_mask_single_token(self):
        c = OffsetCache(10)
        assert c.make_mask(1) is None

    def test_make_mask_multi_token(self):
        c = OffsetCache(0)
        assert c.make_mask(5) == "causal"


class TestSDPAPagedAttentionWrapper:
    def teardown_method(self):
        clear_context()

    def test_exposes_inner_rope_attributes(self):
        class _Inner:
            def __init__(self):
                self.rotary_emb = object()
                self.rope = object()

        inner = _Inner()
        wrapper = SDPAPagedAttentionWrapper(
            inner,
            layer_idx=0,
            kv_cache=object(),
            block_size=16,
        )

        assert wrapper.rotary_emb is inner.rotary_emb
        assert wrapper.rope is inner.rope

    def test_exposes_inner_is_local(self):
        class _Inner:
            is_local = True

        wrapper = SDPAPagedAttentionWrapper(
            _Inner(), layer_idx=0, kv_cache=object(), block_size=16
        )

        assert wrapper.is_local is True

    def test_defaults_is_local_false_when_inner_lacks_it(self):
        wrapper = SDPAPagedAttentionWrapper(
            object(), layer_idx=0, kv_cache=object(), block_size=16
        )

        assert wrapper.is_local is False

    def test_forwards_precomputed_rope_embeddings_without_context(self):
        class _Inner:
            def __init__(self):
                self.calls = []

            def __call__(
                self,
                x,
                mask=None,
                cache=None,
                position_embeddings=None,
                **kwargs,
            ):
                self.calls.append(
                    {
                        "x": x,
                        "mask": mask,
                        "cache": cache,
                        "position_embeddings": position_embeddings,
                        "kwargs": kwargs,
                    }
                )
                return x

        inner = _Inner()
        wrapper = SDPAPagedAttentionWrapper(
            inner,
            layer_idx=0,
            kv_cache=object(),
            block_size=16,
        )
        x = mx.ones((1, 2, 4), dtype=mx.float32)
        position_embeddings = (
            mx.ones((3, 1, 2, 4), dtype=mx.float32),
            mx.zeros((3, 1, 2, 4), dtype=mx.float32),
        )

        out = wrapper(x, None, None, position_embeddings)

        assert out is x
        assert inner.calls[0]["position_embeddings"] is position_embeddings
        assert inner.calls[0]["kwargs"] == {}


class TestPrepare:
    def teardown_method(self):
        clear_context()

    def test_prepare_unified_prefill_single(self):
        # Single prefill request via prepare_unified (start_pos=0)
        prepare_unified([], [([10, 11], 5, 0)], block_size=4)
        ctx = get_context()

        # block 10: slots 40,41,42,43; block 11: slot 44
        assert ctx is not None
        assert ctx.slot_mapping == [40, 41, 42, 43, 44]
        assert ctx.block_tables == [[10, 11]]
        assert ctx.context_lens == [5]
        assert ctx.cu_seqlens == [0, 5]
        assert ctx.offsets == [0]

    def test_prepare_unified_prefill_packed(self):
        # Two prefill requests packed together
        prepare_unified([], [([10], 3, 0), ([20], 2, 0)], block_size=4)
        ctx = get_context()

        assert ctx is not None
        # Request 0: block 10, slots 40,41,42
        # Request 1: block 20, slots 80,81
        assert ctx.slot_mapping == [40, 41, 42, 80, 81]
        assert ctx.cu_seqlens == [0, 3, 5]
        assert ctx.block_tables == [[10], [20]]
        assert ctx.context_lens == [3, 2]
        assert ctx.offsets == [0, 0]

    def test_prepare_unified_prefill_multiblock(self):
        # Single prefill spanning two blocks
        prepare_unified([], [([5, 6], 5, 0)], block_size=4)
        ctx = get_context()

        assert ctx is not None
        assert ctx.cu_seqlens == [0, 5]
        # block 5: slots 20,21,22,23; block 6: slot 24
        assert ctx.slot_mapping == [20, 21, 22, 23, 24]
        assert ctx.block_tables == [[5, 6]]
        assert ctx.context_lens == [5]

    def test_prepare_unified_continuation_chunk(self):
        # Continuation chunk: 3 new tokens starting at position 4
        # block 10 has slots 40-43 (positions 0-3, already cached),
        # block 11 has slots 44-47 (positions 4-6 are the new tokens)
        prepare_unified([], [([10, 11], 3, 4)], block_size=4)
        ctx = get_context()

        assert ctx is not None
        # Only 3 tokens in the query (positions 4, 5, 6)
        assert ctx.cu_seqlens == [0, 3]
        # Slots for positions 4, 5, 6: block 11 slots 44, 45, 46
        assert ctx.slot_mapping == [44, 45, 46]
        assert ctx.block_tables == [[10, 11]]
        # Total context = start_pos + num_tokens = 4 + 3 = 7
        assert ctx.context_lens == [7]
        # RoPE offset = start_pos
        assert ctx.offsets == [4]

    def test_prepare_unified_decode_only(self):
        # Single decode request via prepare_unified
        decode_requests = [([5, 6], 7)]
        prepare_unified(decode_requests, [], block_size=4)
        ctx = get_context()

        # new_pos=7, block_ids[7//4]=block_ids[1]=6, slot=6*4+(7%4)=27
        assert ctx is not None
        assert ctx.slot_mapping == [27]
        assert ctx.context_lens == [8]
        assert ctx.offsets == [7]
        assert ctx.cu_seqlens == [0, 1]

    def test_prepare_unified_spec_decode_keeps_window_whole(self):
        # Speculative verification appends draft rows after the last token.
        # The window stays ONE multi-token segment (cu_seqlens aligned with
        # decode requests); per-row causality is the kernel's varlen job.
        prepare_unified([([5, 6, 7], 7, 3)], [], block_size=4)
        ctx = get_context()

        assert ctx is not None
        assert ctx.slot_mapping == [27, 28, 29]
        assert ctx.block_tables == [[5, 6, 7]]
        assert ctx.context_lens == [10]
        assert ctx.offsets == [7]
        assert ctx.cu_seqlens == [0, 3]
        assert ctx.verify_window_q == 3
        assert ctx.num_decode_requests == 1

    def test_prepare_unified_expands_window_when_merge_disabled(self):
        # merge_verify_windows=False restores the expanded per-token layout
        # for models the window path does not serve (MLA native decode,
        # hybrid GDN layers, heads past PA_WINDOW_MAX_HEAD_SIZE): one
        # single-token segment per window row, byte-for-byte the
        # pre-window-mode metadata.
        prepare_unified(
            [([5, 6, 7], 7, 3)], [], block_size=4, merge_verify_windows=False
        )
        ctx = get_context()

        assert ctx is not None
        assert ctx.slot_mapping == [27, 28, 29]
        assert ctx.block_tables == [[5, 6, 7], [5, 6, 7], [5, 6, 7]]
        assert ctx.context_lens == [8, 9, 10]
        assert ctx.offsets == [7, 8, 9]
        assert ctx.cu_seqlens == [0, 1, 2, 3]
        assert ctx.verify_window_q == 1
        assert ctx.num_decode_requests == 1

    def test_prepare_unified_mixed_window_and_prefill_stays_off_window_mode(self):
        # A verify window sharing the batch with a prefill chunk keeps its
        # merged segment but the batch reports verify_window_q=1, so the
        # dispatch routes it down the tiled path exactly like the
        # non-speculative case (window mode is pure-verification only).
        prepare_unified([([5, 6, 7], 7, 3)], [([10, 11], 5, 0)], block_size=4)
        ctx = get_context()

        assert ctx is not None
        assert ctx.slot_mapping == [27, 28, 29, 40, 41, 42, 43, 44]
        assert ctx.block_tables == [[5, 6, 7], [10, 11]]
        assert ctx.cu_seqlens == [0, 3, 8]
        assert ctx.context_lens == [10, 5]
        assert ctx.offsets == [7, 0]
        assert ctx.verify_window_q == 1
        assert ctx.num_decode_requests == 1

    def test_prepare_unified_mixed(self):
        # 1 decode + 1 prefill
        decode_requests = [([5, 6], 7)]  # seq_len=7
        prefill_requests = [([10, 11], 5, 0)]  # 5 tokens from position 0

        prepare_unified(decode_requests, prefill_requests, block_size=4)
        ctx = get_context()

        assert ctx is not None
        # Decode slot: pos=7, block 6, slot=6*4+3=27
        # Prefill slots: block 10 slots 40,41,42,43; block 11 slot 44
        assert ctx.slot_mapping == [27, 40, 41, 42, 43, 44]
        assert ctx.cu_seqlens == [0, 1, 6]
        assert ctx.offsets == [7, 0]
        assert ctx.context_lens == [8, 5]
        assert ctx.block_tables == [[5, 6], [10, 11]]


class TestPackedRoPE:
    """Tests for per-request RoPE position reset in packed prefill."""

    class RecordingRoPE(nn.RoPE):
        def __init__(self) -> None:
            super().__init__(dims=8, traditional=False, base=10_000.0)
            self.calls = []

        def __call__(self, x, offset=0):
            self.calls.append((x.shape, offset))
            return super().__call__(x, offset=offset)

    def test_native_rope_batches_single_token_segments(self):
        from vllm_metal.attention.impls.varlen_rope_compat import (
            apply_packed_rope,
        )

        rope = self.RecordingRoPE()
        reference_rope = nn.RoPE(dims=8, traditional=False, base=10_000.0)
        module = SimpleNamespace(rope=rope)
        q = mx.arange(1 * 3 * 4 * 8, dtype=mx.float32).reshape(1, 3, 4, 8) / 100
        k = mx.arange(1 * 2 * 4 * 8, dtype=mx.float32).reshape(1, 2, 4, 8) / 100
        cu_seqlens = [0, 1, 2, 3, 4]
        offsets = [3, 11, 29, 47]

        expected_q = mx.concatenate(
            [
                reference_rope(q[:, :, i : i + 1, :], offset=offsets[i])
                for i in range(4)
            ],
            axis=2,
        )
        expected_k = mx.concatenate(
            [
                reference_rope(k[:, :, i : i + 1, :], offset=offsets[i])
                for i in range(4)
            ],
            axis=2,
        )
        q_out, k_out = apply_packed_rope(module, q, k, cu_seqlens, offsets=offsets)
        mx.eval(expected_q, expected_k, q_out, k_out)

        assert len(rope.calls) == 2
        assert rope.calls[0][0] == (4, 3, 1, 8)
        assert rope.calls[1][0] == (4, 2, 1, 8)
        assert rope.calls[0][1].shape == (4,)
        assert rope.calls[0][1].tolist() == offsets
        assert rope.calls[1][1].shape == (4,)
        assert rope.calls[1][1].tolist() == offsets
        assert mx.allclose(q_out, expected_q, rtol=1e-5, atol=1e-5).item()
        assert mx.allclose(k_out, expected_k, rtol=1e-5, atol=1e-5).item()

    @pytest.mark.parametrize(
        "rope",
        [
            Llama3RoPE(
                8,
                scaling_config={
                    "factor": 8.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 8192,
                },
            ),
            ProportionalRoPE(8, 8),
            SuScaledRoPE(8),
            YarnRoPE(8),
        ],
    )
    def test_mlx_lm_rope_wrappers_match_segment_reference(self, rope):
        from vllm_metal.attention.impls.varlen_rope_compat import (
            apply_packed_rope,
        )

        module = SimpleNamespace(rope=rope)
        q = mx.arange(1 * 3 * 4 * 8, dtype=mx.float32).reshape(1, 3, 4, 8) / 100
        k = mx.arange(1 * 2 * 4 * 8, dtype=mx.float32).reshape(1, 2, 4, 8) / 100
        cu_seqlens = [0, 1, 2, 3, 4]
        offsets = [3, 11, 29, 47]

        expected_q = mx.concatenate(
            [rope(q[:, :, i : i + 1, :], offset=offsets[i]) for i in range(4)],
            axis=2,
        )
        expected_k = mx.concatenate(
            [rope(k[:, :, i : i + 1, :], offset=offsets[i]) for i in range(4)],
            axis=2,
        )
        q_out, k_out = apply_packed_rope(module, q, k, cu_seqlens, offsets=offsets)
        mx.eval(expected_q, expected_k, q_out, k_out)

        assert mx.allclose(q_out, expected_q, rtol=1e-5, atol=1e-5).item()
        assert mx.allclose(k_out, expected_k, rtol=1e-5, atol=1e-5).item()

    def test_native_rope_uses_vector_zero_offsets_for_implicit_offsets(self):
        from vllm_metal.attention.impls.varlen_rope_compat import (
            apply_packed_rope,
        )

        rope = self.RecordingRoPE()
        module = SimpleNamespace(rope=rope)
        cu_seqlens = [0, 1, 2, 3, 4]
        q = mx.zeros((1, 3, 4, 8), dtype=mx.float32)
        k = mx.zeros((1, 2, 4, 8), dtype=mx.float32)

        apply_packed_rope(module, q, k, cu_seqlens)

        assert len(rope.calls) == 2
        assert rope.calls[0][1].shape == (4,)
        assert rope.calls[0][1].tolist() == [0, 0, 0, 0]
        assert rope.calls[1][1].shape == (4,)
        assert rope.calls[1][1].tolist() == [0, 0, 0, 0]

    def test_batched_rope_preserves_keys_when_not_applied(self):
        from vllm_metal.attention.impls.varlen_rope_compat import (
            apply_packed_rope,
        )

        rope = self.RecordingRoPE()
        reference_rope = nn.RoPE(dims=8, traditional=False, base=10_000.0)
        module = SimpleNamespace(rope=rope)
        cu_seqlens = [0, 1, 2, 3, 4]
        offsets = [3, 17, 41, 83]
        q = mx.arange(1 * 3 * 4 * 8, dtype=mx.float32).reshape(1, 3, 4, 8) / 100
        k = mx.zeros((1, 2, 4, 8), dtype=mx.float32)
        expected_q = mx.concatenate(
            [
                reference_rope(q[:, :, i : i + 1, :], offset=offsets[i])
                for i in range(4)
            ],
            axis=2,
        )

        q_out, k_out = apply_packed_rope(
            module,
            q,
            k,
            cu_seqlens,
            offsets=offsets,
            apply_keys=False,
        )
        mx.eval(expected_q, q_out)

        assert len(rope.calls) == 1
        assert mx.allclose(q_out, expected_q, rtol=1e-5, atol=1e-5).item()
        assert k_out is k

    def test_multi_token_segments_keep_per_segment_call_contract(self):
        from vllm_metal.attention.impls.varlen_rope_compat import (
            apply_packed_rope,
        )

        rope = self.RecordingRoPE()
        module = SimpleNamespace(rope=rope)
        q = mx.zeros((1, 2, 6, 8))
        k = mx.zeros((1, 1, 6, 8))

        apply_packed_rope(module, q, k, [0, 2, 4, 6], offsets=[3, 5, 8])

        assert rope.calls == [
            ((1, 2, 2, 8), 3),
            ((1, 1, 2, 8), 3),
            ((1, 2, 2, 8), 5),
            ((1, 1, 2, 8), 5),
            ((1, 2, 2, 8), 8),
            ((1, 1, 2, 8), 8),
        ]

    def test_custom_rope_keeps_per_segment_call_contract(self):
        from vllm_metal.attention.impls.varlen_rope_compat import (
            apply_packed_rope,
        )

        class RecordingRoPE:
            def __init__(self):
                self.calls = []

            def __call__(self, x, offset=0):
                self.calls.append((x.shape, offset))
                return x + offset

        rope = RecordingRoPE()
        module = SimpleNamespace(rope=rope)
        q = mx.zeros((1, 2, 3, 8))
        k = mx.zeros((1, 1, 3, 8))

        q_out, k_out = apply_packed_rope(module, q, k, [0, 1, 2, 3], offsets=[3, 5, 8])

        assert [offset for _, offset in rope.calls] == [3, 3, 5, 5, 8, 8]
        assert q_out[0, 0, :, 0].tolist() == [3.0, 5.0, 8.0]
        assert k_out[0, 0, :, 0].tolist() == [3.0, 5.0, 8.0]

    def test_positions_reset_per_request(self):
        """Each packed request's RoPE should start from position 0."""
        import mlx.core as mx

        from vllm_metal.attention.impls.varlen_rope_compat import (
            apply_packed_rope,
        )

        # Minimal RoPE stub: returns input + offset so we can verify offsets
        class FakeRoPE:
            def rope(self, x, offset=0):
                return x + offset

        module = FakeRoPE()
        # Two requests packed: 3 tokens + 2 tokens
        # Shape: (1, heads=1, total_len=5, head_dim=2)
        q = mx.zeros((1, 1, 5, 2))
        k = mx.zeros((1, 1, 5, 2))
        cu_seqlens = [0, 3, 5]

        q_out, k_out = apply_packed_rope(module, q, k, cu_seqlens)

        # All values should be 0 (offset=0 for every request)
        assert q_out.shape == (1, 1, 5, 2)
        assert mx.allclose(q_out, mx.zeros_like(q_out)).item()
        assert mx.allclose(k_out, mx.zeros_like(k_out)).item()

    def test_rejects_positions_on_mlx_lm_rope_path(self):
        """Caller-provided positions are only valid on the M-RoPE path."""
        import mlx.core as mx

        from vllm_metal.attention.impls.varlen_rope_compat import (
            apply_packed_rope,
        )

        class FakeRoPE:
            def rope(self, x, offset=0):
                return x

        q = mx.zeros((1, 1, 3, 2))
        k = mx.zeros((1, 1, 3, 2))
        with pytest.raises(NotImplementedError, match="position-array slot"):
            apply_packed_rope(
                FakeRoPE(),
                q,
                k,
                [0, 3],
                positions=[mx.zeros((3, 1, 3), dtype=mx.int32)],
            )

    def test_attention_rope_uses_precomputed_mrope_payload(self, monkeypatch):
        """Precomputed M-RoPE apply policy lives in the compat layer."""
        import sys
        import types

        import mlx.core as mx

        from vllm_metal.attention.impls.varlen_rope_compat import (
            apply_attention_rope,
        )

        recorded: dict[str, object] = {}

        def fake_apply(q, k, cos, sin, mrope_section):
            recorded["q"] = q
            recorded["k"] = k
            recorded["cos"] = cos
            recorded["sin"] = sin
            recorded["mrope_section"] = mrope_section
            return q + 1, k + 2

        fake_mod = types.ModuleType("mlx_vlm.models.paddleocr_vl.language")
        fake_mod.apply_multimodal_rotary_pos_emb = fake_apply
        monkeypatch.setitem(
            sys.modules, "mlx_vlm.models.paddleocr_vl.language", fake_mod
        )

        class FakePrecomputedMRoPE:
            rope_parameters = {"mrope_section": [1, 1, 1]}

        q = mx.zeros((1, 1, 3, 2), dtype=mx.float32)
        k = mx.zeros((1, 1, 3, 2), dtype=mx.float32)
        cos = mx.ones((3, 1, 3, 2), dtype=mx.float32)
        sin = mx.zeros((3, 1, 3, 2), dtype=mx.float32)

        q_out, k_out = apply_attention_rope(
            FakePrecomputedMRoPE(),
            q,
            k,
            [0, 3],
            position_embeddings=(cos, sin),
        )

        assert recorded["q"] is q
        assert recorded["k"] is k
        assert recorded["cos"] is cos
        assert recorded["sin"] is sin
        assert recorded["mrope_section"] == [1, 1, 1]
        assert mx.allclose(q_out, q + 1).item()
        assert mx.allclose(k_out, k + 2).item()

    def test_attention_rope_uses_qwen35_precomputed_mrope_payload(self, monkeypatch):
        """Qwen3.5 non-fused M-RoPE fallback uses the interleaved apply path."""
        import sys
        import types

        import mlx.core as mx

        from vllm_metal.attention.impls.varlen_rope_compat import (
            apply_attention_rope,
        )

        recorded: dict[str, object] = {}

        def fake_apply(q, k, cos, sin):
            recorded["q"] = q
            recorded["k"] = k
            recorded["cos"] = cos
            recorded["sin"] = sin
            return q + 3, k + 4

        fake_mod = types.ModuleType("mlx_vlm.models.qwen3_5.language")
        fake_mod.apply_multimodal_rotary_pos_emb = fake_apply
        monkeypatch.setitem(sys.modules, "mlx_vlm.models.qwen3_5.language", fake_mod)

        class FakeRotaryEmbedding:
            style = "interleaved"

        class FakeQwen35PrecomputedMRoPE:
            rotary_emb = FakeRotaryEmbedding()

        q = mx.zeros((1, 1, 3, 2), dtype=mx.float32)
        k = mx.zeros((1, 1, 3, 2), dtype=mx.float32)
        cos = mx.ones((3, 1, 3, 2), dtype=mx.float32)
        sin = mx.zeros((3, 1, 3, 2), dtype=mx.float32)

        q_out, k_out = apply_attention_rope(
            FakeQwen35PrecomputedMRoPE(),
            q,
            k,
            [0, 3],
            position_embeddings=(cos, sin),
        )

        assert recorded["q"] is q
        assert recorded["k"] is k
        assert recorded["cos"] is cos
        assert recorded["sin"] is sin
        assert mx.allclose(q_out, q + 3).item()
        assert mx.allclose(k_out, k + 4).item()

    def test_mrope_uses_caller_positions_when_provided(self, monkeypatch):
        """When ``positions[i]`` is an array, M-RoPE consumes it directly."""
        import sys
        import types

        import mlx.core as mx

        from vllm_metal.attention.impls.varlen_rope_compat import (
            apply_packed_rope,
        )

        captured: list[mx.array] = []

        class FakeMRoPE:
            def rotary_emb(self, x, position_ids):
                captured.append(position_ids)
                # Return cos/sin shaped to match rotary_emb's contract; we
                # do not care about correctness here, only routing.
                seg_len = x.shape[2]
                head_dim = x.shape[3]
                cos = mx.zeros((1, 1, seg_len, head_dim))
                sin = mx.zeros((1, 1, seg_len, head_dim))
                return cos, sin

        # Patch apply_multimodal_rotary_pos_emb so we do not need real mlx_vlm
        # math; just route q,k through unchanged.  ``monkeypatch.setitem``
        # restores the original sys.modules entry at teardown so this fake
        # cannot leak into later tests in the same process.
        fake_mod = types.ModuleType("mlx_vlm.models.qwen3_5.language")
        fake_mod.apply_multimodal_rotary_pos_emb = lambda q, k, cos, sin: (q, k)
        monkeypatch.setitem(sys.modules, "mlx_vlm.models.qwen3_5.language", fake_mod)

        q = mx.zeros((1, 1, 3, 2))
        k = mx.zeros((1, 1, 3, 2))
        provided = mx.array([[[0, 1, 2]], [[0, 1, 2]], [[0, 1, 2]]], dtype=mx.int32)
        apply_packed_rope(
            FakeMRoPE(),
            q,
            k,
            [0, 3],
            positions=[provided],
        )

        assert len(captured) == 1
        assert captured[0] is provided

    def test_mrope_mixed_int_offset_and_array_positions(self, monkeypatch):
        """Each segment independently picks int-offset or array-position."""
        import sys
        import types

        import mlx.core as mx

        from vllm_metal.attention.impls.varlen_rope_compat import (
            apply_packed_rope,
        )

        recorded: list[mx.array] = []

        class FakeMRoPE:
            def rotary_emb(self, x, position_ids):
                recorded.append(position_ids)
                seg_len = x.shape[2]
                head_dim = x.shape[3]
                cos = mx.zeros((1, 1, seg_len, head_dim))
                sin = mx.zeros((1, 1, seg_len, head_dim))
                return cos, sin

        fake_mod = types.ModuleType("mlx_vlm.models.qwen3_5.language")
        fake_mod.apply_multimodal_rotary_pos_emb = lambda q, k, cos, sin: (q, k)
        monkeypatch.setitem(sys.modules, "mlx_vlm.models.qwen3_5.language", fake_mod)

        q = mx.zeros((1, 1, 5, 2))
        k = mx.zeros((1, 1, 5, 2))
        # Two segments: first uses int offset (no positions[0]),
        # second uses array positions[1].
        caller_pos = mx.array([[[10, 11]], [[10, 11]], [[10, 11]]], dtype=mx.int32)
        apply_packed_rope(
            FakeMRoPE(),
            q,
            k,
            [0, 3, 5],
            offsets=[7, 0],
            positions=[None, caller_pos],
        )

        assert len(recorded) == 2
        # First seg: arange(7, 10) broadcast over (3, 1, 3)
        assert recorded[0].shape == (3, 1, 3)
        assert recorded[0].tolist() == [[[7, 8, 9]]] * 3
        # Second seg: caller-supplied positions used verbatim
        assert recorded[1] is caller_pos
