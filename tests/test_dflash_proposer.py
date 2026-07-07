# SPDX-License-Identifier: Apache-2.0
"""Tests for the DFlash proposer and the aux hidden-state capture.

No engine or real checkpoints: a tiny ``DFlashDraftModel`` (same config as
``test_dflash_model``) plus hand-built ``ProposeContext`` objects drive the
proposer; a tiny mlx_lm qwen3 model exercises the adapter's aux capture
against the stock forward.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from vllm.sampling_params import SamplingParams

from tests.test_dflash_model import ARGS, _build_weights
from vllm_metal.v1.dflash_model import DFlashDraftModel
from vllm_metal.v1.dflash_proposer import DFlashProposer, _ContextKVSlab
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.spec_decode import PagedDecodeSegment, SpeculativeDecodeController

K = 3  # speculative tokens per step in these tests
AUX_DIM = ARGS.target_hidden_size * ARGS.num_aux_layers


def _model() -> DFlashDraftModel:
    weights = _build_weights(ARGS)
    model = DFlashDraftModel(ARGS)
    model.load_weights([(k, mx.array(v)) for k, v in weights.items()], strict=True)
    mx.eval(model.parameters())
    return model


def _proposer() -> DFlashProposer:
    return DFlashProposer(
        model=_model(),
        model_args=ARGS,
        aux_layer_ids=[0, 1],  # unused by these tests (runner-side contract)
        controller=SpeculativeDecodeController(),
    )


def _request_state(token_ids: list[int], *, temperature: float = 0.0):
    return SimpleNamespace(
        token_ids=list(token_ids),
        sampling_params=SamplingParams(temperature=temperature),
        generated_tokens=1,
    )


def _prefill(
    req_id: str,
    token_ids: list[int],
    *,
    start_pos: int = 0,
    temperature: float = 0.0,
):
    return SimpleNamespace(
        req_id=req_id,
        token_ids=list(token_ids),
        block_ids=[0],
        start_pos=start_pos,
        sampling_params=SamplingParams(temperature=temperature),
    )


def _segment(
    req_id: str,
    *,
    start_row: int,
    cache_start_pos: int,
    draft_token_ids: tuple[int, ...] = (),
    last_token: int = 1,
) -> PagedDecodeSegment:
    input_token_ids = (last_token, *draft_token_ids)
    return PagedDecodeSegment(
        req_id=req_id,
        input_token_ids=input_token_ids,
        start_row=start_row,
        num_query_tokens=len(input_token_ids),
        draft_token_ids=draft_token_ids,
        cache_start_pos=cache_start_pos,
        block_ids=(0,),
    )


def _aux(num_rows: int, seed: int = 0) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array((rng.standard_normal((num_rows, AUX_DIM)) * 0.1).astype(np.float32))


def _ctx(
    *,
    aux: mx.array | None,
    decode_reqs=(),
    decode_segments=(),
    decode_token_ids=(),
    prefill_reqs=(),
    prefill_token_ids=(),
    prefill_result_modes=(),
    request_states=None,
    cu_seqlens=(0,),
    num_speculative_tokens: int = K,
) -> ProposeContext:
    if request_states is None:
        request_states = dict(decode_reqs)
        for prefill in prefill_reqs:
            request_states.setdefault(
                prefill.req_id, _request_state(list(prefill.token_ids))
            )
    return ProposeContext(
        target_hidden_states=None,
        target_aux_hidden_states=aux,
        decode_reqs=list(decode_reqs),
        decode_segments=list(decode_segments),
        decode_token_ids=list(decode_token_ids),
        prefill_reqs=list(prefill_reqs),
        prefill_token_ids=list(prefill_token_ids),
        prefill_result_modes=list(prefill_result_modes),
        request_states=request_states,
        cu_seqlens=list(cu_seqlens),
        num_decode_segments=len(decode_segments),
        num_speculative_tokens=num_speculative_tokens,
        logitsprocs=None,
    )


class TestDFlashProposerIngest:
    def test_intermediate_chunk_ingests_without_drafting(self) -> None:
        proposer = _proposer()
        chunk = [5, 6, 7]
        ctx = _ctx(
            aux=_aux(len(chunk)),
            prefill_reqs=[_prefill("r1", chunk)],
            prefill_token_ids=[0],
            prefill_result_modes=["intermediate"],
            request_states={},
            cu_seqlens=[0, len(chunk)],
        )
        assert proposer.propose(ctx) is None
        assert proposer._slabs["r1"].ctx_len == len(chunk)

        # Continuation chunk starts where the first left off.
        chunk2 = [8, 9]
        ctx2 = _ctx(
            aux=_aux(len(chunk2), seed=1),
            prefill_reqs=[_prefill("r1", chunk2, start_pos=len(chunk))],
            prefill_token_ids=[11],
            prefill_result_modes=["new_final"],
            cu_seqlens=[0, len(chunk2)],
        )
        out = proposer.propose(ctx2)
        assert out is not None and out.req_ids == ["r1"]
        assert proposer._slabs["r1"].ctx_len == len(chunk) + len(chunk2)

    def test_decode_ingests_only_accepted_rows(self) -> None:
        # The load-bearing invariant: a verify step ingests the ACCEPTED-token
        # count, not the drafted-token count. Anything else desyncs the context
        # KV from the target and silently corrupts every later draft.
        proposer = _proposer()
        prompt = [5, 6, 7, 8]
        proposer.propose(
            _ctx(
                aux=_aux(len(prompt)),
                prefill_reqs=[_prefill("r1", prompt)],
                prefill_token_ids=[9],
                prefill_result_modes=["new_final"],
                cu_seqlens=[0, len(prompt)],
            )
        )

        # Verify step: rows [bonus, d1, d2, d3]; only 2 outputs sampled
        # (one accepted draft + correction) => ingest exactly 2 rows, so the
        # slab advances 4 -> 6, not 4 -> 7 (the drafted count).
        state = _request_state(prompt + [9, 10, 42])
        segment = _segment(
            "r1",
            start_row=0,
            cache_start_pos=4,
            draft_token_ids=(10, 11, 12),
            last_token=9,
        )
        ctx = _ctx(
            aux=_aux(4, seed=2),
            decode_reqs=[("r1", state)],
            decode_segments=[segment],
            decode_token_ids=[[10, 42]],
            cu_seqlens=[0, 4],
        )
        out = proposer.propose(ctx)
        assert out is not None and out.req_ids == ["r1"]
        assert proposer._slabs["r1"].ctx_len == 6

    def test_non_greedy_requests_are_skipped(self) -> None:
        proposer = _proposer()
        prompt = [5, 6, 7]
        ctx = _ctx(
            aux=_aux(len(prompt)),
            prefill_reqs=[_prefill("r1", prompt, temperature=0.8)],
            prefill_token_ids=[9],
            prefill_result_modes=["new_final"],
            request_states={"r1": _request_state(prompt + [9], temperature=0.8)},
            cu_seqlens=[0, len(prompt)],
        )
        assert proposer.propose(ctx) is None
        assert "r1" not in proposer._slabs

    def test_context_hole_fails_loud(self) -> None:
        """A forward position gap means hidden states were never observed —
        unrepairable, so propose must raise rather than draft wrongly."""
        proposer = _proposer()
        prompt = [5, 6, 7]
        proposer.propose(
            _ctx(
                aux=_aux(len(prompt)),
                prefill_reqs=[_prefill("r1", prompt)],
                prefill_token_ids=[9],
                prefill_result_modes=["new_final"],
                cu_seqlens=[0, len(prompt)],
            )
        )
        # Decode step whose first row position (10) skips ahead of ctx_len (3).
        state = _request_state(prompt + [9, 10])
        segment = _segment("r1", start_row=0, cache_start_pos=10, last_token=9)
        ctx = _ctx(
            aux=_aux(1, seed=3),
            decode_reqs=[("r1", state)],
            decode_segments=[segment],
            decode_token_ids=[[10]],
            cu_seqlens=[0, 1],
        )
        with pytest.raises(RuntimeError, match="hole"):
            proposer.propose(ctx)

    def test_rewind_truncates_and_reingests(self) -> None:
        """A scheduler preemption-recompute re-forwards committed tokens from
        an earlier position; the slab truncates and rebuilds exactly."""
        proposer = _proposer()
        prompt = [5, 6, 7]
        proposer.propose(
            _ctx(
                aux=_aux(len(prompt)),
                prefill_reqs=[_prefill("r1", prompt)],
                prefill_token_ids=[9],
                prefill_result_modes=["new_final"],
                cu_seqlens=[0, len(prompt)],
            )
        )
        # Recompute: the request re-prefills the same span (now 4 tokens) from
        # position 0. The slab must truncate and rebuild to 4 — not append onto
        # the stale 3, which would leave 7.
        out = proposer.propose(
            _ctx(
                aux=_aux(4, seed=5),
                prefill_reqs=[_prefill("r1", prompt + [9], start_pos=0)],
                prefill_token_ids=[10],
                prefill_result_modes=["new_final"],
                cu_seqlens=[0, 4],
            )
        )
        assert out is not None and out.req_ids == ["r1"]
        assert proposer._slabs["r1"].ctx_len == 4


class TestContextKVSlab:
    def test_append_grows_and_preserves(self) -> None:
        slab = _ContextKVSlab()
        k1 = mx.ones((2, 2, 300, 4))
        v1 = mx.full((2, 2, 300, 4), 2.0)
        slab.append(k1, v1)
        assert slab.ctx_len == 300
        k2 = mx.full((2, 2, 10, 4), 3.0)
        slab.append(k2, k2)
        assert slab.ctx_len == 310
        k_view, v_view = slab.view()
        mx.eval(k_view, v_view)
        assert k_view.shape == (2, 2, 310, 4)
        assert float(k_view[0, 0, 0, 0]) == 1.0
        assert float(k_view[0, 0, 305, 0]) == 3.0
        assert float(v_view[0, 0, 299, 0]) == 2.0


class TestAuxCapture:
    def test_adapter_aux_capture_matches_stock_forward(self) -> None:
        from mlx_lm.models import qwen3 as mlx_qwen3

        from vllm_metal.v1.model_adapter import DefaultModelAdapter

        args = mlx_qwen3.ModelArgs(
            model_type="qwen3",
            hidden_size=64,
            num_hidden_layers=4,
            intermediate_size=96,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rope_theta=10000.0,
            head_dim=16,
            tie_word_embeddings=True,
        )
        model = mlx_qwen3.Model(args)
        mx.eval(model.parameters())
        adapter = DefaultModelAdapter()
        input_ids = mx.array([[3, 5, 7, 11, 13]], dtype=mx.int32)

        stock = adapter.target_forward(model, input_ids, collect_hidden_states=True)
        captured = adapter.target_forward(
            model,
            input_ids,
            collect_hidden_states=True,
            aux_layer_ids=[1, 3],
        )
        mx.eval(stock.logits, captured.logits, captured.aux_hidden_states)

        assert captured.aux_hidden_states is not None
        assert captured.aux_hidden_states.shape == (5, 2 * args.hidden_size)
        np.testing.assert_allclose(
            np.array(captured.logits, dtype=np.float32),
            np.array(stock.logits, dtype=np.float32),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_array_equal(
            np.array(captured.hidden_states, dtype=np.float32),
            np.array(stock.hidden_states, dtype=np.float32),
        )

        # The captured slice for layer id i must equal the residual stream
        # after layer i, computed manually.
        backbone = model.model
        from mlx_lm.models.base import create_attention_mask

        h = backbone.embed_tokens(input_ids)
        mask = create_attention_mask(h, None)
        expected = {}
        for i, layer in enumerate(backbone.layers):
            h = layer(h, mask, None)
            if i in (1, 3):
                expected[i] = h
        aux = captured.aux_hidden_states
        np.testing.assert_array_equal(
            np.array(aux[:, : args.hidden_size]),
            np.array(expected[1][0]),
        )
        np.testing.assert_array_equal(
            np.array(aux[:, args.hidden_size :]),
            np.array(expected[3][0]),
        )
