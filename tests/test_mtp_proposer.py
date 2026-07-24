# SPDX-License-Identifier: Apache-2.0
"""Tests for the native MTP-head proposer and its per-request KV slab.

No engine or real checkpoints: a tiny stub head model that captures its calls
(and advances the slab through ``update_and_fetch`` like the real head) plus
hand-built ``ProposeContext`` objects drive the proposer. Structure mirrors
``test_dflash_proposer.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
from vllm.sampling_params import SamplingParams

from vllm_metal.v1.mtp_proposer import MTPSlab, NativeMTPProposer
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.spec_decode import PagedDecodeSegment, SpeculativeDecodeController

# Head/slab dims. The slab holds different k/v feature dims (latent vs rope),
# inferred from the first append — never hardcoded in the proposer.
HIDDEN = 8
LATENT = 4
ROPE = 2
VOCAB = 100


class _StubHead:
    """Head model stand-in: passes the last committed token through so the
    draft token is deterministic, and advances the slab exactly like the real
    ``forward_slots`` (which appends the layer's MLA latent as an attention
    side effect)."""

    def __init__(self) -> None:
        # (tokens, first_position, hidden_shape) per build_slot_inputs call.
        self.slot_calls: list[tuple[list[int], int, tuple[int, ...]]] = []
        # (num_slots, expected_offset, offset_before) per forward_slots call.
        self.forward_calls: list[tuple[int, int | None, int]] = []
        # Number of compute_logits calls (one per drafted row).
        self.logits_calls = 0

    def build_slot_inputs(
        self, token_ids: mx.array, hidden_rows: mx.array, first_position: int
    ) -> mx.array:
        toks = [int(t) for t in token_ids.tolist()]
        self.slot_calls.append((toks, first_position, tuple(hidden_rows.shape)))
        n = token_ids.shape[0]
        col0 = token_ids.astype(mx.float32).reshape(n, 1)
        return mx.concatenate([col0, mx.zeros((n, HIDDEN - 1))], axis=1)

    def forward_slots(
        self, x: mx.array, cache: MTPSlab, *, expected_offset: int | None = None
    ) -> mx.array:
        n = x.shape[0]
        offset_before = cache.offset
        if expected_offset is not None and cache.offset != expected_offset:
            raise ValueError(
                f"cache offset {cache.offset} != expected_offset {expected_offset}"
            )
        cache.update_and_fetch(mx.ones((1, 1, n, LATENT)), mx.ones((1, 1, n, ROPE)))
        self.forward_calls.append((n, expected_offset, offset_before))
        return x  # column 0 carries the embedding token id

    def compute_logits(self, hidden: mx.array) -> mx.array:
        self.logits_calls += 1
        col0 = mx.round(hidden[:, 0]).astype(mx.int32) % VOCAB
        return (mx.arange(VOCAB)[None, :] == col0[:, None]).astype(mx.float32)


def _proposer() -> tuple[NativeMTPProposer, _StubHead]:
    head = _StubHead()
    proposer = NativeMTPProposer(
        model=head,
        controller=SpeculativeDecodeController(),
    )
    return proposer, head


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


def _hidden(num_rows: int) -> mx.array:
    # Values are irrelevant to the stub head; distinct rows aid debugging.
    return mx.arange(num_rows * HIDDEN).reshape(num_rows, HIDDEN).astype(mx.float32)


def _ctx(
    *,
    hidden: mx.array | None,
    decode_reqs=(),
    decode_segments=(),
    decode_token_ids=(),
    prefill_reqs=(),
    prefill_token_ids=(),
    prefill_result_modes=(),
    request_states=None,
    cu_seqlens=(0,),
    num_speculative_tokens: int = 1,
) -> ProposeContext:
    if request_states is None:
        request_states = dict(decode_reqs)
        for prefill in prefill_reqs:
            request_states.setdefault(
                prefill.req_id, _request_state(list(prefill.token_ids))
            )
    return ProposeContext(
        target_hidden_states=hidden,
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


class TestNativeMTPProposerIngest:
    def test_intermediate_chunk_ingests_without_drafting(self) -> None:
        proposer, head = _proposer()
        chunk = [5, 6, 7]  # positions 0,1,2
        ctx = _ctx(
            hidden=_hidden(len(chunk)),
            prefill_reqs=[_prefill("r1", chunk)],
            prefill_token_ids=[0],
            prefill_result_modes=["intermediate"],
            request_states={"r1": _request_state(chunk)},
            cu_seqlens=[0, len(chunk)],
        )
        # Intermediate chunk drafts nothing; ingests the ready rows (0,1) and
        # defers the last row (position 2) into pending.
        assert proposer.propose(ctx) is None
        assert proposer._slabs["r1"].offset == 2
        assert "r1" in proposer._pending_last_hidden
        assert proposer._pending_last_hidden["r1"][0] == 2
        # Ready slots embed the within-chunk next tokens at position 0.
        assert head.slot_calls[0] == ([6, 7], 0, (2, HIDDEN))

    def test_deferred_last_slot_consumed_by_next_chunk(self) -> None:
        proposer, head = _proposer()
        chunk1 = [5, 6, 7]  # positions 0,1,2
        proposer.propose(
            _ctx(
                hidden=_hidden(len(chunk1)),
                prefill_reqs=[_prefill("r1", chunk1)],
                prefill_token_ids=[0],
                prefill_result_modes=["intermediate"],
                request_states={"r1": _request_state(chunk1)},
                cu_seqlens=[0, len(chunk1)],
            )
        )
        assert proposer._slabs["r1"].offset == 2

        chunk2 = [8, 9]  # positions 3,4
        out = proposer.propose(
            _ctx(
                hidden=_hidden(len(chunk2)),
                prefill_reqs=[_prefill("r1", chunk2, start_pos=len(chunk1))],
                prefill_token_ids=[11],  # sampled final token
                prefill_result_modes=["new_final"],
                cu_seqlens=[0, len(chunk2)],
            )
        )
        # Pending (position 2) prepended, then chunk2 rows -> contiguous 2,3,4.
        assert out is not None
        assert out.req_ids == ["r1"]
        assert out.draft_token_ids == [[11]]  # last slot embeds sampled token 11
        assert proposer._slabs["r1"].offset == 5
        assert "r1" not in proposer._pending_last_hidden
        # Second forward starts at the pending position and covers 3 slots.
        assert head.forward_calls[-1] == (3, 2, 2)
        assert head.slot_calls[-1] == ([8, 9, 11], 2, (3, HIDDEN))

    def test_decode_ingests_only_accepted_rows(self) -> None:
        # The load-bearing invariant: a verify step ingests the ACCEPTED-token
        # count, not the drafted-token count.
        proposer, head = _proposer()
        prompt = [5, 6, 7, 8]
        proposer.propose(
            _ctx(
                hidden=_hidden(len(prompt)),
                prefill_reqs=[_prefill("r1", prompt)],
                prefill_token_ids=[9],
                prefill_result_modes=["new_final"],
                cu_seqlens=[0, len(prompt)],
            )
        )
        assert proposer._slabs["r1"].offset == 4

        # Verify step: rows [bonus, d1, d2, d3]; 2 outputs sampled (one accepted
        # draft + correction) => ingest exactly 2 rows, slab 4 -> 6 (not 4 -> 7).
        state = _request_state(prompt + [9, 10, 42])
        segment = _segment(
            "r1",
            start_row=0,
            cache_start_pos=4,
            draft_token_ids=(10, 11, 12),
            last_token=9,
        )
        out = proposer.propose(
            _ctx(
                hidden=_hidden(4),
                decode_reqs=[("r1", state)],
                decode_segments=[segment],
                decode_token_ids=[[10, 42]],
                cu_seqlens=[0, 4],
            )
        )
        assert out is not None and out.req_ids == ["r1"]
        assert out.draft_token_ids == [[42]]  # last committed token embedded
        assert proposer._slabs["r1"].offset == 6
        assert head.forward_calls[-1] == (2, 4, 4)

    def test_non_greedy_request_is_skipped_entirely(self) -> None:
        proposer, _ = _proposer()
        prompt = [5, 6, 7]
        ctx = _ctx(
            hidden=_hidden(len(prompt)),
            prefill_reqs=[_prefill("r1", prompt, temperature=0.8)],
            prefill_token_ids=[9],
            prefill_result_modes=["new_final"],
            request_states={"r1": _request_state(prompt + [9], temperature=0.8)},
            cu_seqlens=[0, len(prompt)],
        )
        assert proposer.propose(ctx) is None
        assert "r1" not in proposer._slabs
        assert "r1" not in proposer._pending_last_hidden

    def test_rewind_truncates_and_reingests(self) -> None:
        """A scheduler preemption-recompute re-forwards committed tokens from an
        earlier position; the slab truncates and rebuilds exactly."""
        proposer, _ = _proposer()
        prompt = [5, 6, 7]
        proposer.propose(
            _ctx(
                hidden=_hidden(len(prompt)),
                prefill_reqs=[_prefill("r1", prompt)],
                prefill_token_ids=[9],
                prefill_result_modes=["new_final"],
                cu_seqlens=[0, len(prompt)],
            )
        )
        assert proposer._slabs["r1"].offset == 3

        # Recompute: re-prefill the same span (now 4 tokens) from position 0. The
        # slab must truncate and rebuild to 4 — not append onto the stale 3.
        out = proposer.propose(
            _ctx(
                hidden=_hidden(4),
                prefill_reqs=[_prefill("r1", prompt + [9], start_pos=0)],
                prefill_token_ids=[10],
                prefill_result_modes=["new_final"],
                cu_seqlens=[0, 4],
            )
        )
        assert out is not None and out.req_ids == ["r1"]
        assert proposer._slabs["r1"].offset == 4

    def test_rewind_drops_stale_pending(self) -> None:
        """A recompute that lands mid-prefill drops the deferred (pending) row
        from before the preemption; a fresh intermediate defer replaces it."""
        proposer, _ = _proposer()
        chunk1 = [5, 6, 7]
        proposer.propose(
            _ctx(
                hidden=_hidden(len(chunk1)),
                prefill_reqs=[_prefill("r1", chunk1)],
                prefill_token_ids=[0],
                prefill_result_modes=["intermediate"],
                request_states={"r1": _request_state(chunk1)},
                cu_seqlens=[0, len(chunk1)],
            )
        )
        assert proposer._pending_last_hidden["r1"][0] == 2

        # Recompute from position 0 as a fresh intermediate chunk of 2 tokens.
        recompute = [5, 6]
        assert (
            proposer.propose(
                _ctx(
                    hidden=_hidden(len(recompute)),
                    prefill_reqs=[_prefill("r1", recompute, start_pos=0)],
                    prefill_token_ids=[0],
                    prefill_result_modes=["intermediate"],
                    request_states={"r1": _request_state(recompute)},
                    cu_seqlens=[0, len(recompute)],
                )
            )
            is None
        )
        # Slab rewound to 0 then ingested 1 ready row (position 0); the fresh
        # defer is at position 1, replacing the stale pending at position 2.
        assert proposer._slabs["r1"].offset == 1
        assert proposer._pending_last_hidden["r1"][0] == 1

    def test_hole_fails_loud(self) -> None:
        """A forward position gap means hidden states were never observed —
        unrepairable, so propose must raise rather than draft wrongly."""
        proposer, _ = _proposer()
        prompt = [5, 6, 7]
        proposer.propose(
            _ctx(
                hidden=_hidden(len(prompt)),
                prefill_reqs=[_prefill("r1", prompt)],
                prefill_token_ids=[9],
                prefill_result_modes=["new_final"],
                cu_seqlens=[0, len(prompt)],
            )
        )
        # Decode step whose first row position (10) skips ahead of offset (3).
        state = _request_state(prompt + [9, 10])
        segment = _segment("r1", start_row=0, cache_start_pos=10, last_token=9)
        ctx = _ctx(
            hidden=_hidden(1),
            decode_reqs=[("r1", state)],
            decode_segments=[segment],
            decode_token_ids=[[10]],
            cu_seqlens=[0, 1],
        )
        with pytest.raises(RuntimeError, match="hole"):
            proposer.propose(ctx)

    def test_prune_on_finish_removes_slab_and_pending(self) -> None:
        proposer, _ = _proposer()
        chunk = [5, 6, 7]
        proposer.propose(
            _ctx(
                hidden=_hidden(len(chunk)),
                prefill_reqs=[_prefill("r1", chunk)],
                prefill_token_ids=[0],
                prefill_result_modes=["intermediate"],
                request_states={"r1": _request_state(chunk)},
                cu_seqlens=[0, len(chunk)],
            )
        )
        assert "r1" in proposer._slabs
        assert "r1" in proposer._pending_last_hidden

        # Next step: r1 is gone from request_states (finished/aborted) -> pruned.
        assert proposer.propose(_ctx(hidden=_hidden(1), request_states={})) is None
        assert "r1" not in proposer._slabs
        assert "r1" not in proposer._pending_last_hidden

    def test_release_requests_frees_slab_then_recompute_rebuilds_from_zero(
        self,
    ) -> None:
        """Preemption releases the slab (and any deferred row); a resumed
        recompute rebuilds from position 0 without tripping the hole guard."""
        proposer, _ = _proposer()
        prompt = [5, 6, 7]
        proposer.propose(
            _ctx(
                hidden=_hidden(len(prompt)),
                prefill_reqs=[_prefill("r1", prompt)],
                prefill_token_ids=[0],
                prefill_result_modes=["intermediate"],
                request_states={"r1": _request_state(prompt)},
                cu_seqlens=[0, len(prompt)],
            )
        )
        slab = proposer._slabs["r1"]
        assert slab.offset > 0 and slab.keys is not None
        assert "r1" in proposer._pending_last_hidden

        # Preemption: the runner releases the request's drafter state. The slab
        # (and its k/v arrays) and the deferred row are dropped, not retained.
        proposer.release_requests({"r1"})

        assert "r1" not in proposer._slabs
        assert "r1" not in proposer._pending_last_hidden

        # Resume: recompute re-forwards the committed tokens from position 0. A
        # fresh slab is built at offset 0 and rebuilds to 4 -- the hole guard
        # (first_position > offset) must not fire on the rebuild.
        out = proposer.propose(
            _ctx(
                hidden=_hidden(4),
                prefill_reqs=[_prefill("r1", prompt + [9], start_pos=0)],
                prefill_token_ids=[10],
                prefill_result_modes=["new_final"],
                cu_seqlens=[0, 4],
            )
        )
        assert out is not None and out.req_ids == ["r1"]
        assert proposer._slabs["r1"].offset == 4

    def test_missing_hidden_states_fails_loud(self) -> None:
        proposer, _ = _proposer()
        ctx = _ctx(
            hidden=None,
            prefill_reqs=[_prefill("r1", [5, 6, 7])],
            prefill_token_ids=[9],
            prefill_result_modes=["new_final"],
            cu_seqlens=[0, 3],
        )
        with pytest.raises(RuntimeError, match="target hidden states"):
            proposer.propose(ctx)

    def test_position_zero_zeroing_reaches_build_slot_inputs(self) -> None:
        # first_position must propagate so the head can zero the absolute
        # position-0 embedding. A chunk from position 0 sees first_position 0;
        # a decode segment sees its cache_start_pos.
        proposer, head = _proposer()
        proposer.propose(
            _ctx(
                hidden=_hidden(3),
                prefill_reqs=[_prefill("r1", [5, 6, 7])],
                prefill_token_ids=[9],
                prefill_result_modes=["new_final"],
                cu_seqlens=[0, 3],
            )
        )
        assert head.slot_calls[0][1] == 0

        state = _request_state([5, 6, 7, 9, 20])
        segment = _segment("r1", start_row=0, cache_start_pos=3, last_token=9)
        proposer.propose(
            _ctx(
                hidden=_hidden(1),
                decode_reqs=[("r1", state)],
                decode_segments=[segment],
                decode_token_ids=[[20]],
                cu_seqlens=[0, 1],
            )
        )
        assert head.slot_calls[-1][1] == 3

    def test_zero_spec_tokens_ingests_without_drafting_then_resumes(self) -> None:
        # Zero spec tokens: no drafting, but committed slots still ingest so
        # the slab stays contiguous for when speculation re-enables.
        proposer, head = _proposer()
        prompt = [5, 6, 7, 8]
        proposer.propose(
            _ctx(
                hidden=_hidden(len(prompt)),
                prefill_reqs=[_prefill("r1", prompt)],
                prefill_token_ids=[9],
                prefill_result_modes=["new_final"],
                cu_seqlens=[0, len(prompt)],
            )
        )
        assert proposer._slabs["r1"].offset == 4
        logits_calls_after_seed = head.logits_calls

        # Verify step with 2 committed rows, but speculation disabled this step.
        state = _request_state(prompt + [9, 10, 42])
        segment = _segment(
            "r1",
            start_row=0,
            cache_start_pos=4,
            draft_token_ids=(10, 11, 12),
            last_token=9,
        )
        out = proposer.propose(
            _ctx(
                hidden=_hidden(4),
                decode_reqs=[("r1", state)],
                decode_segments=[segment],
                decode_token_ids=[[10, 42]],
                cu_seqlens=[0, 4],
                num_speculative_tokens=0,
            )
        )
        assert out is None
        assert head.logits_calls == logits_calls_after_seed
        # The 2 committed rows were still ingested: slab 4 -> 6.
        assert proposer._slabs["r1"].offset == 6

        # Speculation re-enabled at the now-contiguous position: drafting
        # resumes without tripping the hole guard.
        state = _request_state(prompt + [9, 10, 42, 43])
        segment = _segment("r1", start_row=0, cache_start_pos=6, last_token=42)
        out = proposer.propose(
            _ctx(
                hidden=_hidden(1),
                decode_reqs=[("r1", state)],
                decode_segments=[segment],
                decode_token_ids=[[43]],
                cu_seqlens=[0, 1],
                num_speculative_tokens=1,
            )
        )
        assert out is not None and out.req_ids == ["r1"]
        assert head.logits_calls == logits_calls_after_seed + 1
        assert proposer._slabs["r1"].offset == 7


class TestMTPSlab:
    def test_dim_inference_and_growth_across_step_boundary(self) -> None:
        slab = MTPSlab()
        # k feature dim 5, v feature dim 3 — inferred, not hardcoded.
        k1 = mx.ones((1, 1, 250, 5))
        v1 = mx.full((1, 1, 250, 3), 2.0)
        fk, fv = slab.update_and_fetch(k1, v1)
        assert slab.offset == 250
        assert fk.shape == (1, 1, 250, 5)
        assert fv.shape == (1, 1, 250, 3)

        # 250 -> 270 crosses the 256 growth boundary; history is preserved.
        k2 = mx.full((1, 1, 20, 5), 3.0)
        v2 = mx.full((1, 1, 20, 3), 4.0)
        fk, fv = slab.update_and_fetch(k2, v2)
        assert slab.offset == 270
        assert fk.shape == (1, 1, 270, 5)
        assert fv.shape == (1, 1, 270, 3)
        mx.eval(fk, fv)
        assert float(fk[0, 0, 0, 0]) == 1.0
        assert float(fk[0, 0, 260, 0]) == 3.0
        assert float(fv[0, 0, 249, 0]) == 2.0
        assert float(fv[0, 0, 269, 0]) == 4.0

    def test_truncate_to_rewinds_offset_and_reingest_overwrites(self) -> None:
        slab = MTPSlab()
        slab.update_and_fetch(mx.ones((1, 1, 10, 5)), mx.ones((1, 1, 10, 3)))
        assert slab.offset == 10

        slab.truncate_to(4)
        assert slab.offset == 4
        fk, _ = slab.update_and_fetch(
            mx.full((1, 1, 2, 5), 7.0), mx.full((1, 1, 2, 3), 7.0)
        )
        assert slab.offset == 6
        assert fk.shape == (1, 1, 6, 5)
        mx.eval(fk)
        # Rows 4,5 were overwritten by the re-ingest.
        assert float(fk[0, 0, 4, 0]) == 7.0

    def test_truncate_to_rejects_out_of_range(self) -> None:
        slab = MTPSlab()
        slab.update_and_fetch(mx.ones((1, 1, 5, 5)), mx.ones((1, 1, 5, 3)))
        with pytest.raises(ValueError, match="non-negative"):
            slab.truncate_to(-1)
        with pytest.raises(ValueError, match="cannot advance"):
            slab.truncate_to(6)


class TestNativeMTPProposerBuild:
    def test_build_loads_and_constructs_over_runtime_model(self) -> None:
        model = object()
        runtime = SimpleNamespace(model=model, model_name="/fake-head")
        calls: list[dict] = []

        class _Loader:
            def load_if_needed(self, *, speculative_config, target_config):
                calls.append(
                    {
                        "speculative_config": speculative_config,
                        "target_config": target_config,
                    }
                )
                return runtime

        target_config = {"hidden_size": 2048}

        proposer = NativeMTPProposer.build(
            speculative_config=SimpleNamespace(),
            controller=SpeculativeDecodeController(),
            loader=_Loader(),
            model_type="glm4_moe_lite_mtp",
            target_config=target_config,
        )

        assert isinstance(proposer, NativeMTPProposer)
        assert proposer._model is model
        assert calls and calls[0]["target_config"] is target_config
