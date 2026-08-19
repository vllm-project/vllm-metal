# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
from vllm.sampling_params import SamplingParams

from vllm_metal.v1.model_adapter import DefaultModelAdapter
from vllm_metal.v1.proposer import ProposeContext, QwenNativeMTPProposer
from vllm_metal.v1.spec_decode import PagedDecodeSegment

VOCAB = 64


class _FakeNativeMTPModel:
    supports_mtp = True

    def mtp_forward(self, hidden, next_token_ids, cache):
        del hidden, cache
        predicted = (next_token_ids.astype(mx.int32) + 1) % VOCAB
        return mx.eye(VOCAB, dtype=mx.float32)[predicted] * 100.0

    def __call__(self, input_ids, *, cache=None, return_hidden=False):
        del cache
        hidden = mx.repeat(input_ids[..., None].astype(mx.float32), 4, axis=-1)
        target = (input_ids.astype(mx.int32) + 5) % VOCAB
        logits = mx.eye(VOCAB, dtype=mx.float32)[target] * 100.0
        if return_hidden:
            return logits, hidden
        return logits


class _FakePagedRuntime:
    qwen_mtp_ready = True

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.boundaries: dict[int, mx.array] = {}
        self.boundary_reads: list[int] = []
        self.batch_calls: list[dict[str, object]] = []
        self.fail_boundary = False

    def supports_hybrid_speculative_decode(self) -> bool:
        return True

    def qwen_mtp_boundary_hidden(self, block_ids_by_group, token_position):
        assert len(block_ids_by_group) == 2
        self.boundary_reads.append(token_position)
        if self.fail_boundary or token_position not in self.boundaries:
            raise RuntimeError("missing boundary")
        return self.boundaries[token_position]

    def qwen_mtp_run_pairs(
        self,
        *,
        hidden_rows,
        next_token_ids,
        block_ids_by_group,
        start_pos,
    ):
        assert len(block_ids_by_group) == 2
        tokens = [int(token) for token in next_token_ids]
        self.calls.append(
            {
                "start_pos": start_pos,
                "tokens": tokens,
                "hidden": [float(row[0].item()) for row in hidden_rows],
                "mtp_blocks": list(block_ids_by_group[1]),
            }
        )
        return (tokens[-1] + 1) % VOCAB

    def qwen_mtp_run_pairs_batch(
        self,
        *,
        hidden_rows_batch,
        next_token_ids_batch,
        block_ids_by_group_batch,
        start_positions,
        draft_request_indices=None,
    ):
        indices = (
            list(range(len(hidden_rows_batch)))
            if draft_request_indices is None
            else list(draft_request_indices)
        )
        self.batch_calls.append(
            {
                "requests": len(hidden_rows_batch),
                "draft_request_indices": indices,
            }
        )
        all_drafts = [
            self.qwen_mtp_run_pairs(
                hidden_rows=hidden_rows,
                next_token_ids=next_token_ids,
                block_ids_by_group=block_ids_by_group,
                start_pos=start_pos,
            )
            for hidden_rows, next_token_ids, block_ids_by_group, start_pos in zip(
                hidden_rows_batch,
                next_token_ids_batch,
                block_ids_by_group_batch,
                start_positions,
                strict=True,
            )
        ]
        return [all_drafts[index] for index in indices]


class _Controller:
    @staticmethod
    def can_draft_greedy(req_id, state):
        del req_id, state
        return True


def _runner(model=None, width=1, runtime=None):
    return SimpleNamespace(
        _forward_model=model or _FakeNativeMTPModel(),
        _paged_attention_runtime=runtime or _FakePagedRuntime(),
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="mtp",
                num_speculative_tokens=width,
            )
        ),
        _spec_decode_controller=_Controller(),
    )


def _ctx(
    *,
    hidden,
    decode_reqs=(),
    decode_segments=(),
    decode_token_ids=(),
    prefill_reqs=(),
    prefill_token_ids=(),
    prefill_result_modes=(),
    request_states=None,
    cu_seqlens=(0,),
    finished_req_ids=None,
):
    return ProposeContext(
        target_hidden_states=hidden,
        decode_reqs=decode_reqs,
        decode_segments=decode_segments,
        decode_token_ids=decode_token_ids,
        prefill_reqs=prefill_reqs,
        prefill_token_ids=prefill_token_ids,
        prefill_result_modes=prefill_result_modes,
        request_states=request_states or {},
        cu_seqlens=cu_seqlens,
        num_decode_segments=len(decode_segments),
        num_speculative_tokens=1,
        finished_req_ids=finished_req_ids or set(),
    )


def _hidden(*token_ids):
    values = mx.array(token_ids, dtype=mx.float32)
    return mx.repeat(values[:, None], 4, axis=-1)


def _prefill(req_id, token_ids, start_pos):
    return SimpleNamespace(
        req_id=req_id,
        token_ids=list(token_ids),
        start_pos=start_pos,
        block_ids=[[2, 3, 4, 5], [20, 21, 22, 23]],
    )


class TestQwenTargetHiddenContract:
    def test_adapter_uses_model_logits_and_pre_norm_hidden(self) -> None:
        model = _FakeNativeMTPModel()
        output = DefaultModelAdapter().target_forward(
            model,
            mx.array([[2, 3]], dtype=mx.int32),
            collect_hidden_states=True,
        )
        assert mx.argmax(output.logits[0, 0]).item() == 7
        assert output.hidden_states is not None
        assert output.hidden_states.shape == (2, 4)
        assert output.hidden_states[0, 0].item() == 2.0


class TestQwenNativeMTPProposerPagedTransaction:
    def test_requires_the_trained_one_token_width(self) -> None:
        with pytest.raises(ValueError, match="exactly one speculative token"):
            QwenNativeMTPProposer(_runner(width=3))

    def test_collects_hidden_states_for_intermediate_prefill(self) -> None:
        proposer = QwenNativeMTPProposer(_runner())
        assert proposer.needs_target_hidden_states([], has_final_prefill=False)

    def test_chunked_prefill_decode_and_release_use_scheduler_positions(self) -> None:
        runtime = _FakePagedRuntime()
        proposer = QwenNativeMTPProposer(_runner(runtime=runtime))
        sampling = SamplingParams(temperature=0)
        state = SimpleNamespace(sampling_params=sampling)

        result = proposer.propose(
            _ctx(
                hidden=_hidden(1, 2, 3),
                prefill_reqs=[_prefill("r0", [1, 2, 3], 0)],
                prefill_token_ids=[0],
                prefill_result_modes=["intermediate"],
                request_states={"r0": state},
                cu_seqlens=[0, 3],
            )
        )
        assert result is None
        assert runtime.calls[-1]["start_pos"] == 0
        assert runtime.calls[-1]["tokens"] == [2, 3]

        result = proposer.propose(
            _ctx(
                hidden=_hidden(4, 5),
                prefill_reqs=[_prefill("r0", [4, 5], 3)],
                prefill_token_ids=[6],
                prefill_result_modes=["new_final"],
                request_states={"r0": state},
                cu_seqlens=[0, 2],
            )
        )
        assert result is not None
        assert result.draft_token_ids == [[7]]
        assert [(c["start_pos"], c["tokens"]) for c in runtime.calls[-2:]] == [
            (2, [4, 5]),
            (4, [6]),
        ]

        segment = PagedDecodeSegment(
            req_id="r0",
            input_token_ids=(6, 7),
            start_row=0,
            num_query_tokens=2,
            draft_token_ids=(7,),
            cache_start_pos=5,
            block_ids=((2, 3, 4, 5), (20, 21, 22, 23)),
        )
        result = proposer.propose(
            _ctx(
                hidden=_hidden(6, 7),
                decode_reqs=[("r0", state)],
                decode_segments=[segment],
                decode_token_ids=[[7, 8]],
                request_states={"r0": state},
                cu_seqlens=[0, 2],
            )
        )
        assert result is not None
        assert result.draft_token_ids == [[9]]
        assert runtime.calls[-1]["start_pos"] == 5
        assert runtime.calls[-1]["tokens"] == [7, 8]

        proposer.release_requests({"r0"})
        assert "r0" not in proposer._states

    def test_scheduler_prefix_hit_resumes_after_eagle_recompute_boundary(self) -> None:
        runtime = _FakePagedRuntime()
        runtime.boundaries[99] = _hidden(99)
        proposer = QwenNativeMTPProposer(_runner(runtime=runtime))
        state = SimpleNamespace(sampling_params=SamplingParams(temperature=0))

        result = proposer.propose(
            _ctx(
                hidden=_hidden(100, 101),
                prefill_reqs=[_prefill("hit", [100, 101], 100)],
                prefill_token_ids=[102],
                prefill_result_modes=["cached_final"],
                request_states={"hit": state},
                cu_seqlens=[0, 2],
            )
        )
        assert result is not None
        assert result.draft_token_ids == [[39]]  # (102 + 1) % 64
        assert runtime.boundary_reads == [99]
        # The retained MTP shared tail is not overwritten at position 99. The
        # EAGLE-dropped suffix recomputes target hidden 100 first, then MTP
        # resumes at logical position 100.
        assert [(c["start_pos"], c["tokens"]) for c in runtime.calls] == [
            (100, [101]),
            (101, [102]),
        ]
        assert runtime.calls[0]["hidden"] == [100.0]
        assert "hit" not in proposer._prefix_hit_blocked

    def test_missing_boundary_fails_closed_without_fresh_mtp_state(self) -> None:
        runtime = _FakePagedRuntime()
        runtime.fail_boundary = True
        proposer = QwenNativeMTPProposer(_runner(runtime=runtime))
        state = SimpleNamespace(sampling_params=SamplingParams(temperature=0))
        result = proposer.propose(
            _ctx(
                hidden=_hidden(100, 101),
                prefill_reqs=[_prefill("hit", [100, 101], 100)],
                prefill_token_ids=[102],
                prefill_result_modes=["cached_final"],
                request_states={"hit": state},
                cu_seqlens=[0, 2],
            )
        )
        assert result is None
        assert runtime.calls == []
        assert "hit" in proposer._prefix_hit_blocked

    def test_decode_requests_share_one_mtp_runtime_batch(self) -> None:
        runtime = _FakePagedRuntime()
        proposer = QwenNativeMTPProposer(_runner(runtime=runtime))
        state0 = SimpleNamespace(sampling_params=SamplingParams(temperature=0))
        state1 = SimpleNamespace(sampling_params=SamplingParams(temperature=0))

        proposer.propose(
            _ctx(
                hidden=_hidden(1, 2, 11, 12),
                prefill_reqs=[
                    _prefill("r0", [1, 2], 0),
                    _prefill("r1", [11, 12], 0),
                ],
                prefill_token_ids=[0, 0],
                prefill_result_modes=["intermediate", "intermediate"],
                request_states={"r0": state0, "r1": state1},
                cu_seqlens=[0, 2, 4],
            )
        )
        assert runtime.batch_calls[-1] == {
            "requests": 2,
            "draft_request_indices": [],
        }

        proposer.propose(
            _ctx(
                hidden=_hidden(3, 13),
                prefill_reqs=[
                    _prefill("r0", [3], 2),
                    _prefill("r1", [13], 2),
                ],
                prefill_token_ids=[4, 14],
                prefill_result_modes=["new_final", "new_final"],
                request_states={"r0": state0, "r1": state1},
                cu_seqlens=[0, 1, 2],
            )
        )

        runtime.batch_calls.clear()
        result = proposer.propose(
            _ctx(
                hidden=_hidden(4, 14),
                decode_reqs=[("r0", state0), ("r1", state1)],
                decode_segments=[
                    PagedDecodeSegment(
                        req_id="r0",
                        input_token_ids=(4,),
                        start_row=0,
                        num_query_tokens=1,
                        draft_token_ids=(),
                        cache_start_pos=3,
                        block_ids=((2, 3, 4, 5), (20, 21, 22, 23)),
                    ),
                    PagedDecodeSegment(
                        req_id="r1",
                        input_token_ids=(14,),
                        start_row=1,
                        num_query_tokens=1,
                        draft_token_ids=(),
                        cache_start_pos=3,
                        block_ids=((6, 7, 8, 9), (24, 25, 26, 27)),
                    ),
                ],
                decode_token_ids=[[5], [15]],
                request_states={"r0": state0, "r1": state1},
                cu_seqlens=[0, 1, 2],
            )
        )
        assert result is not None
        assert result.req_ids == ["r0", "r1"]
        assert result.draft_token_ids == [[6], [16]]
        assert runtime.batch_calls == [{"requests": 2, "draft_request_indices": [0, 1]}]
