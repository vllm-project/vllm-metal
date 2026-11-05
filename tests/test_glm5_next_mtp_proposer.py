# SPDX-License-Identifier: Apache-2.0
"""Tests for the GLM-5-Next MTP proposer's request-local drafter cache."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_vlm.speculative.drafters.glm5_next_mtp import (
    Glm5NextMTPDraftModel,
    ModelConfig,
)

from vllm_metal.v1.glm5_next_mtp_proposer import Glm5NextMTPProposer
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.spec_decode import PagedDecodeSegment

HIDDEN = 64
VOCAB = 64
K = 3


def _drafter() -> Glm5NextMTPDraftModel:
    """A tiny random drafter whose index pools hold 4 tokens."""
    mx.random.seed(0)
    config = ModelConfig.from_dict(
        {
            "text_config": {
                "model_type": "glm5_next_text",
                "hidden_size": HIDDEN,
                "intermediate_size": 64,
                "moe_intermediate_size": 32,
                "num_hidden_layers": 1,
                "layer_types": ["linear_attention"],
                "mlp_layer_types": ["dense"],
                "indexer_types": ["full"],
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "q_lora_rank": 32,
                "kv_lora_rank": 32,
                "qk_nope_head_dim": 32,
                "qk_head_dim": 32,
                "qk_rope_head_dim": 0,
                "v_head_dim": 32,
                "index_n_heads": 2,
                "index_head_dim": 32,
                "index_topk": 8,
                "index_kpool": 4,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "n_shared_experts": 1,
                "vocab_size": VOCAB,
                "tie_word_embeddings": False,
            }
        }
    )
    drafter = Glm5NextMTPDraftModel(config).eval()
    target = SimpleNamespace(
        embed_tokens=nn.Embedding(VOCAB, HIDDEN),
        lm_head=nn.Linear(HIDDEN, VOCAB, bias=False),
    )
    drafter.bind(target)
    mx.eval(drafter.parameters(), target.embed_tokens.parameters())
    return drafter


class _Controller:
    """Greedy eligibility stand-in: every request may draft."""

    def can_draft_greedy(self, req_id, request_state, sampled_ids) -> bool:
        return True

    def draft_eligible_requests(
        self,
        decode_reqs,
        decode_token_ids,
        prefill_reqs,
        prefill_token_ids,
        prefill_result_modes,
        request_states,
    ):
        eligible = [
            (req_id, state)
            for (req_id, state), sampled in zip(
                decode_reqs, decode_token_ids, strict=True
            )
            if sampled
        ]
        eligible += [
            (prefill.req_id, request_states[prefill.req_id])
            for prefill, mode in zip(prefill_reqs, prefill_result_modes, strict=True)
            if mode != "intermediate"
        ]
        return eligible


def _request_state():
    return SimpleNamespace(
        sampling_params=SimpleNamespace(max_tokens=None), generated_tokens=0
    )


def _prefill_ctx(prefills, rows, *, final=True, draft_caps=None):
    """``prefills``: ``(req_id, chunk, start_pos, sampled_token)`` in order."""
    cu = [0]
    for _, chunk, _, _ in prefills:
        cu.append(cu[-1] + len(chunk))
    return ProposeContext(
        target_hidden_states=rows,
        decode_reqs=[],
        decode_segments=[],
        decode_token_ids=[],
        prefill_reqs=[
            SimpleNamespace(req_id=req_id, token_ids=list(chunk), start_pos=start)
            for req_id, chunk, start, _ in prefills
        ],
        prefill_token_ids=[sampled for *_, sampled in prefills],
        prefill_result_modes=["final" if final else "intermediate"] * len(prefills),
        request_states={req_id: _request_state() for req_id, *_ in prefills},
        cu_seqlens=cu,
        num_decode_segments=0,
        num_speculative_tokens=K,
        finished_req_ids=set(),
        draft_caps=draft_caps,
    )


def _verify_ctx(req_id, last_token, drafts, sampled, start_pos, rows):
    segment = PagedDecodeSegment(
        req_id=req_id,
        input_token_ids=(last_token, *drafts),
        start_row=0,
        num_query_tokens=len(drafts) + 1,
        draft_token_ids=tuple(drafts),
        cache_start_pos=start_pos,
        block_ids=((0,),),
    )
    request_state = _request_state()
    return ProposeContext(
        target_hidden_states=rows,
        decode_reqs=[(req_id, request_state)],
        decode_segments=[segment],
        decode_token_ids=[list(sampled)],
        prefill_reqs=[],
        prefill_token_ids=[],
        prefill_result_modes=[],
        request_states={req_id: request_state},
        cu_seqlens=[0, len(drafts) + 1],
        num_decode_segments=1,
        num_speculative_tokens=K,
        finished_req_ids=set(),
    )


def _drafts(out) -> dict[str, list[int]]:
    return dict(zip(out.req_ids, out.draft_token_ids, strict=True)) if out else {}


@pytest.fixture(scope="module")
def drafter():
    return _drafter()


@pytest.mark.parametrize("accepted", [0, 1, K])
def test_incremental_cache_matches_single_pass(drafter, accepted) -> None:
    """Chunked prefill, drafting past an index pool's end and a verify step
    leave the same drafts and cache as one prefill over the committed tokens,
    whatever the target accepted."""
    mx.random.seed(1)
    prompt = [5, 9, 12, 7]
    hidden = mx.random.normal((len(prompt) + K + 1, HIDDEN))
    first = 17

    incremental = Glm5NextMTPProposer(drafter=drafter, controller=_Controller())
    incremental.propose(
        _prefill_ctx([("r", prompt[:2], 0, 0)], hidden[:2], final=False)
    )
    drafts = _drafts(
        incremental.propose(
            _prefill_ctx([("r", prompt[2:], 2, first)], hidden[2 : len(prompt)])
        )
    )["r"]
    assert len(drafts) == K

    # The target accepts `accepted` drafts, then emits its own token.
    correction = (drafts[accepted] + 1) % VOCAB if accepted < K else 11
    sampled = [*drafts[:accepted], correction]
    verify_rows = hidden[len(prompt) : len(prompt) + K + 1]
    after_verify = _drafts(
        incremental.propose(
            _verify_ctx("r", first, drafts, sampled, len(prompt), verify_rows)
        )
    )["r"]

    committed = [*prompt, first, *drafts[:accepted]]
    single = Glm5NextMTPProposer(drafter=drafter, controller=_Controller())
    expected = _drafts(
        single.propose(
            _prefill_ctx([("r", committed, 0, correction)], hidden[: len(committed)])
        )
    )["r"]

    assert after_verify == expected
    got, want = incremental._states["r"], single._states["r"]
    assert got.next_position == want.next_position
    for a, b in zip(got.cache[0].caches, want.cache[0].caches, strict=True):
        for x, y in zip(a.state, b.state, strict=True):
            assert (x is None) == (y is None)
            if x is not None:
                assert mx.allclose(x, y, atol=1e-5)


def test_batched_drafts_match_single_request_drafts(drafter) -> None:
    """Several requests drafted together (one LM head call per step) give the
    same drafts as each drafted alone, whatever their draft caps."""
    mx.random.seed(2)
    prompts = {"a": [3, 8, 1, 22], "b": [40, 5], "c": [9, 9, 14]}
    sampled = {"a": 6, "b": 13, "c": 50}
    caps = {"a": K, "b": 1, "c": 2}
    hidden = {r: mx.random.normal((len(p), HIDDEN)) for r, p in prompts.items()}

    expected: dict[str, list[int]] = {}
    for req_id, prompt in prompts.items():
        alone = Glm5NextMTPProposer(drafter=drafter, controller=_Controller())
        expected |= _drafts(
            alone.propose(
                _prefill_ctx(
                    [(req_id, prompt, 0, sampled[req_id])],
                    hidden[req_id],
                    draft_caps={req_id: caps[req_id]},
                )
            )
        )

    together = Glm5NextMTPProposer(drafter=drafter, controller=_Controller())
    out = together.propose(
        _prefill_ctx(
            [(r, prompts[r], 0, sampled[r]) for r in prompts],
            mx.concatenate([hidden[r] for r in prompts]),
            draft_caps=caps,
        )
    )
    assert _drafts(out) == expected
    assert {r: len(d) for r, d in expected.items()} == caps
