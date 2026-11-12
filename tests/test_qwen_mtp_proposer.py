# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3.5 MTP proposer's request-local drafter cache."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_vlm.speculative.drafters.qwen3_5_mtp.config import Qwen3_5MTPConfig
from mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp import (
    Qwen3_5MTPDraftModel,
)

from vllm_metal.config import QWEN_MTP_DRAFTER_KEY
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.qwen_mtp_proposer import (
    QwenMTPProposer,
    drafter_source,
    is_qwen_mtp,
)
from vllm_metal.v1.spec_decode import PagedDecodeSegment

HIDDEN = 32
VOCAB = 64
K = 3


def _drafter() -> Qwen3_5MTPDraftModel:
    mx.random.seed(0)
    config = Qwen3_5MTPConfig.from_dict(
        {
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": HIDDEN,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "vocab_size": VOCAB,
                "rms_norm_eps": 1e-6,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "max_position_embeddings": 512,
                "tie_word_embeddings": False,
                "mtp_num_hidden_layers": 1,
            }
        }
    )
    drafter = Qwen3_5MTPDraftModel(config)
    target = SimpleNamespace(
        embed_tokens=nn.Embedding(VOCAB, HIDDEN),
        lm_head=nn.Linear(HIDDEN, VOCAB, bias=False),
    )
    drafter.bind(target)
    mx.eval(drafter.parameters(), target.embed_tokens.parameters())
    return drafter


class _Controller:
    """Greedy eligibility stand-in: every request may draft unless listed."""

    def __init__(self, blocked: set[str] | None = None) -> None:
        self.blocked = blocked or set()

    def can_draft_greedy(self, req_id, request_state, sampled_ids) -> bool:
        return req_id not in self.blocked

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
            if sampled and req_id not in self.blocked
        ]
        eligible += [
            (prefill.req_id, request_states[prefill.req_id])
            for prefill, mode in zip(prefill_reqs, prefill_result_modes, strict=True)
            if mode != "intermediate" and prefill.req_id not in self.blocked
        ]
        return eligible


def _request_state(max_tokens=None, generated_tokens=0):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(max_tokens=max_tokens),
        generated_tokens=generated_tokens,
    )


def _prefill_ctx(
    req_id,
    chunk,
    start_pos,
    rows,
    sampled_token,
    *,
    final=True,
    request_state=None,
    draft_caps=None,
    finished=frozenset(),
):
    request_state = request_state or _request_state()
    return ProposeContext(
        target_hidden_states=rows,
        decode_reqs=[],
        decode_segments=[],
        decode_token_ids=[],
        prefill_reqs=[
            SimpleNamespace(req_id=req_id, token_ids=list(chunk), start_pos=start_pos)
        ],
        prefill_token_ids=[sampled_token],
        prefill_result_modes=["final" if final else "intermediate"],
        request_states={req_id: request_state},
        cu_seqlens=[0, len(chunk)],
        num_decode_segments=0,
        num_speculative_tokens=K,
        finished_req_ids=set(finished),
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
    """Chunked prefill + a verify step leave the same cache and drafts as one
    prefill over the committed tokens, whatever the target accepted."""
    mx.random.seed(1)
    prompt = [5, 9, 12, 7, 3, 30, 41, 2]
    hidden = mx.random.normal((len(prompt) + K + 1, HIDDEN))
    first = 17

    incremental = QwenMTPProposer(drafter=drafter, controller=_Controller())
    split = 5
    assert (
        incremental.propose(
            _prefill_ctx("r", prompt[:split], 0, hidden[:split], 0, final=False)
        )
        is None
    )
    out = incremental.propose(
        _prefill_ctx("r", prompt[split:], split, hidden[split : len(prompt)], first)
    )
    drafts = _drafts(out)["r"]
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
    single = QwenMTPProposer(drafter=drafter, controller=_Controller())
    expected = _drafts(
        single.propose(
            _prefill_ctx("r", committed, 0, hidden[: len(committed)], correction)
        )
    )["r"]

    assert after_verify == expected
    for got, want in zip(
        incremental._states["r"].cache, single._states["r"].cache, strict=True
    ):
        assert got.offset == want.offset
        for a, b in zip(got.state, want.state, strict=True):
            assert mx.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize(
    ("draft_caps", "max_tokens", "generated", "expected"),
    [
        ({"r": 1}, None, 0, 1),
        (None, 10, 8, 1),
        ({"r": 2}, 10, 0, 2),
        ({"r": 0}, None, 0, 0),
    ],
)
def test_draft_length_respects_caps(
    drafter, draft_caps, max_tokens, generated, expected
) -> None:
    proposer = QwenMTPProposer(drafter=drafter, controller=_Controller())
    out = proposer.propose(
        _prefill_ctx(
            "r",
            [1, 2, 3],
            0,
            mx.zeros((3, HIDDEN)),
            4,
            request_state=_request_state(max_tokens, generated),
            draft_caps=draft_caps,
        )
    )
    assert len(_drafts(out).get("r", [])) == expected


def test_batched_drafts_match_single_request_drafts(drafter) -> None:
    """Several requests drafted together (shared LM head call per step) give
    the same drafts as each drafted alone, whatever their draft caps."""
    mx.random.seed(2)
    prompts = {"a": [3, 8, 1, 22], "b": [40, 5], "c": [9, 9, 14]}
    sampled = {"a": 6, "b": 13, "c": 50}
    caps = {"a": K, "b": 1, "c": 2}
    hidden = {r: mx.random.normal((len(p), HIDDEN)) for r, p in prompts.items()}

    expected = {}
    for req_id, prompt in prompts.items():
        alone = QwenMTPProposer(drafter=drafter, controller=_Controller())
        expected |= _drafts(
            alone.propose(
                _prefill_ctx(
                    req_id,
                    prompt,
                    0,
                    hidden[req_id],
                    sampled[req_id],
                    draft_caps={req_id: caps[req_id]},
                )
            )
        )

    req_ids = list(prompts)
    cu = [0]
    for req_id in req_ids:
        cu.append(cu[-1] + len(prompts[req_id]))
    together = QwenMTPProposer(drafter=drafter, controller=_Controller())
    out = together.propose(
        ProposeContext(
            target_hidden_states=mx.concatenate([hidden[r] for r in req_ids]),
            decode_reqs=[],
            decode_segments=[],
            decode_token_ids=[],
            prefill_reqs=[
                SimpleNamespace(req_id=r, token_ids=prompts[r], start_pos=0)
                for r in req_ids
            ],
            prefill_token_ids=[sampled[r] for r in req_ids],
            prefill_result_modes=["final"] * len(req_ids),
            request_states={r: _request_state() for r in req_ids},
            cu_seqlens=cu,
            num_decode_segments=0,
            num_speculative_tokens=K,
            finished_req_ids=set(),
            draft_caps=caps,
        )
    )
    assert _drafts(out) == expected
    assert {r: len(d) for r, d in expected.items()} == caps


def test_ineligible_request_keeps_no_drafter_cache(drafter) -> None:
    proposer = QwenMTPProposer(drafter=drafter, controller=_Controller({"r"}))
    out = proposer.propose(_prefill_ctx("r", [1, 2, 3], 0, mx.zeros((3, HIDDEN)), 4))
    assert out is None
    assert "r" not in proposer._states


def test_finished_request_id_starts_fresh(drafter) -> None:
    """vLLM can reuse a finished id in the same step; the new request must
    not inherit the old drafter cache."""
    proposer = QwenMTPProposer(drafter=drafter, controller=_Controller())
    fresh = QwenMTPProposer(drafter=drafter, controller=_Controller())
    proposer.propose(_prefill_ctx("r", [1, 2, 3], 0, mx.zeros((3, HIDDEN)), 4))
    reused = proposer.propose(
        _prefill_ctx("r", [7, 8], 0, mx.zeros((2, HIDDEN)), 9, finished={"r"})
    )
    expected = fresh.propose(_prefill_ctx("r", [7, 8], 0, mx.zeros((2, HIDDEN)), 9))
    assert _drafts(reused) == _drafts(expected)
    assert proposer._states["r"].cache[0].offset == fresh._states["r"].cache[0].offset


def test_is_qwen_mtp_matches_draft_model_type() -> None:
    def config(method, model_type):
        return SimpleNamespace(
            method=method,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type=model_type)
            ),
        )

    assert is_qwen_mtp(config("mtp", "qwen3_5_mtp"))
    assert not is_qwen_mtp(config("mtp", "gemma4_assistant"))
    assert not is_qwen_mtp(config("ngram", "qwen3_5_mtp"))
    assert not is_qwen_mtp(None)


def test_drafter_source_prefers_split_drafter() -> None:
    def vllm_config(additional):
        return SimpleNamespace(
            additional_config=additional,
            speculative_config=SimpleNamespace(
                draft_model_config=SimpleNamespace(model="org/target")
            ),
        )

    assert drafter_source(vllm_config({})) == "org/target"
    assert (
        drafter_source(vllm_config({QWEN_MTP_DRAFTER_KEY: "org/target-MTP-4bit"}))
        == "org/target-MTP-4bit"
    )
