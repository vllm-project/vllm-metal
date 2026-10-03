# SPDX-License-Identifier: Apache-2.0
"""Committed feature ownership is independent of temporary draft acceptance."""

from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from tests.test_dflash_paged import make_cache
from tests.test_dspark_paged import make_cache as make_dspark_cache
from vllm_metal.v1.dflash_proposer import DFlashProposer
from vllm_metal.v1.dspark_proposer import DSparkProposer
from vllm_metal.v1.model_runner import PrefillRequest, RequestState
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.spec_decode import PagedDecodeSegment, SpeculativeDecodeController


@pytest.fixture(params=["dflash", "dspark"])
def proposer(request):
    if request.param == "dspark":
        model, cache = make_dspark_cache()
        proposer = DSparkProposer(
            model, num_draft_tokens=3, controller=SpeculativeDecodeController()
        )
    else:
        model, embed, cache = make_cache()
        proposer = DFlashProposer(
            model,
            num_draft_tokens=3,
            embed=embed,
            project=embed.as_linear,
            controller=SpeculativeDecodeController(),
        )
    proposer.bind_cache(cache.storage, group_index=1, max_model_len=64)
    return proposer


def _dense_tokens(proposer, anchors, features, width):
    if isinstance(proposer, DSparkProposer):
        return proposer.draft_model.draft(anchors, features, num_draft_tokens=width)[0]
    return mx.argmax(
        proposer.model.draft_logits(
            anchors,
            features,
            num_draft_tokens=width,
            embed=proposer.embed,
            project=proposer.project,
        ),
        axis=-1,
    )


def _features(count):
    return tuple(mx.random.normal((count, 64)).astype(mx.float16) for _ in range(3))


def _prefill(state, features, start, final):
    count = features[0].shape[0]
    return ProposeContext(
        target_hidden_states=None,
        target_aux_hidden_states=features,
        decode_reqs=[],
        decode_segments=[],
        decode_token_ids=[],
        prefill_reqs=[
            PrefillRequest(
                req_id="r",
                token_ids=[1] * count,
                sampling_params=state.sampling_params,
                block_ids=state.block_ids,
                generator=None,
                prompt_len=None,
                full_prompt_token_ids=None,
                start_pos=start,
            )
        ],
        prefill_token_ids=[2],
        prefill_result_modes=["final" if final else "intermediate"],
        request_states={"r": state},
        cu_seqlens=[0, count],
        num_decode_segments=0,
        num_speculative_tokens=3,
        finished_req_ids=set(),
    )


@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_chunked_prefill_then_rejection_overwrites_temporary_kv(accepted, proposer):
    state = RequestState(
        token_ids=[1] * 15 + [2],
        prompt_len=15,
        sampling_params=SamplingParams(temperature=0),
        block_ids=[[0], [5, 2, 7, 1]],
    )
    first, second = _features(8), _features(7)
    assert proposer.propose(_prefill(state, first, 0, False)) is None
    drafts = proposer.propose(_prefill(state, second, 8, True))
    assert drafts is not None
    tokens = drafts.draft_token_ids[0]
    sampled = tokens[:accepted] + [4]
    state.token_ids.extend(sampled)
    feature = _features(4)
    ctx = ProposeContext(
        target_hidden_states=None,
        target_aux_hidden_states=feature,
        decode_reqs=[("r", state)],
        decode_segments=[
            PagedDecodeSegment(
                req_id="r",
                input_token_ids=(2, *tokens),
                start_row=0,
                num_query_tokens=4,
                draft_token_ids=tuple(tokens),
                cache_start_pos=15,
                block_ids=tuple(tuple(g) for g in state.block_ids),
            )
        ],
        decode_token_ids=[sampled],
        prefill_reqs=[],
        prefill_token_ids=[],
        prefill_result_modes=[],
        request_states={"r": state},
        cu_seqlens=[0, 4],
        num_decode_segments=1,
        num_speculative_tokens=3,
        finished_req_ids=set(),
    )
    actual = proposer.propose(ctx)
    features = [
        mx.concatenate([a, b, c[: accepted + 1]])[None]
        for a, b, c in zip(first, second, feature, strict=True)
    ]
    expected = _dense_tokens(proposer, mx.array([4]), features, 3)
    assert actual.draft_token_ids == expected.tolist()
    assert proposer._valid_ends == {"r": 16 + accepted}
    # Assert the committed KV itself, not just argmax IDs: random residual
    # logits can retain an argmax even if rejected feature rows leak in.
    for i, (keys, values) in enumerate(proposer.model._project_context(features)):
        for stored, reference in (
            (proposer.cache.cache.key_caches[i], keys),
            (proposer.cache.cache.value_caches[i], values),
        ):
            committed = mx.stack(
                [
                    stored[state.block_ids[1][p // 16], p % 16]
                    for p in range(16 + accepted)
                ]
            )
            np.testing.assert_allclose(
                np.array(committed),
                np.array(reference[0].transpose(1, 0, 2)),
                atol=0.004,
                rtol=0.004,
            )
    # Cancellation/preemption invalidates logical coverage, even if the same
    # request ID and physical pages are immediately handed out again.
    proposer.release_requests({"r"})
    assert not proposer._valid_ends
    state.token_ids = [1] * 15 + [2]
    assert proposer.propose(_prefill(state, _features(15), 0, True)) is not None


@pytest.mark.parametrize(
    "kind", ["sampled", "logprobs", "grammar", "zero_k", "context_limit"]
)
def test_non_drafting_rows_still_commit_features(kind, proposer):
    params = SamplingParams(
        temperature=0.7 if kind == "sampled" else 0,
        logprobs=1 if kind == "logprobs" else None,
        structured_outputs=StructuredOutputsParams(choice=["yes", "no"])
        if kind == "grammar"
        else None,
    )
    state = RequestState(
        token_ids=[1] * 15 + [2],
        prompt_len=15,
        sampling_params=params,
        block_ids=[[0], [5, 2, 7, 1]],
    )
    ctx = _prefill(state, _features(15), 0, True)
    if kind == "zero_k":
        ctx = replace(ctx, num_speculative_tokens=0)
    if kind == "context_limit":
        proposer.cache.max_model_len = 17
    assert proposer.propose(ctx) is None
    assert proposer._valid_ends == {"r": 15}


def test_missing_and_discontinuous_features_fail_before_drafting(proposer):
    state = RequestState(
        token_ids=[1, 2],
        prompt_len=1,
        sampling_params=SamplingParams(temperature=0),
        block_ids=[[0], [5]],
    )
    ctx = _prefill(state, _features(1), 0, True)
    with pytest.raises(RuntimeError, match="every packed input row"):
        proposer.propose(replace(ctx, target_aux_hidden_states=()))
    with pytest.raises(RuntimeError, match="discontinuous"):
        proposer.propose(_prefill(state, _features(1), 1, True))


@pytest.mark.parametrize("proposer_cls", [DFlashProposer, DSparkProposer])
@pytest.mark.parametrize("restriction", ["prefix", "lora", "tp", "block_size"])
def test_unsupported_configuration_fails_before_loading(restriction, proposer_cls):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(), quantization=None
            ),
            enable_adaptive_verification=False,
            draft_sample_method="greedy",
            rejection_sample_method="standard",
            dspark_draft_topk=None,
            quantization=None,
            kv_cache_dtype=None,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False, block_size=16),
        lora_config=None,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
    )
    runner = SimpleNamespace(
        vllm_config=config,
        is_hybrid=False,
        is_mla=False,
        _is_vlm=False,
        _is_pooling=False,
    )
    if restriction == "prefix":
        config.cache_config.enable_prefix_caching = True
    elif restriction == "lora":
        config.lora_config = object()
    elif restriction == "tp":
        config.parallel_config.tensor_parallel_size = 2
    else:
        config.cache_config.block_size = 64
    with pytest.raises(NotImplementedError):
        proposer_cls.build(runner)


def test_width_changes_commit_verified_rows_and_reuse_compiled_callables(
    monkeypatch, proposer
):
    state = RequestState(
        token_ids=[1] * 15 + [2],
        prompt_len=15,
        sampling_params=SamplingParams(temperature=0),
        block_ids=[[0], [5, 2, 7, 1]],
    )
    compiled_widths = []
    compile_draft = proposer.cache.compile_draft

    def compile_width(**kwargs):
        compiled_widths.append(kwargs["num_draft_tokens"])
        return compile_draft(**kwargs)

    monkeypatch.setattr(proposer.cache, "compile_draft", compile_width)
    committed = _features(15)
    result = proposer.propose(_prefill(state, committed, 0, True))
    # Incoming verification and outgoing proposal widths are independent.
    # Consecutive K=0 steps still grow committed context before K grows again.
    for step, width in enumerate((1, 0, 0, 3, 1, 3)):
        previous = result.draft_token_ids[0] if result is not None else []
        accepted = min(step % 4, len(previous))
        sampled = previous[:accepted] + [4]
        start = len(state.token_ids) - 1
        segment = PagedDecodeSegment(
            req_id="r",
            input_token_ids=(state.token_ids[-1], *previous),
            start_row=0,
            num_query_tokens=1 + len(previous),
            draft_token_ids=tuple(previous),
            cache_start_pos=start,
            block_ids=tuple(tuple(g) for g in state.block_ids),
        )
        state.token_ids.extend(sampled)
        feature = _features(segment.num_query_tokens)
        result = proposer.propose(
            replace(
                _prefill(state, feature, 0, False),
                decode_reqs=[("r", state)],
                decode_segments=[segment],
                decode_token_ids=[sampled],
                prefill_reqs=[],
                prefill_token_ids=[],
                prefill_result_modes=[],
                num_decode_segments=1,
                num_speculative_tokens=width,
            )
        )
        committed = tuple(
            mx.concatenate([old, new[: len(sampled)]])
            for old, new in zip(committed, feature, strict=True)
        )
        end = len(state.token_ids) - 1
        assert proposer._valid_ends == {"r": end}
        full_features = [f[None] for f in committed]
        # Inspect actual KV: argmax agreement alone can hide a bad ingest.
        for layer, (keys, values) in enumerate(
            proposer.model._project_context(full_features)
        ):
            for stored, expected in (
                (proposer.cache.cache.key_caches[layer], keys),
                (proposer.cache.cache.value_caches[layer], values),
            ):
                actual = mx.stack(
                    [stored[state.block_ids[1][p // 16], p % 16] for p in range(end)]
                )
                np.testing.assert_allclose(
                    np.array(actual),
                    np.array(expected[0].transpose(1, 0, 2)),
                    atol=0.004,
                    rtol=0.004,
                )
        if width == 0:
            assert result is None
        else:
            expected = _dense_tokens(proposer, mx.array([4]), full_features, width)
            assert result.draft_token_ids == expected.tolist()
    assert compiled_widths == [3, 1]
    # Rebinding storage must discard closures over the previous allocation.
    proposer.bind_cache(proposer.cache.storage, group_index=1, max_model_len=64)
    assert not proposer._drafts and not proposer._valid_ends


@pytest.mark.parametrize("width", [-1, 4, True, 1.5])
def test_invalid_width_fails_before_committing_features(width, proposer):
    state = RequestState(
        token_ids=[1, 2],
        prompt_len=1,
        sampling_params=SamplingParams(temperature=0),
        block_ids=[[0], [5]],
    )
    ctx = replace(_prefill(state, _features(1), 0, True), num_speculative_tokens=width)
    before = [np.array(buffer) for buffer in proposer.cache.storage.buffers]
    with pytest.raises(ValueError, match="configured token budget"):
        proposer.propose(ctx)
    assert not proposer._valid_ends and not proposer._drafts
    for actual, expected in zip(proposer.cache.storage.buffers, before, strict=True):
        np.testing.assert_array_equal(np.array(actual), expected)


@pytest.mark.parametrize("width", [1, 3])
def test_context_limit_uses_selected_width(width, proposer):
    proposer.cache.max_model_len = 17
    state = RequestState(
        token_ids=[1] * 15 + [2],
        prompt_len=15,
        sampling_params=SamplingParams(temperature=0),
        block_ids=[[0], [5, 2]],
    )
    result = proposer.propose(
        replace(_prefill(state, _features(15), 0, True), num_speculative_tokens=width)
    )
    assert proposer._valid_ends == {"r": 15}
    if width == 1:
        assert len(result.draft_token_ids[0]) == 1
    else:
        assert result is None
