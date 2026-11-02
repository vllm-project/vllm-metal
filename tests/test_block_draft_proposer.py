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


def _assert_committed(proposer, blocks, features):
    """Compare committed draft KV with independent full-context projections."""
    for layer, (keys, values) in enumerate(
        proposer.model._project_context([f[None] for f in features])
    ):
        for stored, expected in (
            (proposer.cache.cache.key_caches[layer], keys),
            (proposer.cache.cache.value_caches[layer], values),
        ):
            block_size = stored.shape[1]
            actual = mx.stack(
                [
                    stored[blocks[p // block_size], p % block_size]
                    for p in range(len(features[0]))
                ]
            )
            np.testing.assert_allclose(
                np.array(actual),
                np.array(expected[0].transpose(1, 0, 2)),
                atol=0.004,
                rtol=0.004,
            )


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
        mx.concatenate([a, b, c[: accepted + 1]])
        for a, b, c in zip(first, second, feature, strict=True)
    ]
    expected = _dense_tokens(proposer, mx.array([4]), [f[None] for f in features], 3)
    assert actual.draft_token_ids == expected.tolist()
    assert proposer._valid_ends == {"r": 16 + accepted}
    # Assert the committed KV itself, not just argmax IDs: random residual
    # logits can retain an argmax even if rejected feature rows leak in.
    _assert_committed(proposer, state.block_ids[1], features)
    # Cancellation/preemption invalidates logical coverage, even if the same
    # request ID and physical pages are immediately handed out again.
    proposer.release_requests({"r"})
    assert not proposer._valid_ends
    state.token_ids = [1] * 15 + [2]
    assert proposer.propose(_prefill(state, _features(15), 0, True)) is not None


@pytest.mark.parametrize(
    "kind",
    [
        "sampled",
        "logprobs",
        "grammar",
        "zero_k",
        "context_limit",
        "output_budget",
        "last_output",
        "eos",
        "stop_token",
    ],
)
def test_non_drafting_rows_still_commit_features(kind, proposer):
    params = SamplingParams(
        temperature=0.7 if kind == "sampled" else 0,
        max_tokens={"output_budget": 1, "last_output": 2}.get(kind, 16),
        logprobs=1 if kind == "logprobs" else None,
        stop_token_ids=[2] if kind == "stop_token" else None,
        structured_outputs=StructuredOutputsParams(choice=["yes", "no"])
        if kind == "grammar"
        else None,
    )
    if kind == "eos":
        params.update_from_generation_config({}, eos_token_id=2)
    state = RequestState(
        token_ids=[1] * 15 + [2],
        prompt_len=15,
        sampling_params=params,
        generated_tokens=1,
        block_ids=[[0], [5, 2, 7, 1]],
    )
    ctx = _prefill(state, _features(15), 0, True)
    if kind == "zero_k":
        ctx = replace(ctx, num_speculative_tokens=0)
    if kind == "context_limit":
        proposer.cache.max_model_len = 17
    assert proposer.propose(ctx) is None
    assert not proposer._drafts
    assert proposer._valid_ends == {"r": 15}
    _assert_committed(proposer, state.block_ids[1], ctx.target_aux_hidden_states)


@pytest.mark.parametrize("max_tokens", [4, 2])
def test_exhausted_output_budget_commits_features_without_drafting(
    proposer, monkeypatch, max_tokens
):
    state = RequestState(
        token_ids=[1] * 15 + [2],
        prompt_len=15,
        sampling_params=SamplingParams(temperature=0, max_tokens=max_tokens),
        generated_tokens=1,
        block_ids=[[0], [5, 2, 7, 1]],
    )
    prefix = _features(15)
    ctx = _prefill(state, prefix, 0, True)
    result = proposer.propose(ctx)
    assert (result is not None) == (max_tokens > 2)
    state.token_ids.extend([3] * 3)
    state.generated_tokens = 4
    # The final verification can emit more tokens than the output budget;
    # the scheduler will truncate them after this worker step.
    tail = _features(3)
    ctx = replace(
        ctx,
        target_aux_hidden_states=tail,
        decode_reqs=[("r", state)],
        decode_segments=[
            PagedDecodeSegment(
                req_id="r",
                input_token_ids=(2, 3, 3),
                start_row=0,
                num_query_tokens=3,
                draft_token_ids=(3, 3),
                cache_start_pos=15,
                block_ids=tuple(tuple(group) for group in state.block_ids),
            )
        ],
        decode_token_ids=[[3] * 3],
        prefill_reqs=[],
        prefill_token_ids=[],
        prefill_result_modes=[],
        cu_seqlens=[0, 3],
        num_decode_segments=1,
    )
    features = [mx.concatenate([a, b]) for a, b in zip(prefix, tail, strict=True)]

    def unexpected_draft(*args):
        pytest.fail("An exhausted request must not execute another draft forward")

    monkeypatch.setattr(proposer, "_compile_draft", unexpected_draft)
    monkeypatch.setitem(proposer._drafts, 3, unexpected_draft)
    assert proposer.propose(ctx) is None
    assert proposer._valid_ends == {"r": len(features[0])}
    _assert_committed(proposer, state.block_ids[1], features)


@pytest.mark.parametrize("max_tokens", [5, None])
@pytest.mark.parametrize("stop_token", [False, True])
def test_exhausted_row_does_not_suppress_live_drafts(
    proposer, monkeypatch, max_tokens, stop_token
):
    finished = RequestState(
        token_ids=[1] * 15 + [2],
        prompt_len=15,
        sampling_params=SamplingParams(
            temperature=0,
            max_tokens=16 if stop_token else 1,
            stop_token_ids=[2] if stop_token else None,
        ),
        generated_tokens=1,
        block_ids=[[0], [5, 2, 7, 1]],
    )
    live = replace(
        finished,
        sampling_params=SamplingParams(temperature=0, max_tokens=max_tokens),
        block_ids=[[0], [9, 3, 8, 4]],
    )
    features = _features(30)
    ctx = _prefill(finished, [f[:15] for f in features], 0, True)
    ctx = replace(
        ctx,
        target_aux_hidden_states=features,
        prefill_reqs=[
            ctx.prefill_reqs[0],
            ctx.prefill_reqs[0]._replace(
                req_id="live",
                sampling_params=live.sampling_params,
                block_ids=live.block_ids,
            ),
        ],
        prefill_token_ids=[2, 2],
        prefill_result_modes=["final", "final"],
        request_states={"r": finished, "live": live},
        cu_seqlens=[0, 15, 30],
    )
    forward = proposer._compile_draft(3)
    calls = []

    def record_forward(anchors, rows):
        calls.append((anchors.tolist(), rows))
        return forward(anchors, rows)

    monkeypatch.setitem(proposer._drafts, 3, record_forward)
    result = proposer.propose(ctx)
    assert result is not None and result.req_ids == ["live"]
    assert calls == [([2], [(live.block_ids[1], 15)])]
    expected = _dense_tokens(
        proposer, mx.array([2]), [f[None, 15:] for f in features], 3
    )
    assert result.draft_token_ids == expected.tolist()
    assert proposer._valid_ends == {"r": 15, "live": 15}
    _assert_committed(proposer, finished.block_ids[1], [f[:15] for f in features])
    _assert_committed(proposer, live.block_ids[1], [f[15:] for f in features])


@pytest.mark.parametrize(
    "max_tokens,expected_width", [(None, 3), (1, 0), (2, 0), (3, 1), (4, 2), (8, 3)]
)
def test_remaining_output_budget_bounds_draft_forward(
    proposer, monkeypatch, max_tokens, expected_width
):
    state = RequestState(
        token_ids=[1] * 15 + [2],
        prompt_len=15,
        sampling_params=SamplingParams(temperature=0, max_tokens=max_tokens),
        generated_tokens=1,
        block_ids=[[0], [5, 2, 7, 1]],
    )
    features = _features(15)
    widths = []
    compile_draft = proposer._compile_draft

    def record_compile(width):
        widths.append(width)
        return compile_draft(width)

    monkeypatch.setattr(proposer, "_compile_draft", record_compile)
    result = proposer.propose(_prefill(state, features, 0, True))
    _assert_committed(proposer, state.block_ids[1], features)
    if not expected_width:
        assert result is None and not widths
        return
    assert widths == [expected_width]
    expected = _dense_tokens(
        proposer, mx.array([2]), [f[None] for f in features], expected_width
    )
    assert result.req_ids == ["r"]
    assert result.draft_token_ids == expected.tolist()


@pytest.mark.parametrize("long_budget", [4, 8])
def test_mixed_output_budgets_share_one_bounded_draft_batch(
    proposer, monkeypatch, long_budget
):
    states = {
        name: RequestState(
            token_ids=[1] * 15 + [anchor],
            prompt_len=15,
            sampling_params=SamplingParams(temperature=0, max_tokens=budget),
            generated_tokens=1,
            block_ids=[[0], [2 * i, 2 * i + 1]],
        )
        for i, (name, anchor, budget) in enumerate(
            [("r", 2, 2), ("short", 3, 3), ("long", 4, long_budget)]
        )
    }
    features = _features(45)
    ctx = _prefill(states["r"], [f[:15] for f in features], 0, True)
    ctx = replace(
        ctx,
        target_aux_hidden_states=features,
        prefill_reqs=[
            ctx.prefill_reqs[0]._replace(
                req_id=name,
                sampling_params=state.sampling_params,
                block_ids=state.block_ids,
            )
            for name, state in states.items()
        ],
        prefill_token_ids=[2, 3, 4],
        prefill_result_modes=["final"] * 3,
        request_states=states,
        cu_seqlens=[0, 15, 30, 45],
    )
    width = 2 if long_budget == 4 else 3
    calls = []
    compile_draft = proposer._compile_draft

    def record_compile(selected_width):
        forward = compile_draft(selected_width)

        def record_forward(anchors, rows):
            calls.append((selected_width, anchors.tolist(), rows))
            return forward(anchors, rows)

        return record_forward

    monkeypatch.setattr(proposer, "_compile_draft", record_compile)
    result = proposer.propose(ctx)
    assert result.req_ids == ["short", "long"]
    assert calls == [(width, [3, 4], [([2, 3], 15), ([4, 5], 15)])]
    expected = _dense_tokens(
        proposer, mx.array([3, 4]), [f[15:].reshape(2, 15, 64) for f in features], width
    ).tolist()
    assert result.draft_token_ids == [expected[0][:1], expected[1]]
    for i, state in enumerate(states.values()):
        _assert_committed(
            proposer, state.block_ids[1], [f[15 * i : 15 * (i + 1)] for f in features]
        )


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


def test_prefix_caching_cannot_adopt_unknown_decode_coverage(proposer):
    proposer.enable_prefix_caching = True
    state = RequestState(
        token_ids=[1] * 17 + [2],
        prompt_len=16,
        sampling_params=SamplingParams(temperature=0),
        block_ids=[[0, 1], [5, 2]],
    )
    ctx = replace(
        _prefill(state, _features(1), 0, True),
        decode_reqs=[("r", state)],
        decode_segments=[
            PagedDecodeSegment(
                req_id="r",
                input_token_ids=(1,),
                start_row=0,
                num_query_tokens=1,
                draft_token_ids=(),
                cache_start_pos=16,
                block_ids=tuple(tuple(g) for g in state.block_ids),
            )
        ],
        decode_token_ids=[[2]],
        prefill_reqs=[],
        prefill_token_ids=[],
        prefill_result_modes=[],
        num_decode_segments=1,
    )
    with pytest.raises(RuntimeError, match="discontinuous"):
        proposer.propose(ctx)
    assert not proposer._valid_ends


@pytest.mark.parametrize("start", [0, 16])
def test_first_prefill_draft_uses_sampled_anchor_at_absolute_position(
    proposer, monkeypatch, start
):
    proposer.enable_prefix_caching = True
    state = RequestState(
        token_ids=[1] * 31 + [2],
        prompt_len=31,
        sampling_params=SamplingParams(temperature=0),
        block_ids=[[0, 1], [5, 2, 7, 1]],
    )
    features = _features(31)
    if start:
        proposer.cache.write_context(
            [f[:start] for f in features], [(state.block_ids[1], 0, start)]
        )
    compile_draft = proposer._compile_draft
    calls = []

    def record_compile(width):
        forward = compile_draft(width)

        def record_forward(anchors, rows):
            calls.append((anchors.tolist(), rows))
            return forward(anchors, rows)

        return record_forward

    monkeypatch.setattr(proposer, "_compile_draft", record_compile)
    result = proposer.propose(
        _prefill(state, [f[start:] for f in features], start, True)
    )
    assert result is not None
    # The final prompt token is 1; the target has already emitted anchor 2 at
    # position 31. A cached suffix must preserve that absolute position and
    # must not draft from the final prompt token or ingest anchor 2 twice.
    assert calls == [([2], [(state.block_ids[1], 31)])]


@pytest.mark.parametrize("prefix_caching", [False, True])
def test_build_forwards_prefix_caching(monkeypatch, proposer, prefix_caching):
    spec = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(), quantization=None, max_model_len=64
        ),
        num_speculative_tokens=3,
        enable_adaptive_verification=False,
        draft_sample_method="greedy",
        rejection_sample_method="standard",
        dspark_draft_topk=None,
        quantization=None,
        kv_cache_dtype=None,
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=spec,
            cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
            additional_config={},
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(to_dict=dict)),
        kv_cache_dtype=mx.float16,
        _spec_decode_controller=SpeculativeDecodeController(),
    )
    if isinstance(proposer, DSparkProposer):
        model = proposer.draft_model
    else:
        model = proposer.model
        runner._target_input_embeddings = proposer.embed
        runner._forward_model = SimpleNamespace(
            args=SimpleNamespace(tie_word_embeddings=True),
            model=SimpleNamespace(embed_tokens=proposer.embed),
        )
    method = proposer.name.lower()
    proposer_cls = type(proposer)
    # Replace checkpoint I/O only; exercise the real builder and constructor.
    monkeypatch.setattr(proposer_cls, "_checkpoint_path", lambda _: "unused")
    monkeypatch.setattr(
        f"vllm_metal.v1.{method}_proposer.load_{method}", lambda *a, **kw: model
    )
    built = proposer_cls.build(runner)
    built.bind_cache(proposer.cache.storage, group_index=1, max_model_len=64)
    state = RequestState(
        token_ids=[1] * 31 + [2],
        prompt_len=31,
        sampling_params=SamplingParams(temperature=0),
        block_ids=[[0, 1], [5, 2, 7, 1]],
    )
    features = _features(31)
    built.cache.write_context([f[:16] for f in features], [(state.block_ids[1], 0, 16)])
    ctx = _prefill(state, [f[16:] for f in features], 16, True)
    if prefix_caching:
        assert built.propose(ctx) is not None
        _assert_committed(built, state.block_ids[1], features)
    else:
        with pytest.raises(RuntimeError, match="discontinuous"):
            built.propose(ctx)


@pytest.mark.parametrize("proposer_cls", [DFlashProposer, DSparkProposer])
@pytest.mark.parametrize("restriction", ["lora", "tp", "block_size"])
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
    if restriction == "lora":
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
        _assert_committed(proposer, state.block_ids[1], committed)
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


@pytest.mark.parametrize("producer", ["greedy", "sampled", "grammar", "zero_k"])
def test_prefix_hit_reuses_committed_kv_without_writing_shared_blocks(
    proposer, producer
):
    proposer.enable_prefix_caching = True
    params = SamplingParams(
        temperature=0.7 if producer == "sampled" else 0,
        structured_outputs=StructuredOutputsParams(choice=["yes", "no"])
        if producer == "grammar"
        else None,
    )
    state = RequestState(
        token_ids=[1] * 31 + [2],
        prompt_len=31,
        sampling_params=params,
        block_ids=[[0], [5, 2, 7, 1]],
    )
    original = _features(31)
    ctx = _prefill(state, original, 0, True)
    if producer == "zero_k":
        ctx = replace(ctx, num_speculative_tokens=0)
    result = proposer.propose(ctx)
    assert (result is not None) == (producer == "greedy")
    shared = [
        np.array(stored[5])
        for caches in (
            proposer.cache.cache.key_caches,
            proposer.cache.cache.value_caches,
        )
        for stored in caches
    ]
    proposer.release_requests({"r"})
    # Reuse one full block and allocate private suffix/lookahead pages.
    state.sampling_params = SamplingParams(temperature=0)
    state.block_ids = [[0], [5, 3, 8, 4]]
    state.token_ids = [1] * 16 + [3] * 7 + [4]
    state.prompt_len = 23
    suffix = _features(7)
    result = proposer.propose(_prefill(state, suffix, 16, True))
    assert result is not None
    features = [
        mx.concatenate([prefix[:16], tail])
        for prefix, tail in zip(original, suffix, strict=True)
    ]
    assert (
        result.draft_token_ids
        == _dense_tokens(
            proposer, mx.array([4]), [f[None] for f in features], 3
        ).tolist()
    )
    for saved, stored in zip(
        shared,
        [
            stored
            for caches in (
                proposer.cache.cache.key_caches,
                proposer.cache.cache.value_caches,
            )
            for stored in caches
        ],
        strict=True,
    ):
        np.testing.assert_array_equal(np.array(stored[5]), saved)
    _assert_committed(proposer, state.block_ids[1], features)


def test_prefix_hit_does_not_allow_a_later_gap(proposer):
    proposer.enable_prefix_caching = True
    state = RequestState(
        token_ids=[1] * 40,
        prompt_len=40,
        sampling_params=SamplingParams(temperature=0),
        block_ids=[[0], [5, 2, 7, 1]],
    )
    assert proposer.propose(_prefill(state, _features(4), 16, False)) is None
    before = [np.array(buffer) for buffer in proposer.cache.storage.buffers]
    with pytest.raises(RuntimeError, match="discontinuous"):
        proposer.propose(_prefill(state, _features(4), 24, False))
    for actual, expected in zip(proposer.cache.storage.buffers, before, strict=True):
        np.testing.assert_array_equal(np.array(actual), expected)


@pytest.mark.parametrize("start", [32, 48])
def test_prefix_hit_beyond_draft_context_stays_target_only(proposer, start):
    proposer.enable_prefix_caching = True
    proposer.cache.max_model_len = 32
    state = RequestState(
        token_ids=[1] * (start + 1) + [2],
        prompt_len=start + 1,
        sampling_params=SamplingParams(temperature=0),
        block_ids=[[0], [5, 2, 7, 1]],
    )
    before = [np.array(buffer) for buffer in proposer.cache.storage.buffers]
    assert proposer.propose(_prefill(state, _features(1), start, True)) is None
    assert not proposer._valid_ends
    for actual, expected in zip(proposer.cache.storage.buffers, before, strict=True):
        np.testing.assert_array_equal(np.array(actual), expected)
