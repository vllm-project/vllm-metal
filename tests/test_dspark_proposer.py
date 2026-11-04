# SPDX-License-Identifier: Apache-2.0
"""DSpark serving alignment, admission and vLLM runner compatibility."""

import itertools
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode

from tests.test_block_draft_proposer import _assert_committed, _features, _prefill
from tests.test_dspark_paged import make_cache
from tests.test_dspark_quantization import make_model
from vllm_metal.patches.dspark_config import enable_dspark_for_metal_runner
from vllm_metal.v1 import dspark_proposer
from vllm_metal.v1.dspark_adaptive import AdaptiveVerificationProfile, VerificationCosts
from vllm_metal.v1.dspark_proposer import DSparkProposer
from vllm_metal.v1.model_runner import RequestState
from vllm_metal.v1.spec_decode import PagedDecodeSegment, SpeculativeDecodeController


@pytest.mark.parametrize("width", [1, 3, 7])
@pytest.mark.parametrize("draft_topk", [None, 8])
@pytest.mark.parametrize("quantized", [False, True])
def test_dspark_uses_exactly_k_slots_at_context_and_page_limit(
    width, draft_topk, quantized
):
    model, cache = make_cache(quantized=quantized)
    proposer = DSparkProposer(
        model,
        num_draft_tokens=7,
        controller=SpeculativeDecodeController(),
        draft_topk=draft_topk,
    )
    proposer.bind_cache(cache.storage, group_index=1, max_model_len=16)
    length = 16 - width
    features = _features(length)
    state = RequestState(
        token_ids=[1] * length + [4],
        prompt_len=length,
        sampling_params=SamplingParams(temperature=0),
        block_ids=[[0], [3]],
    )
    actual = proposer.propose(
        replace(_prefill(state, features, 0, True), num_speculative_tokens=width)
    )
    assert actual is not None and actual.req_ids == ["r"]
    expected = model.draft(
        mx.array([4]),
        [f[None] for f in features],
        num_draft_tokens=width,
        draft_topk=draft_topk,
    )[0]
    assert actual.draft_token_ids == expected.tolist()
    assert len(actual.draft_token_ids[0]) == width
    assert proposer._valid_ends == {"r": length}
    assert torch.all(cache.storage.tensors["target"] == 7)
    for name in proposer.layer_names:
        assert torch.all(torch.isfinite(cache.storage.tensors[name][3]))
        assert torch.all(torch.isnan(cache.storage.tensors[name][4]))


@pytest.mark.parametrize("width", [0, 8, True, 1.5])
def test_dspark_rejects_width_outside_its_trained_block(width):
    model, _ = make_cache()
    with pytest.raises(ValueError, match="trained block_size"):
        DSparkProposer(
            model, num_draft_tokens=width, controller=SpeculativeDecodeController()
        )


@pytest.mark.parametrize(
    "option,value",
    [
        ("draft_sample_method", "probabilistic"),
        ("rejection_sample_method", "block"),
        ("rejection_sample_method", "synthetic"),
    ],
)
def test_unsupported_drafting_options_fail_before_loading(option, value):
    spec = SimpleNamespace(
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        enable_adaptive_verification=False,
        draft_sample_method="greedy",
        rejection_sample_method="standard",
        dspark_draft_topk=None,
    )
    setattr(spec, option, value)
    runner = SimpleNamespace(vllm_config=SimpleNamespace(speculative_config=spec))
    with pytest.raises(NotImplementedError, match="greedy drafting"):
        DSparkProposer.build(runner)


@pytest.mark.parametrize(
    "explicit,checkpoint,expected",
    [
        (None, None, None),
        (None, 8, 8),
        (16, 8, 16),
        (64, None, 64),
        (np.int64(8), None, 8),
        (None, np.int64(8), 8),
    ],
)
@pytest.mark.parametrize("quantized", [False, True])
def test_candidate_limit_resolves_explicit_option_before_checkpoint(
    monkeypatch, explicit, checkpoint, expected, quantized
):
    model = make_model(mx.float16)
    spec = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(vocab_size=64, dspark_draft_topk=checkpoint),
            quantization=None,
            max_model_len=32,
        ),
        enable_adaptive_verification=False,
        draft_sample_method="greedy",
        rejection_sample_method="standard",
        dspark_draft_topk=explicit,
        quantization=None,
        kv_cache_dtype=None,
        num_speculative_tokens=7,
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=spec,
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            additional_config={"dspark_draft_quantization": "q4"} if quantized else {},
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(to_dict=dict)),
        kv_cache_dtype=mx.float16,
        _spec_decode_controller=SpeculativeDecodeController(),
    )
    monkeypatch.setattr(DSparkProposer, "_checkpoint_path", lambda runner: "unused")

    def load(*args, draft_quantization, **kwargs):
        assert draft_quantization == ("q4" if quantized else None)
        if draft_quantization is not None:
            model.quantize_draft_linears()
        return model

    monkeypatch.setattr(dspark_proposer, "load_dspark", load)
    proposer = DSparkProposer.build(runner)
    assert proposer.draft_topk == expected
    assert proposer.max_model_len == 32
    assert isinstance(proposer.draft_model.lm_head, nn.QuantizedLinear) == quantized


@pytest.mark.parametrize("source", ["explicit", "checkpoint"])
@pytest.mark.parametrize("value", [0, -1, 65, True, 1.5])
def test_invalid_candidate_limit_fails_before_checkpoint_loading(source, value):
    spec = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                vocab_size=64,
                dspark_draft_topk=value if source == "checkpoint" else None,
            )
        ),
        enable_adaptive_verification=False,
        draft_sample_method="greedy",
        rejection_sample_method="standard",
        dspark_draft_topk=value if source == "explicit" else None,
    )
    runner = SimpleNamespace(vllm_config=SimpleNamespace(speculative_config=spec))
    with pytest.raises(ValueError, match="draft_topk"):
        DSparkProposer.build(runner)


@pytest.mark.parametrize(
    "quantization,model_quantization,cache_dtype,target_dtype,error",
    [
        ("fp8", None, None, mx.float16, "unquantized draft"),
        (None, "fp8", None, mx.float16, "unquantized draft"),
        (None, None, "fp8", mx.float16, "draft KV"),
        (None, None, "float16", mx.bfloat16, "draft KV"),
        (None, None, "bfloat16", mx.float16, "draft KV"),
        (None, None, None, mx.float16, None),
        (None, None, "auto", mx.bfloat16, None),
        (None, None, "float16", mx.float16, None),
        (None, None, "bfloat16", mx.bfloat16, None),
    ],
)
def test_draft_precision_options_checked_before_loading(
    monkeypatch, quantization, model_quantization, cache_dtype, target_dtype, error
):
    spec = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(), quantization=model_quantization
        ),
        enable_adaptive_verification=False,
        draft_sample_method="greedy",
        rejection_sample_method="standard",
        dspark_draft_topk=None,
        quantization=quantization,
        kv_cache_dtype=cache_dtype,
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(speculative_config=spec),
        kv_cache_dtype=target_dtype,
    )

    class LoadingReachedError(Exception):
        pass

    def checkpoint_path(runner):
        raise LoadingReachedError

    monkeypatch.setattr(DSparkProposer, "_checkpoint_path", checkpoint_path)
    if error is None:
        with pytest.raises(LoadingReachedError):
            DSparkProposer.build(runner)
    else:
        with pytest.raises(NotImplementedError, match=error):
            DSparkProposer.build(runner)


@pytest.mark.parametrize("confidence", [False, True])
@pytest.mark.parametrize("draft_topk", [None, 8])
@pytest.mark.parametrize("quantized", [False, True])
def test_profile_materializes_dspark_owned_heads(
    monkeypatch, confidence, draft_topk, quantized
):
    model, _ = make_cache(confidence=confidence, quantized=quantized)
    proposer = DSparkProposer(
        model,
        num_draft_tokens=7,
        controller=SpeculativeDecodeController(),
        draft_topk=draft_topk,
    )
    draft = model.draft
    observed = []

    def record(anchors, features, *, num_draft_tokens, draft_topk, corrected_logits):
        observed.append((anchors.shape, num_draft_tokens, draft_topk))
        return draft(
            anchors,
            features,
            num_draft_tokens=num_draft_tokens,
            draft_topk=draft_topk,
            corrected_logits=corrected_logits,
        )

    monkeypatch.setattr(model, "draft", record)
    proposer.profile([f[None] for f in _features(4)], 2)
    assert observed == [((2,), 7, draft_topk)]
    assert proposer.kv_specs(16)[proposer.layer_names[0]].dtype == torch.float16


@pytest.mark.parametrize("method", [None, "dflash", "dspark"])
@pytest.mark.parametrize(
    "worker", ["vllm_metal.v1.worker.MetalWorker", "vllm.v1.worker.gpu_worker.Worker"]
)
def test_config_bridge_preserves_other_runner_rejections(monkeypatch, method, worker):
    # Use the real upstream validator; only replace unrelated model probes.
    original = VllmConfig._get_v1_model_runner_unsupported_features
    original = getattr(original, "__wrapped__", original)
    monkeypatch.setattr(
        VllmConfig, "_get_v1_model_runner_unsupported_features", original
    )
    enable_dspark_for_metal_runner()
    first = VllmConfig._get_v1_model_runner_unsupported_features
    enable_dspark_for_metal_runner()
    assert VllmConfig._get_v1_model_runner_unsupported_features is first
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            worker_cls=worker,
            prefill_context_parallel_size=2,
            pipeline_parallel_size=2,
            enable_batch_sharded_sampling=True,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        speculative_config=None
        if method is None
        else SimpleNamespace(
            method=method,
            enable_adaptive_verification=True,
            is_xpress=lambda: False,
            draft_lm_head_quantization=None,
            draft_model_config=None,
        ),
        model_config=SimpleNamespace(
            is_diffusion=True, architecture="LlamaForCausalLM"
        ),
        _dflash_needs_multi_kv_group=lambda: True,
        _is_dflash_candidate_draft=lambda: True,
    )
    before = original(config)
    after = first(config)
    expected = [
        item
        for item in before
        if not (
            worker == "vllm_metal.v1.worker.MetalWorker"
            and method == "dspark"
            and item in ("dspark speculative decoding", "adaptive draft verification")
        )
    ]
    assert after == expected
    assert "prefill context parallel" in after
    if method is not None and not (
        method == "dspark" and worker == "vllm_metal.v1.worker.MetalWorker"
    ):
        assert "adaptive draft verification" in after


def test_adaptive_prefix_budget_matches_exhaustive_expected_throughput():
    """Search past local rate dips, enforce capacities, and preserve prefixes."""
    costs = VerificationCosts(
        128, (1.0, 2.0), ((8.0, 8.1, 8.2, 8.3), (6.0, 20.0, 20.0, 6.5, 6.6, 8.0, 20.0))
    )
    profile = AdaptiveVerificationProfile((1.0, 1.0, 1.0), (costs,))
    logits = [[2.0, 0.0, -1.0], [1.0, 3.0, -2.0]]
    for capacities in itertools.product(range(4), repeat=2):
        admitted = profile.admit(logits, capacities, 128)

        def score(lengths):
            expected = 2.0
            for row, length in zip(logits, lengths, strict=True):
                survival = 1.0
                for value in row[:length]:
                    survival *= 1 / (1 + math.exp(-value))
                    expected += survival
            return expected / (2 + costs.verify_ms[1][sum(lengths)])

        exhaustive = max(
            score(lengths)
            for lengths in itertools.product(*(range(c + 1) for c in capacities))
        )
        assert score(admitted) == pytest.approx(exhaustive)
        assert all(
            0 <= value <= cap for value, cap in zip(admitted, capacities, strict=True)
        )


def test_adaptive_budget_uses_calibration_and_interpolated_context_cost():
    cheap = VerificationCosts(0, (1.0,), ((1.0, 1.0, 1.0),))
    costly = VerificationCosts(100, (1.0,), ((1.0, 5.0, 5.0),))
    profile = AdaptiveVerificationProfile((1.0, 1.0), (cheap, costly))
    assert profile.admit([[0.0, 0.0]], [2], 0) == [2]
    assert profile.admit([[0.0, 0.0]], [2], 100) == [0]
    assert profile.admit([[0.0, 0.0]], [2], 50) == [0]
    costs = VerificationCosts(0, (1.0,), ((1.0, 1.5, 3.0),))
    confident = AdaptiveVerificationProfile((1.0, 1.0), (costs,))
    calibrated = replace(confident, temperatures=(0.1, 0.1))
    assert confident.admit([[-1.0, -1.0]], [2], 0) == [1]
    assert calibrated.admit([[-1.0, -1.0]], [2], 0) == [0]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_adaptive_rejects_nonfinite_confidence(bad):
    profile = AdaptiveVerificationProfile(
        (1.0,), (VerificationCosts(0, (1.0,), ((1.0, 1.0),)),)
    )
    with pytest.raises(ValueError, match="finite"):
        profile.admit([[bad]], [1], 0)


def test_adaptive_profile_rejects_stale_identity_and_incomplete_costs(tmp_path):
    data = {
        "version": 1,
        "identity": {"target": "revision-1"},
        "temperatures": [1.0, 2.0],
        "costs": [{"context": 0, "draft_ms": [1.0], "verify_ms": [[1.0, 2.0, 3.0]]}],
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data))

    def load(identity):
        return AdaptiveVerificationProfile.load(
            str(path), identity=identity, width=2, max_num_seqs=1
        )

    assert load(data["identity"]).temperatures == (1.0, 2.0)
    with pytest.raises(ValueError, match="does not match"):
        load({"target": "revision-2"})
    data["costs"][0]["verify_ms"][0].pop()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="finite positive"):
        load(data["identity"])


@pytest.mark.parametrize("method", ["dspark", "dflash"])
@pytest.mark.parametrize(
    "worker", ["vllm_metal.v1.worker.MetalWorker", "vllm.v1.worker.gpu_worker.Worker"]
)
def test_only_native_dspark_can_adapt_without_cuda_graphs(monkeypatch, method, worker):
    original = VllmConfig._validate_adaptive_verification
    original = getattr(original, "__wrapped__", original)
    monkeypatch.setattr(VllmConfig, "_validate_adaptive_verification", original)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=method, enable_adaptive_verification=True
        ),
        parallel_config=SimpleNamespace(worker_cls=worker, pipeline_parallel_size=1),
        lora_config=None,
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
    )
    with pytest.raises(ValueError, match="full CUDA graphs"):
        original(config)
    enable_dspark_for_metal_runner()
    installed = VllmConfig._validate_adaptive_verification
    enable_dspark_for_metal_runner()
    assert VllmConfig._validate_adaptive_verification is installed
    if method == "dspark" and worker == "vllm_metal.v1.worker.MetalWorker":
        installed(config)
        config.lora_config = object()
        with pytest.raises(ValueError, match="LoRA"):
            installed(config)
        config.lora_config = None
        config.parallel_config.pipeline_parallel_size = 2
        with pytest.raises(ValueError, match="pipeline parallelism"):
            installed(config)
    else:
        with pytest.raises(ValueError, match="full CUDA graphs"):
            installed(config)


def test_adaptive_proposer_returns_prefix_of_full_block_at_short_output_budget():
    model, cache = make_cache()
    model.confidence_head.proj.weight = mx.zeros_like(model.confidence_head.proj.weight)
    model.confidence_head.proj.bias = mx.zeros_like(model.confidence_head.proj.bias)
    costs = VerificationCosts(
        0, (1.0,), ((1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0, 100.0),)
    )
    profile = AdaptiveVerificationProfile((1.0,) * 7, (costs,))
    proposer = DSparkProposer(
        model,
        num_draft_tokens=7,
        controller=SpeculativeDecodeController(),
        adaptive_profile=profile,
    )
    proposer.bind_cache(cache.storage, group_index=1, max_model_len=64)
    features = _features(4)
    state = RequestState(
        token_ids=[1] * 4 + [4],
        prompt_len=4,
        sampling_params=SamplingParams(temperature=0, max_tokens=3),
        generated_tokens=1,
        block_ids=[[0], [3]],
    )
    actual = proposer.propose(_prefill(state, features, 0, True))
    expected = model.draft(
        mx.array([4]), [f[None] for f in features], num_draft_tokens=7
    )[0].tolist()[0]
    assert actual.draft_token_ids == [expected[:1]]
    assert proposer._valid_ends == {"r": 4}
    assert torch.all(cache.storage.tensors["target"] == 7)


def test_zero_admission_commits_target_context_before_drafting_resumes():
    model, cache = make_cache()
    expensive = VerificationCosts(0, (1.0,), ((1.0,) + (100.0,) * 7,))
    proposer = DSparkProposer(
        model,
        num_draft_tokens=7,
        controller=SpeculativeDecodeController(),
        adaptive_profile=AdaptiveVerificationProfile((1.0,) * 7, (expensive,)),
    )
    proposer.bind_cache(cache.storage, group_index=1, max_model_len=64)
    state = RequestState(
        token_ids=[1] * 4 + [4],
        prompt_len=4,
        sampling_params=SamplingParams(temperature=0),
        block_ids=[[0], [3]],
    )
    prefix = _features(4)
    ctx = _prefill(state, prefix, 0, True)
    assert proposer.propose(ctx).draft_token_ids == [[]]
    state.token_ids.append(5)
    next_features = _features(1)
    ctx = replace(
        ctx,
        target_aux_hidden_states=next_features,
        decode_reqs=[("r", state)],
        decode_segments=[PagedDecodeSegment("r", (4,), 0, 1, (), 4, ((0,), (3,)))],
        decode_token_ids=[[5]],
        prefill_reqs=[],
        prefill_token_ids=[],
        prefill_result_modes=[],
        cu_seqlens=[0, 1],
        num_decode_segments=1,
    )
    # The next cost decision can admit drafts again after a target-only step.
    proposer.adaptive_profile = AdaptiveVerificationProfile(
        (1.0,) * 7, (VerificationCosts(0, (1.0,), ((1.0,) * 8,)),)
    )
    actual = proposer.propose(ctx)
    all_features = [
        mx.concatenate([old, new])
        for old, new in zip(prefix, next_features, strict=True)
    ]
    expected = model.draft(
        mx.array([5]), [f[None] for f in all_features], num_draft_tokens=7
    )[0].tolist()[0]
    assert actual.draft_token_ids == [expected[:3]]
    assert proposer._valid_ends == {"r": 5}
    _assert_committed(proposer, state.block_ids[1], all_features)


def test_dspark_width_rejection_does_not_write_shared_storage():
    model, cache = make_cache()
    proposer = DSparkProposer(
        model, num_draft_tokens=7, controller=SpeculativeDecodeController()
    )
    proposer.bind_cache(cache.storage, group_index=1, max_model_len=64)
    state = RequestState(
        token_ids=[1, 4],
        prompt_len=1,
        sampling_params=SamplingParams(temperature=0),
        block_ids=[[0], [3]],
    )
    before = [np.array(buffer) for buffer in cache.storage.buffers]
    with pytest.raises(ValueError, match="configured token budget"):
        proposer.propose(
            replace(_prefill(state, _features(1), 0, True), num_speculative_tokens=8)
        )
    assert not proposer._valid_ends
    for buffer, expected in zip(cache.storage.buffers, before, strict=True):
        np.testing.assert_array_equal(np.array(buffer), expected)
