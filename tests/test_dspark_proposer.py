# SPDX-License-Identifier: Apache-2.0
"""DSpark serving alignment and admission."""

from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from vllm import SamplingParams

from tests.test_block_draft_proposer import _features, _prefill
from tests.test_dspark_paged import make_cache
from tests.test_dspark_quantization import make_model
from vllm_metal.v1 import dspark_proposer
from vllm_metal.v1.dspark_proposer import DSparkProposer
from vllm_metal.v1.model_runner import RequestState
from vllm_metal.v1.spec_decode import SpeculativeDecodeController


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
        ("enable_adaptive_verification", True),
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
