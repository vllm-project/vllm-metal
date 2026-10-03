# SPDX-License-Identifier: Apache-2.0
"""DSpark serving alignment, admission and vLLM runner compatibility."""

from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.config import VllmConfig

from tests.test_block_draft_proposer import _features, _prefill
from tests.test_dspark_paged import make_cache
from vllm_metal.patches.dspark_config import enable_dspark_for_metal_runner
from vllm_metal.v1.dspark_proposer import DSparkProposer
from vllm_metal.v1.model_runner import RequestState
from vllm_metal.v1.spec_decode import SpeculativeDecodeController


@pytest.mark.parametrize("width", [1, 3, 7])
def test_dspark_uses_exactly_k_slots_at_context_and_page_limit(width):
    model, cache = make_cache()
    proposer = DSparkProposer(
        model, num_draft_tokens=7, controller=SpeculativeDecodeController()
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
        mx.array([4]), [f[None] for f in features], num_draft_tokens=width
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
        ("dspark_draft_topk", 16),
        ("checkpoint_topk", 16),
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
    if option == "checkpoint_topk":
        spec.draft_model_config.hf_config.dspark_draft_topk = value
    else:
        setattr(spec, option, value)
    runner = SimpleNamespace(vllm_config=SimpleNamespace(speculative_config=spec))
    with pytest.raises(NotImplementedError, match="greedy drafting"):
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
def test_profile_materializes_dspark_owned_heads(monkeypatch, confidence):
    model, _ = make_cache(confidence=confidence)
    proposer = DSparkProposer(
        model, num_draft_tokens=7, controller=SpeculativeDecodeController()
    )
    draft = model.draft
    observed = []

    def record(anchors, features, *, num_draft_tokens):
        observed.append((anchors.shape, num_draft_tokens))
        return draft(anchors, features, num_draft_tokens=num_draft_tokens)

    monkeypatch.setattr(model, "draft", record)
    proposer.profile([f[None] for f in _features(4)], 2)
    assert observed == [((2,), 7)]
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
            enable_batch_sharded_sampling=True,
        ),
        speculative_config=None
        if method is None
        else SimpleNamespace(method=method, enable_adaptive_verification=True),
        model_config=SimpleNamespace(is_diffusion=True),
        _dflash_needs_multi_kv_group=lambda: True,
        _is_dflash2_draft=lambda: True,
    )
    before = original(config)
    after = first(config)
    expected = [
        item
        for item in before
        if not (
            worker == "vllm_metal.v1.worker.MetalWorker"
            and method == "dspark"
            and item == "dspark speculative decoding"
        )
    ]
    assert after == expected
    assert "prefill context parallel" in after
    if method is not None:
        assert "adaptive draft verification" in after


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
