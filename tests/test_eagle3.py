# SPDX-License-Identifier: Apache-2.0
"""EAGLE3 numerical reference and prefix/cache lifecycle regression tests."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache
from vllm import SamplingParams
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from tests.stub_runner import make_cache_config, make_stub_runner
from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.v1.eagle3 import Eagle3Config, Eagle3Model
from vllm_metal.v1.eagle3_proposer import Eagle3Proposer
from vllm_metal.v1.model_runner import PrefillRequest
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.spec_decode import PagedDecodeSegment, SpeculativeDecodeController

TARGET = {
    "model_type": "qwen3",
    "hidden_size": 64,
    "num_hidden_layers": 8,
    "vocab_size": 128,
}


def _model(kv_heads=2, norm_before_residual=True):
    config = Eagle3Config.from_dict(
        {
            "speculators_model_type": "eagle3",
            "draft_vocab_size": 64,
            "norm_before_residual": norm_before_residual,
            "transformer_layer_config": {
                "model_type": "llama",
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": kv_heads,
                "head_dim": 64,
                "rms_norm_eps": 1e-6,
                "vocab_size": 128,
                "rope_theta": 10000.0,
                "tie_word_embeddings": False,
            },
        },
        TARGET,
    )
    model = Eagle3Model(config)
    model.d2t = mx.arange(64, dtype=mx.int32)
    mx.eval(model.parameters())
    return model


def _proposer(model, *, max_model_len=4096):
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=model.config.layer.num_key_value_heads,
        head_size=model.config.layer.head_dim,
        dtype=torch.float32,
    )
    config = SimpleNamespace(
        cache_config=make_cache_config(block_size=16),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )
    groups = get_kv_cache_groups(config, {"draft_layers.0.self_attn": spec})
    cache_config = get_kv_cache_config_from_groups(
        config, groups, 16 * spec.page_size_bytes
    )
    cache_config.kv_cache_layout = "LBNHC"
    proposer = Eagle3Proposer(
        model=model,
        controller=SpeculativeDecodeController(),
        storage=KVCacheStorage(cache_config),
        max_model_len=max_model_len,
    )
    proposer.adopt_scheduler_group(0, max_model_len)
    return proposer


@pytest.mark.parametrize(("target_kv_heads", "draft_group"), [(1, 0), (2, 0), (1, 1)])
def test_eagle3_shares_target_storage_and_scheduler_group(
    monkeypatch, target_kv_heads, draft_group
):
    from vllm_metal.config import MetalConfig
    from vllm_metal.v1.draft_model_proposer import DraftDims

    runner = make_stub_runner(
        model=_model(target_kv_heads),
        num_layers=1,
        num_kv_cache_layers=1,
        num_kv_heads=target_kv_heads,
        head_dim=64,
        kv_cache_dtype=mx.float32,
        cache_config=make_cache_config(block_size=16, num_gpu_blocks_override=7),
        _eagle3_model=_model(),
        _draft_dims=DraftDims(num_layers=1, num_kv_heads=2, head_dim=64),
    )
    config = runner.vllm_config
    config.speculative_config = SimpleNamespace(
        method="eagle3", draft_model_config=SimpleNamespace(max_model_len=128)
    )
    config.scheduler_config = SimpleNamespace(disable_hybrid_kv_cache_manager=False)
    runner.scheduler_config = config.scheduler_config
    monkeypatch.setattr(
        "vllm_metal.v1.cache_policy.get_config",
        lambda: MetalConfig(mlx_device="gpu", turboquant=False),
    )
    specs = runner.get_kv_cache_spec()
    groups = get_kv_cache_groups(config, specs)
    if draft_group:
        # Separate scheduler groups can alias a region with padded page sizes.
        target_spec = replace(
            specs["layers.0.self_attn"],
            page_size_padded=specs["draft_layers.0.self_attn"].page_size_bytes,
        )
        groups = [
            KVCacheGroupSpec(["layers.0.self_attn"], target_spec),
            KVCacheGroupSpec(
                ["draft_layers.0.self_attn"], specs["draft_layers.0.self_attn"]
            ),
        ]
    final_config = get_kv_cache_config_from_groups(config, groups, 1 << 20)
    final_config.kv_cache_layout = "LBNHC"
    runner.model_config.max_model_len = 48
    runner.initialize_kv_cache(final_config)
    runtime, draft = runner.paged_attention_runtime, runner._drafter
    owner = runtime.cache_storage
    assert runtime.kv_cache.key_caches.storage is draft._kv.key_caches.storage is owner
    assert draft._storage is owner and draft._max_model_len == 48
    assert draft._scheduler_group_index == draft_group
    assert draft._kv.group_index_for_layer(0) == 0  # Local to the draft view.

    owner.tensors["layers.0.self_attn"][1].fill_(42)
    prompt, features = list(range(20)), mx.random.normal((19, 192))
    result = _prefill(draft, [("r", prompt, features, [2, 3], 0, 19)], k=3)
    assert result.draft_token_ids == [
        _plain_drafts(runner._eagle3_model, prompt, features)
    ]
    assert torch.all(owner.tensors["layers.0.self_attn"][1] == 42)


def _prefill(proposer, requests, *, k=3, prompt_len=None):
    """requests = (id, full prompt, feature rows, cache blocks, start, end)."""
    prefills, sampled, modes, feature_rows = [], [], [], []
    states = {}
    cu = [0]
    for req_id, prompt, features, blocks, start, end in requests:
        params = SamplingParams(temperature=0)
        intermediate = end < len(prompt) - 1
        prefills.append(
            PrefillRequest(
                req_id=req_id,
                token_ids=prompt[start:end],
                sampling_params=params,
                block_ids=[[] for _ in range(proposer._scheduler_group_index)]
                + [blocks],
                generator=None,
                prompt_len=None if intermediate else end,
                start_pos=start,
                full_prompt_token_ids=prompt[:prompt_len]
                if prompt_len is not None
                else prompt[:-1],
            )
        )
        sampled.append(-1 if intermediate else prompt[end])
        modes.append("intermediate" if intermediate else "new_final")
        feature_rows.append(features[start:end])
        cu.append(cu[-1] + end - start)
        states[req_id] = SimpleNamespace(token_ids=prompt, sampling_params=params)
    return proposer.propose(
        ProposeContext(
            target_hidden_states=None,
            target_aux_hidden_states=tuple(
                mx.split(mx.concatenate(feature_rows), 3, axis=-1)
            ),
            decode_reqs=[],
            decode_segments=[],
            decode_token_ids=[],
            prefill_reqs=prefills,
            prefill_token_ids=sampled,
            prefill_result_modes=modes,
            request_states=states,
            cu_seqlens=cu,
            num_decode_segments=0,
            num_speculative_tokens=k,
            finished_req_ids=set(),
        )
    )


def _plain_drafts(model, prompt, features, k=3, cache=None):
    cache = KVCache() if cache is None else cache
    normalized, hidden = model(
        mx.array(prompt[1:])[None],
        model.combine_hidden_states(features)[None],
        mask="causal",
        cache=cache,
    )
    columns = [model.top_tokens(normalized[:, -1])]
    hidden = hidden[:, -1:]
    for _ in range(k - 1):
        normalized, hidden = model(columns[-1][:, None], hidden, cache=cache)
        columns.append(model.top_tokens(normalized[:, 0]))
    return mx.stack(columns, axis=1).tolist()[0]


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_ragged_batch_matches_independent_full_context_drafts(k, kv_heads, monkeypatch):
    from vllm_metal.attention.context import get_context
    from vllm_metal.attention.impls import sdpa_wrapper

    native_forward = sdpa_wrapper.sdpa_forward
    packed_rows = []

    def record_forward(inner, x, ctx, *args, **kwargs):
        assert x.shape[0] == 1
        assert x.shape[1] == len(ctx.slot_mapping)
        packed_rows.append(x.shape[1])
        return native_forward(inner, x, ctx, *args, **kwargs)

    monkeypatch.setattr(sdpa_wrapper, "sdpa_forward", record_forward)
    model = _model(kv_heads)
    proposer = _proposer(model)
    prompts = [list(range(35)), list(range(21, 43))]
    features = [mx.random.normal((len(p) - 1, 192)) for p in prompts]
    result = _prefill(
        proposer,
        [
            ("a", prompts[0], features[0], [1, 2, 3], 0, 34),
            ("b", prompts[1], features[1], [4, 5], 0, 21),
        ],
        k=k,
    )
    assert result.draft_token_ids == [
        _plain_drafts(model, p, f, k) for p, f in zip(prompts, features, strict=False)
    ]
    assert packed_rows == [55] + [2] * (k - 1)
    assert get_context() is None


@pytest.mark.parametrize("target_end", [29, 30, 32, 35])
def test_context_limit_bounds_ingest_and_only_suppresses_long_rows(
    target_end, monkeypatch
):
    from vllm_metal.attention.impls import sdpa_wrapper

    native_forward = sdpa_wrapper.sdpa_forward
    context_ends = []

    def record_forward(inner, x, ctx, *args, **kwargs):
        context_ends.extend(ctx.context_lens)
        return native_forward(inner, x, ctx, *args, **kwargs)

    monkeypatch.setattr(sdpa_wrapper, "sdpa_forward", record_forward)
    model = _model()
    proposer = _proposer(model, max_model_len=32)
    short = list(range(8))
    short_features = mx.random.normal((7, 192))
    long = list(range(target_end + 1))
    long_features = mx.random.normal((target_end, 192))
    result = _prefill(
        proposer,
        [
            ("short", short, short_features, [1], 0, 7),
            ("long", long, long_features, [2, 3, 4], 0, target_end),
        ],
    )
    # K=3 fits exactly at 29+3=32. Extra physical block slack must never
    # permit queries beyond the draft model's own context limit.
    assert result.req_ids == (["short", "long"] if target_end == 29 else ["short"])
    assert result.draft_token_ids[0] == _plain_drafts(model, short, short_features)
    assert max(context_ends) == (31 if target_end == 29 else min(target_end, 32))


def test_prefill_beyond_shorter_draft_context_skips_forward(monkeypatch):
    model = _model()
    proposer = _proposer(model, max_model_len=32)
    prompt = list(range(40))
    features = mx.random.normal((39, 192))

    def unexpected_forward(*args):
        pytest.fail("No draft input remains within the model context limit")

    monkeypatch.setattr(proposer, "_forward", unexpected_forward)
    assert _prefill(proposer, [("long", prompt, features, [1, 2, 3], 32, 39)]) is None


@pytest.mark.parametrize("reverse", [False, True])
def test_prefix_created_in_the_same_batch_uses_paged_kv(reverse):
    model = _model()
    proposer = _proposer(model)
    first = list(range(35))
    first_features = mx.random.normal((34, 192))
    second = first[:32] + [87, 92, 12, 41, 56, 22, 75]
    second_features = mx.concatenate((first_features[:32], mx.random.normal((6, 192))))
    requests = [
        ("owner", first, first_features, [1, 2, 3], 0, 34),
        ("hit", second, second_features, [1, 4, 5], 16, 38),
    ]
    result = _prefill(proposer, requests[::-1] if reverse else requests)
    reference_cache = KVCache()
    expected = _plain_drafts(model, second, second_features, cache=reference_cache)
    # The owner populates the shared first block in this same packed forward.
    # Both request orders must see those writes before paged attention reads it.
    for pool, reference in (
        (proposer._kv.key_caches[0], reference_cache.keys),
        (proposer._kv.value_caches[0], reference_cache.values),
    ):
        np.testing.assert_allclose(
            np.array(pool[4, 0]), np.array(reference[0, :, 16]), atol=3e-3, rtol=3e-3
        )
    assert result.draft_token_ids[0 if reverse else 1] == expected


def test_chunked_prefill_and_zero_budget_populate_reusable_prefix():
    model = _model()
    proposer = _proposer(model)
    prompt = list(range(51))
    features = mx.random.normal((50, 192))
    assert _prefill(proposer, [("a", prompt, features, [1], 0, 16)], k=0) is None
    assert _prefill(proposer, [("a", prompt, features, [1, 2], 16, 32)], k=0) is None
    result = _prefill(proposer, [("a", prompt, features, [1, 2, 3, 4], 32, 50)])
    expected = _plain_drafts(model, prompt, features)
    assert result.draft_token_ids == [expected]
    result = _prefill(proposer, [("b", prompt, features, [1, 2, 5, 6], 32, 50)])
    assert result.draft_token_ids == [expected]


@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_verification_rebuilds_kv_from_actual_target_features(accepted):
    model = _model()
    proposer = _proposer(model)
    prompt = list(range(32))
    features = mx.random.normal((31, 192))
    previous = _prefill(proposer, [("a", prompt, features, [1, 2, 3], 0, 31)])
    drafts = previous.draft_token_ids[0]
    output = drafts[:accepted] + [79]
    verified_features = mx.random.normal((4, 192))
    committed = prompt + output
    params = SamplingParams(temperature=0)
    state = SimpleNamespace(
        token_ids=committed, block_ids=[[1, 2, 3]], sampling_params=params
    )
    segment = PagedDecodeSegment(
        req_id="a",
        input_token_ids=(prompt[-1], *drafts),
        start_row=0,
        num_query_tokens=4,
        draft_token_ids=tuple(drafts),
        cache_start_pos=31,
        block_ids=((1, 2, 3),),
    )
    result = proposer.propose(
        ProposeContext(
            target_hidden_states=None,
            target_aux_hidden_states=tuple(mx.split(verified_features, 3, axis=-1)),
            decode_reqs=[("a", state)],
            decode_segments=[segment],
            decode_token_ids=[output],
            prefill_reqs=[],
            prefill_token_ids=[],
            prefill_result_modes=[],
            request_states={"a": state},
            cu_seqlens=[0, 4],
            num_decode_segments=1,
            num_speculative_tokens=3,
            finished_req_ids=set(),
        )
    )
    canonical = mx.concatenate((features, verified_features[: accepted + 1]))
    assert result.draft_token_ids == [_plain_drafts(model, committed, canonical)]


@pytest.mark.parametrize("original", [False, True])
def test_checkpoint_formats_preserve_weights_and_native_outputs(tmp_path, original):
    import mlx.nn as nn

    model = _model(norm_before_residual=not original)
    layer = asdict(model.config.layer)
    if original:
        config = {**layer, "draft_vocab_size": 64}
        weights = {
            name.replace("layers.0.", "midlayer."): torch.from_numpy(np.array(value))
            for name, value in tree_flatten(model.parameters())
            if name != "embed_tokens.weight"
        }
        torch.save(weights, tmp_path / "pytorch_model.bin")
    else:
        quantization = {"bits": 4, "group_size": 32, "mode": "affine"}
        nn.quantize(
            model,
            **quantization,
            class_predicate=lambda _, module: isinstance(module, nn.Linear),
        )
        config = {
            "transformer_layer_config": layer,
            "draft_vocab_size": 64,
            "quantization": quantization,
        }
        mx.save_safetensors(
            str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
        )
    (tmp_path / "config.json").write_text(json.dumps(config))
    loaded = Eagle3Model.load(
        str(tmp_path), TARGET, mx.float32, target_embedding=model.embed_tokens
    )
    assert loaded.config.norm_before_residual is (not original)
    assert loaded.embed_tokens is not model.embed_tokens
    assert loaded.fc.weight.dtype == (mx.float32 if original else mx.uint32)
    assert loaded.embed_tokens.weight.dtype == mx.float32
    tokens, features = mx.array([[1, 2, 3]]), mx.random.normal((1, 3, 192))
    expected, _ = model(tokens, model.combine_hidden_states(features), mask="causal")
    actual, _ = loaded(tokens, loaded.combine_hidden_states(features), mask="causal")
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


def test_preempted_request_replays_output_history_beyond_original_prompt():
    model = _model()
    proposer = _proposer(model)
    history, features = list(range(41)), mx.random.normal((40, 192))
    assert _prefill(proposer, [("r", history, features, [1], 0, 16)], k=0) is None
    proposer.release_requests({"r"})
    # Replayed output tokens belong to the request history, not its original prompt.
    assert (
        _prefill(
            proposer, [("r", history, features, [1, 2], 16, 32)], k=0, prompt_len=16
        )
        is None
    )
    final = _prefill(proposer, [("r", history, features, [1, 2, 3], 32, 40)])
    assert final.draft_token_ids == [_plain_drafts(model, history, features)]
