# SPDX-License-Identifier: Apache-2.0
"""EAGLE3 numerical reference and prefix/cache lifecycle regression tests."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
import torch.nn.functional as torch_f
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache
from vllm import SamplingParams

from vllm_metal.v1.eagle3 import Eagle3Config, Eagle3Model
from vllm_metal.v1.eagle3_proposer import Eagle3Proposer
from vllm_metal.v1.model_runner import PrefillRequest
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.spec_decode import PagedDecodeSegment, SpeculativeDecodeController


def _model(kv_heads=2, norm_before_residual=True):
    target = {
        "model_type": "qwen3",
        "hidden_size": 64,
        "num_hidden_layers": 8,
        "vocab_size": 128,
    }
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
        target,
    )
    model = Eagle3Model(config)
    model.d2t = mx.arange(64, dtype=mx.int32)
    mx.eval(model.parameters())
    return model


def _reference(model, tokens, features):
    """Independent PyTorch implementation of the checkpoint's dense equations."""
    weights = {
        name: torch.from_numpy(np.array(value))
        for name, value in tree_flatten(model.parameters())
    }
    tokens = torch.from_numpy(np.array(tokens).astype(np.int64))
    hidden = torch.from_numpy(np.array(features))

    def linear(x, name):
        return torch_f.linear(x, weights[name + ".weight"])

    def norm(x, name):
        return torch_f.rms_norm(x, (x.shape[-1],), weights[name + ".weight"], 1e-6)

    hidden = linear(hidden, "fc")
    embeddings = torch_f.embedding(tokens, weights["embed_tokens.weight"])
    residual = (
        norm(hidden, "layers.0.hidden_norm")
        if model.config.norm_before_residual
        else hidden
    )
    x = torch.cat(
        (
            norm(embeddings, "layers.0.input_layernorm"),
            norm(hidden, "layers.0.hidden_norm"),
        ),
        -1,
    )
    b, length, _ = x.shape
    head_dim = model.config.layer.head_dim
    queries = (
        linear(x, "layers.0.self_attn.q_proj")
        .view(b, length, 4, head_dim)
        .transpose(1, 2)
    )
    kv_heads = model.config.layer.num_key_value_heads
    keys = (
        linear(x, "layers.0.self_attn.k_proj")
        .view(b, length, kv_heads, head_dim)
        .transpose(1, 2)
    )
    values = (
        linear(x, "layers.0.self_attn.v_proj")
        .view(b, length, kv_heads, head_dim)
        .transpose(1, 2)
    )
    angles = torch.arange(length)[:, None] * (
        10000.0 ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    )
    cosine, sine = angles.cos()[None, None], angles.sin()[None, None]

    def rope(x):
        left, right = x.chunk(2, -1)
        return torch.cat(
            (left * cosine - right * sine, right * cosine + left * sine), -1
        )

    attention = torch_f.scaled_dot_product_attention(
        rope(queries), rope(keys), values, is_causal=True, enable_gqa=True
    )
    hidden = residual + linear(
        attention.transpose(1, 2).reshape(b, length, 4 * head_dim),
        "layers.0.self_attn.o_proj",
    )
    x = norm(hidden, "layers.0.post_attention_layernorm")
    hidden = hidden + linear(
        torch_f.silu(linear(x, "layers.0.mlp.gate_proj"))
        * linear(x, "layers.0.mlp.up_proj"),
        "layers.0.mlp.down_proj",
    )
    return norm(hidden, "norm").numpy(), hidden.numpy()


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("norm_before_residual", [True, False])
def test_head_matches_pytorch_gqa_mha_reference(kv_heads, norm_before_residual):
    model = _model(kv_heads, norm_before_residual)
    tokens = mx.array([[1, 9, 7, 22, 63], [8, 3, 12, 91, 24]])
    features = mx.random.normal((2, 5, 192))
    expected, expected_recurrent = _reference(model, tokens, features)
    # Check the equations in full FP32. MLX's M5 GPU matmul defaults to
    # TF32 (MLX_ENABLE_TF32), while this PyTorch reference uses CPU FP32.
    # The paged/ragged tests below exercise the GPU execution separately.
    with mx.stream(mx.cpu):
        actual, recurrent = model(
            tokens, model.combine_hidden_states(features), mask="causal"
        )
    np.testing.assert_allclose(np.array(actual), expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        np.array(recurrent), expected_recurrent, rtol=2e-5, atol=2e-5
    )


def _proposer(model, *, max_model_len=4096):
    proposer = Eagle3Proposer(
        model=model,
        controller=SpeculativeDecodeController(),
        num_blocks=16,
        max_model_len=max_model_len,
        block_size=16,
        dtype=mx.float32,
    )
    proposer.adopt_scheduler_group(0, max_model_len)
    return proposer


def _prefill(proposer, requests, *, k=3):
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
                block_ids=[blocks],
                generator=None,
                prompt_len=None if intermediate else end,
                start_pos=start,
                full_prompt_token_ids=prompt[:-1],
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


def test_quantized_checkpoint_preserves_packed_weights_and_dense_embeddings(tmp_path):
    import mlx.nn as nn

    model = _model()
    nn.quantize(
        model,
        bits=4,
        group_size=32,
        class_predicate=lambda _, module: isinstance(module, nn.Linear),
    )
    mx.eval(model.parameters())
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    config = {
        "speculators_model_type": "eagle3",
        "draft_vocab_size": 64,
        "norm_before_residual": True,
        "transformer_layer_config": asdict(model.config.layer),
        "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    loaded = Eagle3Model.load(
        str(tmp_path),
        {
            "model_type": "qwen3",
            "hidden_size": 64,
            "num_hidden_layers": 8,
            "vocab_size": 128,
        },
        mx.float32,
    )
    assert loaded.fc.weight.dtype == mx.uint32
    assert loaded.embed_tokens.weight.dtype == mx.float32
    tokens = mx.array([[1, 2, 3]])
    features = mx.random.normal((1, 3, 192))
    expected, _ = model(tokens, model.combine_hidden_states(features), mask="causal")
    actual, _ = loaded(tokens, loaded.combine_hidden_states(features), mask="causal")
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


def test_original_checkpoint_loads_midlayer_weights_and_borrows_target_embedding(
    tmp_path,
):
    model = _model(norm_before_residual=False)
    config = {**asdict(model.config.layer), "draft_vocab_size": 64}
    (tmp_path / "config.json").write_text(json.dumps(config))
    state = {
        name.replace("layers.0.", "midlayer."): torch.from_numpy(np.array(value))
        for name, value in tree_flatten(model.parameters())
        if name != "embed_tokens.weight"
    }
    torch.save(state, tmp_path / "pytorch_model.bin")
    target = {
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 8,
        "vocab_size": 128,
    }
    with pytest.raises(ValueError, match="target's dense embedding"):
        Eagle3Model.load(str(tmp_path), target, mx.float32)
    loaded = Eagle3Model.load(
        str(tmp_path), target, mx.float32, target_embedding=model.embed_tokens
    )
    assert loaded.config.norm_before_residual is False
    assert loaded.embed_tokens is not model.embed_tokens
    tokens, features = mx.array([[1, 2, 3]]), mx.random.normal((1, 3, 192))
    expected, _ = model(tokens, model.combine_hidden_states(features), mask="causal")
    actual, _ = loaded(tokens, loaded.combine_hidden_states(features), mask="causal")
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


def test_preempted_request_replays_output_history_beyond_original_prompt():
    model = _model()
    proposer = _proposer(model)
    history = list(range(41))
    features = mx.random.normal((40, 192))
    _prefill(proposer, [("r", history[:17], features[:16], [1], 0, 16)], k=0)
    proposer.release_requests({"r"})
    state = SimpleNamespace(
        token_ids=history, sampling_params=SamplingParams(temperature=0)
    )
    prefill = PrefillRequest(
        req_id="r",
        token_ids=history[16:32],
        sampling_params=state.sampling_params,
        block_ids=[[1, 2]],
        generator=None,
        prompt_len=None,
        start_pos=16,
        full_prompt_token_ids=history[:16],
    )
    result = proposer.propose(
        ProposeContext(
            target_hidden_states=None,
            target_aux_hidden_states=tuple(mx.split(features[16:32], 3, axis=-1)),
            decode_reqs=[],
            decode_segments=[],
            decode_token_ids=[],
            prefill_reqs=[prefill],
            prefill_token_ids=[-1],
            prefill_result_modes=["intermediate"],
            request_states={"r": state},
            cu_seqlens=[0, 16],
            num_decode_segments=0,
            num_speculative_tokens=0,
            finished_req_ids=set(),
        )
    )
    assert result is None
    final = _prefill(proposer, [("r", history, features, [1, 2, 3], 32, 40)])
    assert final.draft_token_ids == [_plain_drafts(model, history, features)]


def test_upstream_cache_manager_recomputes_boundary_before_draft_prefix_reuse():
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    from vllm.v1.request import Request

    init_none_hash(sha256)
    hasher = get_request_block_hasher(16, sha256)
    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=16,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["draft_layers.0.self_attn"],
                    FullAttentionSpec(
                        block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float32
                    ),
                )
            ],
        ),
        max_model_len=128,
        scheduler_block_size=16,
        hash_block_size=16,
        use_eagle=True,
        num_prefill_lookahead=1,
    )

    def request(name, tokens):
        return Request(
            name,
            tokens,
            SamplingParams(temperature=0, max_tokens=8),
            None,
            block_hasher=hasher,
        )

    model = _model()
    proposer = _proposer(model)
    original = list(range(51))
    features = mx.random.normal((50, 192))
    owner = request("owner", original)
    manager.allocate_slots(owner, 50, num_lookahead_tokens=3)
    blocks = manager.get_blocks(owner.request_id).get_block_ids()[0]
    _prefill(proposer, [("owner", original, features, blocks, 0, 50)])
    manager.cache_blocks(owner, 50)
    manager.free(owner)

    # The next token differs just after two matched blocks. The final matched
    # block contains a KV pair depending on that next token and must be replayed.
    changed = original[:32] + [87, 92, 12, 41, 56, 22, 75]
    changed_features = mx.concatenate((features[:32], mx.random.normal((6, 192))))
    resumed = request("resumed", changed)
    hit_blocks, hit_tokens, _ = manager.get_computed_blocks(resumed)
    assert hit_tokens == 16
    shared = hit_blocks.get_block_ids()[0][0]
    # Freeze values on the CPU; another MLX array can alias in-place writes.
    shared_keys = np.array(proposer._kv.key_caches[0][shared]).copy()
    manager.allocate_slots(
        resumed,
        38 - hit_tokens,
        num_new_computed_tokens=hit_tokens,
        new_computed_blocks=hit_blocks,
        num_lookahead_tokens=3,
    )
    blocks = manager.get_blocks(resumed.request_id).get_block_ids()[0]
    result = _prefill(
        proposer,
        [("resumed", changed, changed_features, blocks, hit_tokens, 38)],
    )
    reference_cache = KVCache()
    assert result.draft_token_ids == [
        _plain_drafts(model, changed, changed_features, cache=reference_cache)
    ]
    np.testing.assert_array_equal(
        shared_keys, np.array(proposer._kv.key_caches[0][shared])
    )
    physical = [blocks[pos // 16] * 16 + pos % 16 for pos in range(38)]
    for pool, reference in (
        (proposer._kv.key_caches[0], reference_cache.keys),
        (proposer._kv.value_caches[0], reference_cache.values),
    ):
        actual = pool.reshape(-1, 2, 64)[mx.array(physical)].transpose(1, 0, 2)
        np.testing.assert_allclose(
            np.array(actual), np.array(reference[0, :, :38]), atol=3e-3, rtol=3e-3
        )
