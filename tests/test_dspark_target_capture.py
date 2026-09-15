# SPDX-License-Identifier: Apache-2.0
"""Runner delivery of absolute feature spans on no-sample prefill steps.

Also hosts the native-versus-paged capture parity check for a tiny Qwen3
target. The check runs offline: it downloads nothing and loads no drafter.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import make_prompt_cache

import vllm_metal.v1.model_runner as mr
from tests.stub_runner import make_stub_runner
from tests.test_hidden_state_tap import _toy_backbone
from tests.test_v1_model_runner_generate import (
    TestIntermediateBodyOnlyForward as _PrefillFixture,
)
from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.context import OffsetCache, clear_context, prepare_grouped
from vllm_metal.attention.impls.sdpa_wrapper import patch_sdpa_attention
from vllm_metal.v1.model_adapter import DefaultModelAdapter


def check_target(model, layer_ids: list[int]) -> list[dict]:
    """Bit-exact capture parity at the same head/attention shapes."""
    adapter = DefaultModelAdapter()
    assert adapter.supports_selective_logits(model)
    original_layers = model.model.layers
    results = []

    def compare(label, plain, captured):
        mx.eval(plain, captured)
        error = float(
            mx.max(
                mx.abs(plain.astype(mx.float32) - captured.astype(mx.float32))
            ).item()
        )
        assert bool(mx.array_equal(plain, captured).item()), (label, error)
        results.append(
            {"case": label, "shape": list(plain.shape), "max_abs_error": error}
        )

    for selective in (False, True):
        caches = [make_prompt_cache(model), make_prompt_cache(model)]
        for index, length in enumerate((1, 7, 5, 3)):
            ids = (mx.arange(length, dtype=mx.int32) + 20 + index)[None]
            selection = mx.array([length - 1]) if selective else None
            outputs = [
                adapter.target_forward(
                    model,
                    ids,
                    cache=cache,
                    logits_indices=selection,
                    capture_layer_ids=layer_ids if capture else None,
                )
                for capture, cache in zip((False, True), caches, strict=True)
            ]
            compare(
                f"native_{'selected' if selective else 'full'}_chunk{index}",
                outputs[0].logits,
                outputs[1].logits,
            )
            assert outputs[1].hidden_states.shape == (
                length,
                len(layer_ids) * model.args.hidden_size,
            )
            for plain, captured in zip(*caches, strict=True):
                assert plain.offset == captured.offset
                assert mx.array_equal(plain.keys, captured.keys)
                assert mx.array_equal(plain.values, captured.values)

    # Fresh per-mode paged storage; the actual attention wrappers are shared
    # with the target body copy, and must write the same physical KV slots.
    for merged in (False, True):
        for selective in (False, True):
            outputs = []
            saved_kv = []
            for capture in (False, True):
                args = model.args
                kv = MetalPagedKVCache(
                    args.num_hidden_layers,
                    args.num_key_value_heads,
                    args.head_dim,
                    3,
                    32,
                    dtype=mx.float16,
                )
                assert patch_sdpa_attention(model, kv, 32) == args.num_hidden_layers
                offset_caches = [OffsetCache(0) for _ in model.layers]
                prepare_grouped([], [([[0]], 3, 0), ([[1]], 2, 0)], (32,))
                try:
                    seed = model(mx.array([[10, 11, 12, 20, 21]]), cache=offset_caches)
                    mx.eval(seed)
                finally:
                    clear_context()
                prepare_grouped(
                    [([[0]], 3, 3)],
                    [([[1]], 4, 2), ([[2]], 5, 0)],
                    (32,),
                    merge_verify_windows=merged,
                )
                try:
                    output = adapter.target_forward(
                        model,
                        mx.array([[13, 14, 15, 22, 23, 24, 25, 30, 31, 32, 33, 34]]),
                        cache=offset_caches,
                        logits_indices=mx.array([0, 1, 2, 6, 11])
                        if selective
                        else None,
                        capture_layer_ids=layer_ids if capture else None,
                    )
                    mx.eval(output.logits, output.hidden_states)
                finally:
                    clear_context()
                outputs.append(output)
                saved_kv.append([*kv.key_caches, *kv.value_caches])
            compare(
                f"paged_mixed_merged{merged}_selected{selective}",
                outputs[0].logits,
                outputs[1].logits,
            )
            assert outputs[1].hidden_states.shape == (
                12,
                len(layer_ids) * model.args.hidden_size,
            )
            assert all(
                bool(mx.array_equal(a, b).item())
                for a, b in zip(*saved_kv, strict=True)
            )
    assert model.model.layers is original_layers
    return results


@pytest.mark.parametrize("prompt_logprobs", [False, True])
def test_intermediate_features_reach_proposer_without_sampling(
    monkeypatch, prompt_logprobs
):
    helper = _PrefillFixture()
    first = helper._intermediate_prefill_request("a", start_pos=3)
    second = helper._intermediate_prefill_request("b", start_pos=0)
    # Two independent spans: equal row counts must not obscure their positions.
    runner = make_stub_runner(
        model=SimpleNamespace(model=_toy_backbone(3), lm_head=lambda h: h),
    )
    runner.num_layers = 3
    runner._paged_block_size = 4
    runner._paged_group_block_sizes = (4,)
    seen = []
    runner._drafter = SimpleNamespace(
        capture_layer_ids=[0, 2],
        needs_target_hidden_states=lambda *a, **k: True,
        propose=lambda ctx: seen.append(ctx),
    )
    monkeypatch.setattr(mr, "prepare_grouped", lambda *a, **k: None)
    monkeypatch.setattr(
        runner._prompt_logprobs_tracker, "wants_any", lambda ids: prompt_logprobs
    )
    # Prompt-logprobs collection has separate coverage; verify it forces the
    # full head here without consuming a random sample on intermediate rows.
    monkeypatch.setattr(runner, "_gather_prefill_prompt_logprobs", lambda *a: None)
    monkeypatch.setattr(
        mr, "sample_prefill_tokens", lambda *a, **k: pytest.fail("sampled")
    )
    batch = mr._ExecutionBatch()
    for request in (first, second):
        idx = batch.add_output(request.req_id, [])
        batch.paged_prefill_entries.append(
            mr._PendingPrefillEntry(idx, request, "intermediate")
        )
    runner._start_paged_forward(
        batch, [first, second], [], helper._make_scheduler_output({"a": 2, "b": 2})
    )
    assert (runner._execute_model_state.logits is not None) is prompt_logprobs
    runner._sample_paged_batch()
    assert len(seen) == 1
    ctx = seen[0]
    assert ctx.target_hidden_states.shape == (4, 2)
    assert ctx.target_hidden_states[:, 0].tolist() == [5, 6, 5, 6]
    assert ctx.cu_seqlens == [0, 2, 4]
    assert [(pr.req_id, pr.start_pos) for pr in ctx.prefill_reqs] == [
        ("a", 3),
        ("b", 0),
    ]
    assert ctx.prefill_token_ids == []
    assert ctx.prefill_result_modes == ["intermediate", "intermediate"]
    assert batch.sampled_tokens == [[], []]
    assert runner._paged_request_seq_lens == {"a": 5, "b": 2}


def test_native_and_paged_capture_match_tiny_qwen3():
    from mlx_lm.models.qwen3 import Model, ModelArgs

    mx.random.seed(42)
    model = Model(
        ModelArgs(
            model_type="qwen3",
            vocab_size=64,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=128,
            rms_norm_eps=1e-6,
            max_position_embeddings=128,
            rope_theta=1_000_000.0,
            tie_word_embeddings=False,
        )
    )
    model.set_dtype(mx.float16)
    results = check_target(model, [0, 1])
    assert len(results) == 12
