# SPDX-License-Identifier: Apache-2.0
"""The compatibility bridge must observe, not replace, native model execution."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import gemma4, gemma4_text, llama, qwen3

from vllm_metal.patches.aux_hidden_states import AuxHiddenStateCapture
from vllm_metal.v1.model_adapter import DefaultModelAdapter


def _model(family, *, head_dim=8):
    config = {
        "model_type": family,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": head_dim,
        "rms_norm_eps": 1e-6,
        "vocab_size": 128,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
    }
    if family.startswith("gemma"):
        config.update(
            model_type="gemma4_text",
            global_head_dim=16,
            num_global_key_value_heads=1,
            num_kv_shared_layers=3 if family == "gemma4_text_shared" else 0,
            hidden_size_per_layer_input=0,
            sliding_window=4,
            layer_types=["sliding_attention", "sliding_attention", "full_attention"]
            * 2,
            attention_k_eq_v=True,
            tie_word_embeddings=True,
            use_double_wide_mlp=False,
        )
    module = {"llama": llama, "qwen3": qwen3, "gemma4_text": gemma4_text}[
        config["model_type"]
    ]
    if family == "gemma4":
        model = gemma4.Model(gemma4.ModelArgs(text_config=config, vocab_size=128))
    else:
        model = module.Model(module.ModelArgs.from_dict(config))
    # Match loaded inference models: do not capture lazy random initialization
    # as part of either compiled forward graph.
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize(
    "family", ["llama", "qwen3", "gemma4_text", "gemma4", "gemma4_text_shared"]
)
@pytest.mark.parametrize("compiled", [False, True])
def test_capture_preserves_native_logits_and_layer_semantics(family, compiled):
    model = _model(family)
    adapter = DefaultModelAdapter()
    body = adapter.text_model(model).model
    original_layers = list(body.layers)
    # Preserve requested ordering; layer 0 means the scaled first-layer input.
    capture = AuxHiddenStateCapture(model, (6, 0, 2))
    weights_before = dict(tree_flatten(model.parameters()))

    def observed(tokens):
        result = adapter.target_forward(model, tokens, aux_capture=capture)
        assert result.hidden_states is None
        return result.logits, result.aux_hidden_states

    native = mx.compile(model) if compiled else model
    observed = mx.compile(observed) if compiled else observed
    previous = None
    for ids in ([[1, 2, 3, 4, 5, 6]], [[7, 8, 9, 10, 11, 12]]):
        tokens = mx.array(ids)
        expected = native(tokens)
        logits, auxiliary = observed(tokens)
        mx.eval(expected, logits, auxiliary)
        np.testing.assert_array_equal(np.array(logits), np.array(expected))
        assert all(a is b for a, b in zip(body.layers, original_layers, strict=True))
        np.testing.assert_array_equal(
            np.array(auxiliary[1]),
            np.array((body.embed_tokens(tokens) * getattr(body, "embed_scale", 1))[0]),
        )
        # Execute native prefixes as an independent check of the tap positions.
        for layer_id, hidden in zip((6, 2), (auxiliary[0], auxiliary[2]), strict=True):
            layers = body.layers
            try:
                body.layers = layers[:layer_id]
                prefix = body(tokens)
            finally:
                body.layers = layers
            np.testing.assert_allclose(
                np.array(body.norm(hidden)[None]),
                np.array(prefix),
                atol=2e-5,
                rtol=2e-5,
            )
        if previous is not None:
            assert not np.array_equal(previous, np.array(auxiliary[0]))
        previous = np.array(auxiliary[0])
    weights_after = dict(tree_flatten(model.parameters()))
    assert weights_before.keys() == weights_after.keys()
    assert all(weights_before[k] is weights_after[k] for k in weights_before)


@pytest.mark.parametrize(
    "family", ["llama", "qwen3", "gemma4_text", "gemma4_text_shared"]
)
def test_capture_preserves_cache_updates(family):
    model = _model(family)
    capture = AuxHiddenStateCapture(model, (1, 3, 5))
    if family.startswith("gemma4_text"):
        native_cache, observed_cache = model.make_cache(), model.make_cache()
    else:
        from mlx_lm.models.cache import KVCache

        native_cache = [KVCache() for _ in model.model.layers]
        observed_cache = [KVCache() for _ in model.model.layers]
    for ids in ([[1, 2, 3, 4, 5, 6]], [[7]]):
        tokens = mx.array(ids)
        expected = model(tokens, cache=native_cache)
        actual, auxiliary = capture.run(model, tokens, cache=observed_cache)
        mx.eval(expected, actual, auxiliary)
        np.testing.assert_array_equal(np.array(actual), np.array(expected))
        for a, b in zip(native_cache, observed_cache, strict=True):
            assert a.offset == b.offset
            for (_, x), (_, y) in zip(
                tree_flatten(a.state), tree_flatten(b.state), strict=True
            ):
                np.testing.assert_array_equal(np.array(x), np.array(y))


def test_capture_preserves_parameter_paths_during_call_and_restores_on_failure():
    model = _model("qwen3")
    other = _model("qwen3")
    body = model.model
    layers = list(body.layers)
    before = dict(tree_flatten(model.parameters()))
    capture = AuxHiddenStateCapture(model, (1, 3, 5))

    def failing_forward(tokens):
        during = dict(tree_flatten(model.parameters()))
        assert before.keys() == during.keys()
        assert all(before[k] is during[k] for k in before)
        assert type(other.model.layers[0]) is type(layers[0])
        model(tokens)
        raise ValueError("test exception")

    with pytest.raises(ValueError, match="test exception"):
        capture.run(failing_forward, mx.array([[1, 2]]))
    assert all(a is b for a, b in zip(body.layers, layers, strict=True))
    # An exception must not leave stale features in the following invocation.
    result, aux = capture.run(model, mx.array([[3, 4]]))
    assert len(aux) == 3
    mx.eval(result, aux)


def test_selective_logits_keep_all_auxiliary_rows_and_final_states_separate():
    model = _model("qwen3")
    adapter = DefaultModelAdapter()
    capture = AuxHiddenStateCapture(model, (1, 3, 5))
    tokens = mx.array([[1, 2, 3, 4, 5]])
    full = adapter.target_forward(
        model, tokens, collect_hidden_states=True, aux_capture=capture
    )
    selected = adapter.target_forward(
        model,
        tokens,
        collect_hidden_states=True,
        logits_indices=mx.array([1, 4]),
        aux_capture=capture,
    )
    assert full.hidden_states.shape == selected.hidden_states.shape == (5, 32)
    assert all(h.shape == (5, 32) for h in selected.aux_hidden_states)
    np.testing.assert_array_equal(
        np.array(selected.logits), np.array(full.logits[:, [1, 4]])
    )
    for a, b in zip(full.aux_hidden_states, selected.aux_hidden_states, strict=True):
        np.testing.assert_array_equal(np.array(a), np.array(b))


@pytest.mark.parametrize("layer_ids", [(), (-1,), (7,), (1.5,), (True,), ("1",)])
def test_invalid_layer_ids_are_rejected_before_forward(layer_ids):
    with pytest.raises(ValueError, match="Invalid auxiliary layer IDs"):
        AuxHiddenStateCapture(_model("qwen3"), layer_ids)


def test_empty_backbone_is_rejected():
    model = _model("qwen3")
    model.model.layers = []
    with pytest.raises(ValueError, match="Invalid auxiliary layer IDs"):
        AuxHiddenStateCapture(model, (0,))


def test_unknown_decoder_contract_is_rejected_without_mutating_model():
    model = _model("qwen3")
    model.model.layers[0] = nn.Identity()
    layers = list(model.model.layers)
    with pytest.raises(NotImplementedError, match="Auxiliary capture"):
        AuxHiddenStateCapture(model, (1,))
    assert all(a is b for a, b in zip(model.model.layers, layers, strict=True))


@pytest.mark.parametrize("compiled", [False, True])
def test_bypassed_capture_fails_and_restores_layers(compiled):
    model = _model("qwen3")
    layers = list(model.model.layers)
    capture = AuxHiddenStateCapture(model, (1,))
    tokens = mx.array([[1, 2]])
    if compiled:
        forward = mx.compile(model)
        mx.eval(forward(tokens))  # Warm the graph before installing observers.
    else:
        other = _model("qwen3")
        forward = other
    with pytest.raises(RuntimeError, match="Native forward bypassed"):
        capture.run(forward, tokens)
    assert all(a is b for a, b in zip(model.model.layers, layers, strict=True))


def test_duplicate_ids_and_disabled_capture():
    model = _model("qwen3")
    tokens = mx.array([[1, 2, 3, 4, 5, 6]])
    adapter = DefaultModelAdapter()
    baseline = adapter.target_forward(model, tokens)
    assert baseline.aux_hidden_states == ()
    capture = AuxHiddenStateCapture(model, (2, 0, 2))
    result = adapter.target_forward(model, tokens, aux_capture=capture)
    mx.eval(baseline.logits, result.logits, result.aux_hidden_states)
    np.testing.assert_array_equal(np.array(baseline.logits), np.array(result.logits))
    assert all(h.shape == (6, 32) for h in result.aux_hidden_states)
    np.testing.assert_array_equal(
        np.array(result.aux_hidden_states[0]), np.array(result.aux_hidden_states[2])
    )
    np.testing.assert_array_equal(
        np.array(result.aux_hidden_states[1]),
        np.array(model.model.embed_tokens(tokens).reshape(6, 32)),
    )


def test_adapter_rejects_capture_with_missing_token_rows(monkeypatch):
    model = _model("qwen3")
    adapter = DefaultModelAdapter()
    forward = adapter._target_forward

    # A row-reducing optimization may preserve last-token logits while dropping
    # features needed by a drafter. Never silently return misaligned features.
    def last_token_only(model, tokens, **kwargs):
        return forward(model, tokens[:, -1:], **kwargs)

    monkeypatch.setattr(adapter, "_target_forward", last_token_only)
    with pytest.raises(ValueError, match="one state per input token"):
        adapter.target_forward(
            model,
            mx.array([[1, 2, 3]]),
            aux_capture=AuxHiddenStateCapture(model, (0, 2)),
        )


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_capture_preserves_paged_writes_and_packed_row_order(family):
    """Exercise the shared storage and real Metal kernels across a page boundary."""
    import torch
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    from vllm_metal.attention.context import (
        OffsetCache,
        clear_context,
        get_context,
        prepare_grouped,
    )
    from vllm_metal.attention.runtime.sdpa import SDPAPagedAttentionRuntime

    models = [_model(family, head_dim=64) for _ in range(2)]
    models[1].load_weights(tree_flatten(models[0].parameters()))
    names = tuple(f"layers.{i}.self_attn" for i in range(6))
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float32
    )
    config = VllmConfig()
    config.cache_config.kv_cache_layout = "LBNHC"
    kv_config = get_kv_cache_config_from_groups(
        config,
        [KVCacheGroupSpec(layer_names=list(names), kv_cache_spec=spec)],
        6 * 8 * spec.page_size_bytes,
    )
    kv_config.kv_cache_layout = "LBNHC"
    runtimes = []
    for model in models:
        runtime = SDPAPagedAttentionRuntime(block_size=16, dtype=mx.float32)
        runtime.initialize_from_config(kv_config, names)
        assert runtime.patch_model(model) == 6
        runtimes.append(runtime)
    capture = AuxHiddenStateCapture(models[1], (6, 0, 2))
    adapter = DefaultModelAdapter()
    # Non-contiguous pages; ragged prefill, reordered decode, then a mixed batch.
    steps = [
        ([], [([[3, 1]], 18, 0), ([[5]], 3, 0)], list(range(1, 22))),
        ([([[5]], 3), ([[3, 1]], 18)], [], [22, 23]),
        ([([[3, 1]], 19)], [([[5]], 2, 4)], [24, 25, 26]),
    ]
    written_slots = set()
    try:
        for decode, prefill, ids in steps:
            tokens = mx.array([ids])
            results = []
            for index, (model, runtime) in enumerate(
                zip(models, runtimes, strict=True)
            ):
                prepare_grouped(decode, prefill, (16,))
                ctx = get_context()
                written_slots.update(ctx.slot_mapping)
                result = adapter.target_forward(
                    model,
                    tokens,
                    cache=[OffsetCache(max(ctx.offsets)) for _ in range(6)],
                    aux_capture=capture if index else None,
                )
                outputs = [result.logits, *result.aux_hidden_states]
                runtime.extend_forward_eval_outputs(outputs)
                mx.eval(outputs)
                results.append(result)
                clear_context()
            plain, observed = results
            np.testing.assert_array_equal(
                np.array(plain.logits), np.array(observed.logits)
            )
            assert all(h.shape == (len(ids), 32) for h in observed.aux_hidden_states)
            np.testing.assert_array_equal(
                np.array(observed.aux_hidden_states[1]),
                np.array(models[1].model.embed_tokens(tokens)[0]),
            )
            # The final pre-norm capture must project to every packed logit row.
            projected = models[1].lm_head(
                models[1].model.norm(observed.aux_hidden_states[0])
            )
            np.testing.assert_array_equal(
                np.array(projected), np.array(observed.logits[0])
            )
            slots = mx.array(sorted(written_slots))
            for name in ("key_caches", "value_caches"):
                caches = [getattr(r.kv_cache, name) for r in runtimes]
                for a, b in zip(*caches, strict=True):
                    # Compare initialized slots only; unused pages have no contract.
                    np.testing.assert_array_equal(
                        np.array(a.reshape(-1, 2, 64)[slots]),
                        np.array(b.reshape(-1, 2, 64)[slots]),
                    )
    finally:
        clear_context()
