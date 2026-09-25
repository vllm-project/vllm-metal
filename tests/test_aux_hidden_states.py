# SPDX-License-Identifier: Apache-2.0
"""The compatibility bridge must observe, not replace, native model execution."""

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import gemma4, gemma4_text, llama, qwen3

from vllm_metal.patches.aux_hidden_states import AuxHiddenStateCapture
from vllm_metal.v1.model_adapter import DefaultModelAdapter


def _model(family):
    config = {
        "model_type": family,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
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
            num_kv_shared_layers=0,
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


@pytest.mark.parametrize("family", ["llama", "qwen3", "gemma4_text", "gemma4"])
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


@pytest.mark.parametrize("family", ["llama", "qwen3", "gemma4_text"])
def test_capture_preserves_cache_updates(family):
    model = _model(family)
    capture = AuxHiddenStateCapture(model, (1, 3, 5))
    if family == "gemma4_text":
        native_cache, observed_cache = model.make_cache(), model.make_cache()
    else:
        from mlx_lm.models.cache import KVCache

        native_cache = [KVCache() for _ in model.layers]
        observed_cache = [KVCache() for _ in model.layers]
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
