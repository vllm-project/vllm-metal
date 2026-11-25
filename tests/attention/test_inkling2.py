# SPDX-License-Identifier: Apache-2.0
"""Paged loop visits must reproduce independent native request histories."""

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from mlx_vlm.models.inkling2.config import TextConfig
from mlx_vlm.models.inkling2.language import LanguageModel
from vllm.v1.core.kv_cache_planning import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)

from tests.stub_runner import _grouping_config
from vllm_metal.attention.context import clear_context, prepare_grouped
from vllm_metal.attention.runtime.inkling2 import Inkling2Plan, Inkling2Runtime
from vllm_metal.v1.model_adapter import DefaultModelAdapter


@pytest.fixture
def models():
    mx.random.seed(42)
    config = TextConfig(
        model_type="inkling2_loop_text",
        hidden_size=64,
        num_hidden_layers=4,
        vocab_size=64,
        unpadded_vocab_size=60,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=64,
        sliding_window_size=8,
        local_layer_ids=[0, 1, 3],
        d_rel=4,
        rel_extent=8,
        log_scaling_n_floor=16,
        n_layers_per_attn_res_block=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        intermediate_size=32,
        loop_config={"core_start": 1, "core_end": 3, "num_passes": 3, "gate_rank": 8},
    )
    reference = LanguageModel(config)
    reference.load_weights(
        [
            (name, mx.random.normal(value.shape) * 0.1)
            for name, value in tree_flatten(reference.parameters())
        ]
    )
    reference.eval()
    paged = LanguageModel(config)
    paged.load_weights(tree_flatten(reference.parameters()))
    paged.eval()
    mx.eval(reference.parameters(), paged.parameters())
    plan = Inkling2Plan.from_config(config)
    runtime = Inkling2Runtime(plan, mx.float32)
    specs = plan.cache_specs(8, torch.float32)
    grouping = _grouping_config(8, 2048)
    groups = get_kv_cache_groups(grouping, specs)
    storage = get_kv_cache_config_from_groups(
        grouping,
        groups,
        2048 * max(s.page_size_bytes for s in specs.values()) * len(specs),
    )
    storage.kv_cache_layout = "LBNHC"
    runtime.initialize_from_config(storage)
    runtime.patch_model(paged)
    yield reference, paged, runtime
    clear_context()


def run_paged(model, runtime, requests):
    """Requests are (tokens, absolute offset, physical block base)."""
    prefill = []
    tokens = []
    sizes = runtime.kv_group_block_sizes()
    for ids, offset, base in requests:
        tables = [
            list(
                range(
                    base + group * 128,
                    base + group * 128 + (offset + len(ids) + size - 1) // size,
                )
            )
            for group, size in enumerate(sizes)
        ]
        prefill.append((tables, len(ids), offset))
        tokens.extend(ids)
    prepare_grouped([], prefill, sizes)
    try:
        logits = model(mx.array([tokens])).logits
        outputs = [logits]
        runtime.extend_forward_eval_outputs(outputs)
        mx.eval(outputs)
        return np.array(logits)
    finally:
        clear_context()


def test_looped_packed_prefill_and_decode_preserve_request_histories(models):
    reference, paged, runtime = models
    caches = [reference.make_cache(), reference.make_cache()]
    prefixes = [[1, 2, 3, 4, 5, 6], [8, 9, 10]]
    expected = [
        reference(mx.array([ids]), cache=cache).logits
        for ids, cache in zip(prefixes, caches, strict=True)
    ]
    actual = run_paged(paged, runtime, [(prefixes[0], 0, 1), (prefixes[1], 0, 40)])
    np.testing.assert_allclose(
        actual, np.array(mx.concatenate(expected, axis=1)), atol=2e-5, rtol=2e-5
    )
    offsets = [6, 3]
    for step in range(8):
        inputs = [[12 + step], [30 + step]]
        expected = [
            reference(mx.array([ids]), cache=cache).logits
            for ids, cache in zip(inputs, caches, strict=True)
        ]
        actual = run_paged(
            paged, runtime, [(inputs[0], offsets[0], 1), (inputs[1], offsets[1], 40)]
        )
        np.testing.assert_allclose(
            actual, np.array(mx.concatenate(expected, axis=1)), atol=2e-5, rtol=2e-5
        )
        offsets = [offset + 1 for offset in offsets]


def test_chunked_prefill_restarts_convolutions_from_cached_prefix(models):
    reference, paged, runtime = models
    ids = list(range(1, 22))
    expected = reference(mx.array([ids]), cache=reference.make_cache()).logits
    chunks = []
    for start, end in ((0, 4), (4, 7), (7, 13), (13, 21)):
        chunks.append(run_paged(paged, runtime, [(ids[start:end], start, 1)]))
    np.testing.assert_allclose(
        np.concatenate(chunks, axis=1), np.array(expected), atol=2e-5, rtol=2e-5
    )
    # A new request can resume an earlier cached boundary without depending
    # on the previous request's final convolution state.
    suffix = [45, 46, 47]
    expected = reference(
        mx.array([ids[:4] + suffix]), cache=reference.make_cache()
    ).logits[:, 4:]
    actual = run_paged(paged, runtime, [(suffix, 4, 1)])
    np.testing.assert_allclose(actual, np.array(expected), atol=2e-5, rtol=2e-5)


def test_selective_logits_keep_native_head_and_vocabulary(models):
    reference, _, _ = models
    adapter = DefaultModelAdapter()
    ids = mx.array([[1, 2, 3]])
    expected = reference(ids, cache=reference.make_cache()).logits[:, -1:]
    actual = adapter.target_forward(reference, ids, logits_indices=mx.array([2])).logits
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), atol=1e-7, rtol=1e-6
    )


def test_copied_prefix_preserves_every_visit_and_convolution_stream(models):
    reference, paged, runtime = models
    prefix = [1, 2, 3, 4, 5]
    run_paged(paged, runtime, [(prefix, 0, 1)])
    copies = [
        (1 + group * 128 + block, 40 + group * 128 + block)
        for group, size in enumerate(runtime.kv_group_block_sizes())
        for block in range((len(prefix) + size - 1) // size)
    ]
    runtime.copy_blocks(copies)
    mx.eval(runtime.storage.buffers)
    suffixes = [[12, 13], [30, 31]]
    actual = run_paged(paged, runtime, [(suffixes[0], 5, 1), (suffixes[1], 5, 40)])
    expected = [
        reference(mx.array([prefix + suffix]), cache=reference.make_cache()).logits[
            :, 5:
        ]
        for suffix in suffixes
    ]
    np.testing.assert_allclose(
        actual, np.array(mx.concatenate(expected, axis=1)), atol=2e-5, rtol=2e-5
    )


def test_cache_specs_reject_page_splitting_that_could_materialize_kv():
    plan = Inkling2Plan((0,), (-1,), (2,), (64,), 64, 4)
    with pytest.raises(ValueError, match="native kernel block size"):
        plan.cache_specs(64, torch.float32)
