# SPDX-License-Identifier: Apache-2.0
"""Shared real-model check for ``call_lm_hidden_states`` / head parity."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from vllm_metal.v1.model_adapter import DefaultModelAdapter


def assert_hidden_states_reproduce_call_lm_logits(
    adapter: Any,
    model: Any,
    input_ids: mx.array,
    *,
    num_layers: int,
) -> None:
    """Assert ``project_logits`` applied to backbone states equals LM logits.

    Covers the full projection and the runner's ``logits_indices`` selected-row
    path.  ``num_layers`` must match the backbone's layer count: the cache list
    is zipped with the layers, so a short list silently skips the tail layers.
    """
    embeds = adapter.embed_tokens(input_ids)
    seq_len = input_ids.shape[1]
    position_ids = mx.broadcast_to(
        mx.arange(seq_len, dtype=mx.int32)[None, None, :], (3, 1, seq_len)
    )
    cache = [None] * num_layers

    logits = adapter.call_lm(input_ids, embeds, cache, position_ids).logits
    hidden = adapter.call_lm_hidden_states(input_ids, embeds, cache, position_ids)
    head = DefaultModelAdapter()
    rows = mx.array([1, seq_len - 1], dtype=mx.int32)

    assert mx.array_equal(head.project_logits(model, hidden), logits).item()
    assert mx.allclose(
        head.project_logits(model, hidden, logits_indices=rows)[0],
        logits[0, rows],
    ).item()
