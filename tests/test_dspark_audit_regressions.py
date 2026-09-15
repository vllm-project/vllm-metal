# SPDX-License-Identifier: Apache-2.0
"""Passing regressions for the original wrong-logits and context audit findings."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from tests.test_dspark_proposer import _context, _proposer, _seed, _state
from tests.test_hidden_state_tap import _toy_backbone
from vllm_metal.v1.model_adapter import DefaultModelAdapter
from vllm_metal.v1.model_runner import MetalModelRunner


def test_capture_preserves_packed_logits_selection():
    adapter = DefaultModelAdapter()
    adapter._target_backbone = lambda model: _toy_backbone(1)
    adapter._compute_target_logits = lambda model, hidden: hidden
    layout = MetalModelRunner._paged_logits_layout(
        SimpleNamespace(_selective_logits_supported=True),
        [0, 3, 7, 12],
        num_decode_segments=1,
    )
    result = adapter.target_forward(
        object(),
        mx.arange(12, dtype=mx.float32)[None],
        cache=[None],
        capture_layer_ids=[0],
        logits_indices=layout.indices,
    )
    assert result.hidden_states.shape[0] == 12
    assert result.logits.shape == (1, 5, 1)
    assert result.logits[0, :, 0].tolist() == [0, 1, 2, 6, 11]


@pytest.mark.parametrize(
    "cached,committed", [(5, 4), (2, 8)], ids=["rollback", "missing-span"]
)
def test_context_plan_requires_complete_physical_coverage(cached, committed):
    proposer = _proposer()
    state = _state(list(range(cached + 1)))
    _seed(proposer, state, k=0)
    state.token_ids = list(range(committed))
    start = committed - 2
    result = proposer.propose(
        _context(decode=[("r", state, start, [start], [committed - 1])])
    )
    # Failing closed is valid. Returning a proposal requires every physical
    # layer to cover precisely the committed prefix preceding the anchor.
    if result is not None:
        context = proposer._contexts["r"]
        assert context.covered_end == committed - 1
        assert all(cache.length == committed - 1 for cache in context.caches)
    if cached > committed:
        assert result is not None  # rollback must retain the valid prefix
    else:
        assert result is None  # missing features must not be renumbered
