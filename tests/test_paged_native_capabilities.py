# SPDX-License-Identifier: Apache-2.0
"""Public native ABI capabilities are distinct from test-kernel availability."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_metal.metal import paged_attention_capabilities


def test_structured_capabilities_are_normalized():
    query = MagicMock(
        return_value={
            "gqa_decode": False,
            "gqa_disable": False,
            "decode_routing_metadata": True,
        }
    )
    ops = SimpleNamespace(paged_attention_capabilities=query)
    assert paged_attention_capabilities(ops) == {
        "gqa_decode": False,
        "gqa_disable": False,
        "decode_routing_metadata": True,
        "gqa_batch_context_lens": False,
    }
    query.assert_called_once_with()


def test_missing_binding_fails_loudly():
    # The query is part of the required extension ABI. A build without it
    # fails loudly here rather than guessing at older capability probes.
    with pytest.raises(AttributeError):
        paged_attention_capabilities(SimpleNamespace())


def test_missing_capabilities_fail_closed():
    ops = SimpleNamespace(paged_attention_capabilities=lambda: {})
    assert not any(paged_attention_capabilities(ops).values())
