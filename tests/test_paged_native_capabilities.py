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
    }
    query.assert_called_once_with()


def test_missing_binding_directs_to_a_rebuild():
    # Unstamped artifacts can still load, so a missing binding points at the
    # remedy rather than surfacing a bare AttributeError.
    with pytest.raises(RuntimeError, match="rebuild.*native extension"):
        paged_attention_capabilities(SimpleNamespace())


def test_binding_errors_are_not_reported_as_missing():
    query = MagicMock(side_effect=AttributeError("binding failed"))

    with pytest.raises(AttributeError, match="binding failed"):
        paged_attention_capabilities(
            SimpleNamespace(paged_attention_capabilities=query)
        )


def test_missing_capabilities_fail_closed():
    ops = SimpleNamespace(paged_attention_capabilities=lambda: {})
    assert not any(paged_attention_capabilities(ops).values())
