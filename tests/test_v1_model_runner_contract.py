# SPDX-License-Identifier: Apache-2.0
"""Lightweight contract tests for MetalModelRunner (no model load)."""

from types import SimpleNamespace

import pytest


def test_execute_model_rejects_intermediate_tensors() -> None:
    """``execute_model`` accepts the ``intermediate_tensors`` arg vLLM's Ray DAG
    passes, but rejects non-None (pipeline-parallel) values — Metal has no
    cross-stage activation hand-off. The guard runs before any ``self`` access,
    so an uninitialized instance is enough to exercise it.
    """
    from vllm_metal.v1.model_runner import MetalModelRunner

    runner = object.__new__(MetalModelRunner)
    with pytest.raises(NotImplementedError, match="pipeline-parallel"):
        runner.execute_model(SimpleNamespace(), intermediate_tensors=SimpleNamespace())
