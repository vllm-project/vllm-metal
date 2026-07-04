# SPDX-License-Identifier: Apache-2.0
"""Opt-in greedy golden-token parity for the local GGUF serve path.

Loads a real local ``.gguf`` and greedy-decodes a fixed prompt across two slow
checks:

- ``test_gguf_greedy_matches_committed_golden`` pins the first ``N`` token ids to
  a committed golden sequence — a deterministic regression guard for the served
  Q8_0 model.
- ``test_gguf_greedy_matches_dense_reference_prefix`` asserts the leading tokens
  match an ``mlx_lm`` dense reference of the same checkpoint (loaded at its
  native dtype), proving the quantized model decodes the same high-confidence
  prefix as the float reference. The quantized and dense decodes diverge after
  that prefix (expected from quantization — for Qwen3-0.6B they agree through
  index 11 and diverge at index 12), so only the agreeing prefix is asserted.

These load the GGUF directly through ``GGUFModelLoader``; the full ``vllm serve``
HTTP smoke through the Metal paged runner lives in ``tools/gguf_serve_smoke.py``.
Slow and env-gated (skipped unless the model paths are set); kept OUT of the
deterministic unit files (real checkpoints, not synthetic fixtures). Run with::

    VLLM_METAL_MEMORY_FRACTION=0.5 \\
    VLLM_METAL_TEST_GGUF_SERVE_PATH=<...Qwen3-0.6B-Q8_0.gguf> \\
    VLLM_METAL_TEST_GGUF_TOKENIZER_PATH=<...Qwen3-0.6B dir> \\
    VLLM_METAL_TEST_GGUF_DENSE_PATH=<...Qwen3-0.6B dir> \\
    pytest tests/test_gguf_serve_golden.py -m slow
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

pytest.importorskip("gguf")

from mlx_lm import load as mlx_lm_load  # noqa: E402

from vllm_metal.gguf.loader import GGUFModelLoader  # noqa: E402

_PROMPT = "The capital of France is"
_N = 16
# Greedy decode of _PROMPT through Qwen3-0.6B-Q8_0.gguf (target_dtype=bf16).
# The leading 12 ids match the dense mlx_lm reference; they diverge at index 12
# (quantized vs dense), so the full sequence is pinned here as a GGUF-path
# regression while the prefix is cross-checked against dense below.
_GOLDEN_IDS = [
    12095,
    13,
    576,
    6722,
    315,
    9625,
    374,
    1083,
    279,
    6722,
    315,
    279,
    5429,
    315,
    9625,
    13,
]
# Compared against the dense reference; safely inside the 12-token agreement.
_DENSE_AGREE_PREFIX = 8

_GGUF_ENV = "VLLM_METAL_TEST_GGUF_SERVE_PATH"
_TOK_ENV = "VLLM_METAL_TEST_GGUF_TOKENIZER_PATH"
_DENSE_ENV = "VLLM_METAL_TEST_GGUF_DENSE_PATH"


def _require_path(env: str) -> str:
    value = os.environ.get(env)
    if not value:
        pytest.skip(f"{env} not set")
    if not Path(value).exists():
        pytest.skip(f"{env} path does not exist: {value}")
    return value


def _greedy_token_ids(model: Any, tokenizer: Any, prompt: str, n: int) -> list[int]:
    ids = list(tokenizer.encode(prompt))
    generated: list[int] = []
    for _ in range(n):
        logits = model(mx.array([ids]))[:, -1, :]
        next_id = int(mx.argmax(logits, axis=-1).item())
        generated.append(next_id)
        ids.append(next_id)
    return generated


@pytest.mark.slow
def test_gguf_greedy_matches_committed_golden() -> None:
    gguf_path = _require_path(_GGUF_ENV)
    tokenizer_dir = _require_path(_TOK_ENV)

    model, tokenizer = GGUFModelLoader(
        gguf_path, config_dir=tokenizer_dir, target_dtype=mx.bfloat16
    ).load()

    assert _greedy_token_ids(model, tokenizer, _PROMPT, _N) == _GOLDEN_IDS


@pytest.mark.slow
def test_gguf_greedy_matches_dense_reference_prefix() -> None:
    gguf_path = _require_path(_GGUF_ENV)
    tokenizer_dir = _require_path(_TOK_ENV)
    dense_dir = _require_path(_DENSE_ENV)

    dense_model, dense_tokenizer = mlx_lm_load(dense_dir)
    dense_ids = _greedy_token_ids(
        dense_model, dense_tokenizer, _PROMPT, _DENSE_AGREE_PREFIX
    )
    del dense_model
    mx.clear_cache()

    gguf_model, gguf_tokenizer = GGUFModelLoader(
        gguf_path, config_dir=tokenizer_dir, target_dtype=mx.bfloat16
    ).load()
    gguf_ids = _greedy_token_ids(
        gguf_model, gguf_tokenizer, _PROMPT, _DENSE_AGREE_PREFIX
    )

    assert gguf_ids == dense_ids, (
        "Q8_0 GGUF greedy prefix must match the dense mlx_lm reference "
        f"(gguf={gguf_ids}, dense={dense_ids})"
    )
    assert gguf_ids == _GOLDEN_IDS[:_DENSE_AGREE_PREFIX]
