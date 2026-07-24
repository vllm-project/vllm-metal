# SPDX-License-Identifier: Apache-2.0
"""Opt-in greedy golden-token parity for the local GGUF serve path.

Loads a real local ``.gguf`` and greedy-decodes a fixed prompt across two slow
checks, parameterized over the supported GGUF model families:

- ``test_gguf_greedy_matches_committed_golden`` pins the first ``N`` token ids
  to a committed golden sequence — a deterministic regression guard for the
  served quantized model.
- ``test_gguf_greedy_matches_dense_reference_prefix`` asserts the leading
  tokens match an ``mlx_lm`` dense reference of the same checkpoint (loaded at
  its native dtype), proving the quantized model decodes the same
  high-confidence prefix as the float reference. The quantized and dense
  decodes may diverge after that prefix (expected from quantization), so only
  each family's agreeing prefix is asserted.

These load the GGUF directly through ``GGUFModelLoader``; the full
``vllm serve`` HTTP smoke through the Metal paged runner lives in
``tools/gguf_serve_smoke.py``. Slow and env-gated (each family skips unless its
model paths are set); kept OUT of the deterministic unit files (real
checkpoints, not synthetic fixtures). Run a family with its env trio, e.g.::

    VLLM_METAL_MEMORY_FRACTION=0.5 \\
    VLLM_METAL_TEST_GGUF_SERVE_PATH=<...Qwen3-0.6B-Q8_0.gguf> \\
    VLLM_METAL_TEST_GGUF_TOKENIZER_PATH=<...Qwen3-0.6B dir> \\
    VLLM_METAL_TEST_GGUF_DENSE_PATH=<...Qwen3-0.6B dir> \\
    pytest tests/test_gguf_serve_golden.py -m slow

(Qwen3 Q4_1 uses ``VLLM_METAL_TEST_GGUF_Q41_*``, llama uses
``VLLM_METAL_TEST_GGUF_LLAMA_*``, and mistral uses
``VLLM_METAL_TEST_GGUF_MISTRAL_*`` for the same trio.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

pytest.importorskip("gguf")

from mlx_lm import load as mlx_lm_load  # noqa: E402

from vllm_metal.gguf.loader import GGUFModelLoader  # noqa: E402

_DEFAULT_PROMPT = "The capital of France is"
_N = 16


@dataclass(frozen=True)
class _GoldenCase:
    """One GGUF family's committed golden material and env-gated local paths."""

    label: str
    quantization: str
    prompt: str
    gguf_env: str
    tokenizer_env: str
    dense_env: str
    golden_ids: tuple[int, ...]
    dense_agree_prefix: int
    target_dtype: mx.Dtype


_CASES = [
    # Qwen3-0.6B-Q8_0.gguf: the leading 12 ids match the dense reference and
    # diverge at index 12 (quantized vs dense); the asserted prefix stays
    # safely inside that agreement.
    _GoldenCase(
        label="qwen3",
        quantization="Q8_0",
        prompt=_DEFAULT_PROMPT,
        gguf_env="VLLM_METAL_TEST_GGUF_SERVE_PATH",
        tokenizer_env="VLLM_METAL_TEST_GGUF_TOKENIZER_PATH",
        dense_env="VLLM_METAL_TEST_GGUF_DENSE_PATH",
        golden_ids=(
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
        ),
        dense_agree_prefix=8,
        target_dtype=mx.bfloat16,
    ),
    # Qwen_Qwen3-0.6B-Q4_1.gguf: all 16 ids match the dense reference for this
    # high-confidence factual sequence. The file has a tied, redundant Q6_K
    # output.weight; the loader correctly skips it and uses the Q4_1 embedding
    # table as the tied output projection.
    _GoldenCase(
        label="qwen3-q4_1",
        quantization="Q4_1",
        prompt="The capital of France is Paris. The capital of Germany is",
        gguf_env="VLLM_METAL_TEST_GGUF_Q41_SERVE_PATH",
        tokenizer_env="VLLM_METAL_TEST_GGUF_Q41_TOKENIZER_PATH",
        dense_env="VLLM_METAL_TEST_GGUF_Q41_DENSE_PATH",
        golden_ids=(
            19846,
            13,
            576,
            6722,
            315,
            15344,
            374,
            21718,
            13,
            576,
            6722,
            315,
            6323,
            374,
            26194,
            13,
        ),
        dense_agree_prefix=16,
        target_dtype=mx.bfloat16,
    ),
    # Llama-3.2-1B-Instruct-Q8_0.gguf: agreement through index 12, divergence
    # at 13. Without the llama.cpp q/k RoPE row-un-permutation this prefix
    # agreement is only 1 token (attention is broken), so this pins the fix
    # end-to-end.
    _GoldenCase(
        label="llama",
        quantization="Q8_0",
        prompt=_DEFAULT_PROMPT,
        gguf_env="VLLM_METAL_TEST_GGUF_LLAMA_SERVE_PATH",
        tokenizer_env="VLLM_METAL_TEST_GGUF_LLAMA_TOKENIZER_PATH",
        dense_env="VLLM_METAL_TEST_GGUF_LLAMA_DENSE_PATH",
        golden_ids=(
            12366,
            13,
            578,
            469,
            3168,
            301,
            22703,
            374,
            7559,
            304,
            12366,
            13,
            578,
            9928,
            49606,
            16730,
        ),
        dense_agree_prefix=13,
        target_dtype=mx.bfloat16,
    ),
    # Mistral-7B-Instruct-v0.3-Q8_0.gguf: the file declares
    # general.architecture="llama" (mistral converts under the llama arch, so
    # the q/k RoPE un-permutation applies through the adapter's config-side
    # mapping); the config dir says model_type="mistral" and omits head_dim.
    # All 16 ids match the dense reference inside the window.
    _GoldenCase(
        label="mistral",
        quantization="Q8_0",
        prompt=_DEFAULT_PROMPT,
        gguf_env="VLLM_METAL_TEST_GGUF_MISTRAL_SERVE_PATH",
        tokenizer_env="VLLM_METAL_TEST_GGUF_MISTRAL_TOKENIZER_PATH",
        dense_env="VLLM_METAL_TEST_GGUF_MISTRAL_DENSE_PATH",
        golden_ids=(
            6233,
            29493,
            1330,
            1040,
            6333,
            1070,
            1040,
            5717,
            4610,
            1117,
            28566,
            4573,
            29491,
            781,
            781,
            1782,
        ),
        dense_agree_prefix=16,
        target_dtype=mx.bfloat16,
    ),
]
_CASE_IDS = [case.label for case in _CASES]


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
@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_gguf_greedy_matches_committed_golden(case: _GoldenCase) -> None:
    gguf_path = _require_path(case.gguf_env)
    tokenizer_dir = _require_path(case.tokenizer_env)

    model, tokenizer = GGUFModelLoader(
        gguf_path, config_dir=tokenizer_dir, target_dtype=case.target_dtype
    ).load()

    ids = _greedy_token_ids(model, tokenizer, case.prompt, _N)
    assert ids == list(case.golden_ids)


@pytest.mark.slow
@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_gguf_greedy_matches_dense_reference_prefix(case: _GoldenCase) -> None:
    gguf_path = _require_path(case.gguf_env)
    tokenizer_dir = _require_path(case.tokenizer_env)
    dense_dir = _require_path(case.dense_env)

    dense_model, dense_tokenizer = mlx_lm_load(dense_dir)
    dense_ids = _greedy_token_ids(
        dense_model, dense_tokenizer, case.prompt, case.dense_agree_prefix
    )
    del dense_model
    mx.clear_cache()

    gguf_model, gguf_tokenizer = GGUFModelLoader(
        gguf_path, config_dir=tokenizer_dir, target_dtype=case.target_dtype
    ).load()
    gguf_ids = _greedy_token_ids(
        gguf_model, gguf_tokenizer, case.prompt, case.dense_agree_prefix
    )

    assert gguf_ids == dense_ids, (
        f"{case.label} {case.quantization} GGUF greedy prefix must match the "
        f"dense mlx_lm reference (gguf={gguf_ids}, dense={dense_ids})"
    )
    assert gguf_ids == list(case.golden_ids)[: case.dense_agree_prefix]
