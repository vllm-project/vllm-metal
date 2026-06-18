# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-checkpoint tests for the GGUF loader.

These depend on local ``.gguf`` files plus companion config/tokenizer dirs and are
env-gated, so they live apart from the deterministic, repo-owned contract suite in
``test_gguf_loader.py``. They provide the real-file evidence (greedy parity against
committed golden ids, and arch/qtype fail-fast on real hybrid/K-quant files).
"""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

gguf = pytest.importorskip("gguf")

from vllm_metal.gguf.loader import GGUFLoadError, load_gguf_model  # noqa: E402

_QWEN3_GOLDEN_PROMPT = "The capital of France is"
# Greedy (temp=0) continuation of the prompt from the dense Qwen/Qwen3-0.6B (bf16)
# reference, captured once via mlx_lm; the Q8_0 GGUF load must reproduce these ids
# exactly. Regenerate by greedy-decoding the dense model for len(...) tokens.
_QWEN3_GOLDEN_TOKENS = [
    12095, 13, 576, 6722, 315, 9625, 374, 1083, 279, 6722,
    315, 279, 5429, 315, 9625, 13, 576, 6722, 315, 9625,
    374, 1083, 279, 6722, 315, 279, 7513, 9145, 13, 576,
]  # fmt: skip

_QWEN3_GGUF = os.environ.get("VLLM_METAL_TEST_QWEN3_GGUF")
_QWEN3_DENSE_DIR = os.environ.get("VLLM_METAL_TEST_QWEN3_DENSE_DIR")
_QWEN35_GGUF = os.environ.get("VLLM_METAL_TEST_QWEN35_GGUF")
_QWEN35_DIR = os.environ.get("VLLM_METAL_TEST_QWEN35_DIR")
_Q4KM_GGUF = os.environ.get("VLLM_METAL_TEST_Q4KM_GGUF")


def _has_gguf_wrappers(model: nn.Module) -> bool:
    found: set[str] = set()

    def walk(module: nn.Module) -> None:
        for child in module.children().values():
            for leaf in child if isinstance(child, list) else [child]:
                if isinstance(leaf, nn.Module):
                    found.add(type(leaf).__name__)
                    walk(leaf)

    walk(model)
    return {"GGUFLinear", "GGUFEmbedding"} <= found


@pytest.mark.skipif(
    not (_QWEN35_GGUF and _QWEN35_DIR),
    reason="set VLLM_METAL_TEST_QWEN35_GGUF + _DIR to a real Qwen3.5 hybrid GGUF",
)
def test_real_qwen35_rejected_as_non_dense():
    # Qwen3.5 (GGUF arch 'qwen35', a linear-attention hybrid) is rejected by the
    # dense allowlist — an arch-level reject, not a qtype error (the file is Q8_0).
    with pytest.raises(
        GGUFLoadError, match="'qwen35' is not a supported dense decoder"
    ):
        load_gguf_model(_QWEN35_GGUF, config_dir=_QWEN35_DIR, target_dtype=mx.float32)


@pytest.mark.skipif(
    not (_Q4KM_GGUF and _QWEN3_DENSE_DIR),
    reason="set VLLM_METAL_TEST_Q4KM_GGUF + _QWEN3_DENSE_DIR to a real K-quant GGUF",
)
def test_real_k_quant_rejected_unsupported_qtype():
    with pytest.raises(GGUFLoadError, match="Unsupported qtype"):
        load_gguf_model(
            _Q4KM_GGUF, config_dir=_QWEN3_DENSE_DIR, target_dtype=mx.float32
        )


@pytest.mark.slow
@pytest.mark.skipif(
    not (_QWEN3_GGUF and _QWEN3_DENSE_DIR),
    reason="set VLLM_METAL_TEST_QWEN3_GGUF + _QWEN3_DENSE_DIR",
)
def test_load_real_gguf_generates_with_parity():
    model, tokenizer = load_gguf_model(
        _QWEN3_GGUF, config_dir=_QWEN3_DENSE_DIR, target_dtype=mx.bfloat16
    )
    assert _has_gguf_wrappers(model)  # production wrappers actually installed

    prompt = mx.array(tokenizer.encode(_QWEN3_GOLDEN_PROMPT))
    sampler = make_sampler(temp=0.0)
    tokens = [
        int(tok)
        for (tok, _), _ in zip(
            generate_step(prompt, model, sampler=sampler),
            range(len(_QWEN3_GOLDEN_TOKENS)),
            strict=False,
        )
    ]
    assert tokens == _QWEN3_GOLDEN_TOKENS
