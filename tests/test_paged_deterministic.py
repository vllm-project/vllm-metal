# SPDX-License-Identifier: Apache-2.0
"""Deterministic smoke test: vLLM offline inference with golden token comparison.

Golden token IDs were generated using vLLM offline inference with
temperature=0 (greedy decoding) on Qwen/Qwen3-0.6B, running one sequence
at a time (max_num_seqs=1) to avoid batch-invariance issues on Metal.

Ground truth: the MLX inline-cache path (full-precision MLX softmax).
The paged-attention path is expected to converge on it numerically. After
the v2 softmax exp→exp2/log2 fold landed, 5/6 prompts produce the same
argmax tokens as MLX through 10 decode steps.

The remaining prompt ("One plus one equals") has top-2 logits at the
divergence step within ULP (',' vs '.'). Either greedy answer is valid
English; the split is inherent to model rounding behavior at that token,
not a kernel bug. We keep a paged-specific fallback for that one prompt.

Run (paged KV path, the default):
    python -m pytest tests/test_paged_deterministic.py -v -s -m slow

To test the MLX inline cache path instead, pass env vars explicitly:
    VLLM_METAL_USE_PAGED_ATTENTION=0 VLLM_METAL_MEMORY_FRACTION=auto \
        python -m pytest tests/test_paged_deterministic.py -v -s -m slow

Note: MLX requires VLLM_METAL_MEMORY_FRACTION=auto (numeric fractions are
only valid for the paged attention path).
"""

from __future__ import annotations

import os

import pytest
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 10
DEFAULT_USE_PAGED_ATTENTION = "1"
DEFAULT_PAGED_MEMORY_FRACTION = "0.2"
DEFAULT_MLX_MEMORY_FRACTION = "auto"

PROMPTS = [
    "The capital of France is",
    "The weather today is not",
    "One plus one equals",
    "The largest planet in our solar system is",
    "Water boils at a temperature of",
    "Machine learning is",
]

# fmt: off
# Ground truth: MLX inline cache (greedy, full precision). Regenerate via:
#   VLLM_ENABLE_V1_MULTIPROCESSING=0 \
#     python tools/gen_golden_token_ids_for_deterministics.py
# Environment: mlx 0.31.1, mlx-lm 0.31.1
GOLDEN_MLX = {
    "The capital of France is":                   [12095, 13, 576, 6722, 315, 9625, 374, 1083, 279, 6722],
    "The weather today is not":                   [1661, 13, 576, 9315, 374, 220, 17, 15, 12348, 13],
    "One plus one equals":                        [825, 11, 825, 5519, 825, 16819, 1378, 13, 2055, 11],
    "The largest planet in our solar system is":  [1112, 30, 362, 13, 43562, 425, 13, 48976, 356, 13],
    "Water boils at a temperature of":            [220, 16, 15, 15, 30937, 13, 3555, 374, 279, 9315],
    "Machine learning is":                        [264, 7988, 5392, 429, 702, 13791, 1506, 279, 2070, 315],
}

# Per-prompt fallback for prompts whose top-2 logits at the divergence
# step are within ULP. Paged-path rounding can legitimately pick a
# different argmax than MLX for those prompts. Listed here as documented
# exceptions; the test accepts either the MLX golden or this fallback.
GOLDEN_PAGED_FALLBACK = {
    "One plus one equals":                        [825, 13, 3776, 5519, 825, 16819, 1378, 13, 3776, 5519],
}
# fmt: on


def _setenv_default(mp: pytest.MonkeyPatch, key: str, default: str) -> str:
    """Set an env var only when absent and return the effective value."""
    value = os.environ.get(key)
    if value is None:
        mp.setenv(key, default)
        return default
    return value


@pytest.fixture(autouse=True, scope="module")
def _set_env():
    """Set default env vars for this test.

    Uses MonkeyPatch.context() so env changes are automatically reverted
    after the module, avoiding side effects on other tests.

    Defaults to the paged KV cache path to ensure the test actually exercises
    the paged attention kernel, but respects any env vars already set by the
    user (e.g. to run the MLX path).
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        # Default to paged attention, but allow explicit caller override.
        use_paged = _setenv_default(
            mp,
            "VLLM_METAL_USE_PAGED_ATTENTION",
            DEFAULT_USE_PAGED_ATTENTION,
        )

        # Choose a path-specific memory default, while preserving caller override.
        memory_default = (
            DEFAULT_PAGED_MEMORY_FRACTION
            if use_paged == "1"
            else DEFAULT_MLX_MEMORY_FRACTION
        )
        _setenv_default(mp, "VLLM_METAL_MEMORY_FRACTION", memory_default)
        yield


@pytest.fixture(scope="module")
def vllm_outputs():
    """Run vLLM offline inference once for all prompts.

    Pinned to ``enable_prefix_caching=False`` so the golden token IDs
    (cache-off reference) remain the invariant under test regardless of
    upstream default changes.
    """
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=512,
        max_num_seqs=1,
        enable_prefix_caching=False,
    )

    if os.environ.get("VLLM_METAL_USE_PAGED_ATTENTION", "0") == "1":
        runner = llm.llm_engine.model_executor.driver_worker.model_runner
        assert runner._paged_attention_backend is not None, (
            "Paged attention backend not initialised"
        )
        from vllm_metal.metal_kernel_backend.paged_attention import (
            MetalKernelPagedAttentionWrapper,
        )

        attn = runner.model.model.layers[0].self_attn
        assert isinstance(attn, MetalKernelPagedAttentionWrapper)

    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    outputs = llm.generate(PROMPTS, sp)
    return {o.prompt: o for o in outputs}


class TestPagedDeterministic:
    @pytest.mark.slow
    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_generate_matches_golden(self, vllm_outputs, prompt):
        output = vllm_outputs[prompt]
        token_ids = list(output.outputs[0].token_ids)
        text = output.outputs[0].text

        mlx_expected = GOLDEN_MLX[prompt]
        fallback = GOLDEN_PAGED_FALLBACK.get(prompt)

        print(
            f"VLLM_METAL_USE_PAGED_ATTENTION: {os.environ.get('VLLM_METAL_USE_PAGED_ATTENTION')}"
        )
        print(f"\n  prompt: {prompt!r}")
        print(f"  output: {text!r}")
        print(f"  ids:    {token_ids}")

        if token_ids == mlx_expected:
            print("  result: MATCHED MLX ground truth")
            return
        if fallback is not None and token_ids == fallback:
            print("  result: MATCHED paged fallback (documented divergence)")
            return

        print("  result: NO MATCH")
        print(f"  expected (MLX):   {mlx_expected}")
        if fallback is not None:
            print(f"  fallback (paged): {fallback}")

        msg = (
            f"Output for {prompt!r} did not match the MLX ground truth"
            f"{' or its paged fallback' if fallback is not None else ''}.\n"
            f"Got:              {token_ids}\n"
            f"Expected (MLX):   {mlx_expected}\n"
        )
        if fallback is not None:
            msg += f"Fallback (paged): {fallback}\n"
        raise AssertionError(msg)
