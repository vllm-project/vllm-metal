# SPDX-License-Identifier: Apache-2.0
"""Deterministic golden-token test for IBM Granite 3.3 on the paged path.

Granite reuses the standard GQA paged attention path. Its four scalar
multipliers (``embedding_multiplier``, ``residual_multiplier``,
``attention_multiplier``, ``logits_scaling``) are applied entirely inside
mlx-lm's Granite model forward (``mlx_lm.models.granite``), so vllm-metal
needs no Granite-specific adapter: the paged path only swaps each layer's
``self_attn``, and ``attention_multiplier`` reaches the Metal kernel via
``inner.scale`` (mlx-lm sets ``Attention.scale = attention_multiplier``).

Two golden sets are kept (mirrors ``tests/test_gemma4_golden.py``):

- ``GOLDEN_MLX_LM``: tokens from running the checkpoint directly through
  ``mlx_lm.stream_generate`` (greedy, chat template, EOS honored). The
  independent reference, captured outside vllm-metal.
- ``GOLDEN_PAGED``: tokens from vllm-metal's paged attention path on the
  same prompts. At capture the paged path was **byte-for-byte identical**
  to the mlx_lm reference (no floating-point tie-break drift), so it is
  kept as an explicit alias. When the paged path first diverges from
  mlx_lm on a tie-break, expand ``GOLDEN_PAGED`` into its own dict holding
  the divergent tokens.

``test_golden_data_wellformed`` (unmarked, so it runs in CI under
``pytest -m "not slow"``) guards the golden tables themselves: it catches
hand-edit typos (missing prompt, wrong length, malformed token id) and
asserts the two tables stay the same shape and share a long common prefix.
The end-to-end ``TestGranitePagedGolden`` (``slow``) loads the 8B
checkpoint and is opt-in only.

Captured with mlx 0.31.x, mlx-lm 0.31.3 on Apple M4 Pro (48 GB), macOS.
Regenerate both via (paged is the default backend on v0.2.0+; pass
``VLLM_METAL_USE_PAGED_ATTENTION=0 VLLM_METAL_MEMORY_FRACTION=auto`` for
the MLX inline-cache cross-check):

    VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
        python tools/gen_golden_token_ids_for_deterministics.py \\
        --model mlx-community/granite-3.3-8b-instruct-4bit \\
        --max-tokens 20 --chat-template

Run the end-to-end test (downloads ~4.7 GB on first run):
    pytest tests/test_granite_golden.py -v -s -m slow
"""

from __future__ import annotations

import pytest
from vllm import LLM, SamplingParams

MODEL_NAME = "mlx-community/granite-3.3-8b-instruct-4bit"
MAX_TOKENS = 20

PROMPTS = [
    "The capital of France is",
    "The weather today is not",
    "One plus one equals",
    "The largest planet in our solar system is",
    "Water boils at a temperature of",
    "Machine learning is",
]

# fmt: off
# Ground truth: independent mlx_lm.stream_generate (greedy, chat template,
# EOS honored — "One plus one equals" terminates early at 18 tokens).
GOLDEN_MLX_LM = {
    "The capital of France is":                   [1318, 18926, 432, 45600, 438, 2716, 297, 32, 2716, 297, 438, 646, 1755, 322, 35566, 1578, 461, 22187, 1353, 4193],
    "The weather today is not":                   [59, 3464, 17636, 436, 322, 26890, 7000, 32, 12619, 844, 4322, 6377, 2769, 14335, 432, 322, 20023, 844, 4484, 16574],
    "One plus one equals":                        [35, 474, 225, 35, 280, 225, 36, 32, 886, 3632, 432, 1591, 10952, 1591, 438, 3134, 32, 0],
    "The largest planet in our solar system is":  [1318, 22909, 34127, 328, 2992, 44909, 2664, 438, 971, 13706, 32, 2030, 1182, 8967, 436, 2819, 24932, 4761, 47017, 30],
    "Water boils at a temperature of":            [35, 34, 34, 18667, 390, 9859, 3263, 308, 36, 35, 36, 18667, 506, 2371, 1575, 30830, 27, 821, 6216, 821],
    "Machine learning is":                        [7090, 9608, 438, 312, 17272, 432, 5549, 31251, 629, 21488, 308, 6218, 27, 688, 35167, 7668, 544, 322, 8226, 432],
}
# fmt: on

# Paged path was byte-for-byte identical to the mlx_lm reference at capture;
# kept as an explicit alias (see module docstring). Split into a separate
# dict only once a real tie-break divergence is observed.
GOLDEN_PAGED = GOLDEN_MLX_LM

# Minimum tokens the paged and mlx_lm goldens must agree on, capped per
# prompt by the prompt's own (possibly EOS-shortened) length.
_MIN_COMMON_PREFIX = 10


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    # Callers assert ``len(a) == len(b)`` first, so strict zip is safe and
    # turns an accidental length mismatch into a loud error.
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if x != y:
            return i
    return len(a)


def test_golden_data_wellformed() -> None:
    """Guard the golden tables themselves (runs in CI; e2e test is slow-only).

    A hand-edit typo while regenerating goldens — a dropped prompt, a
    truncated list, a negative/float token id, or the two tables drifting
    out of sync — fails here fast, instead of silently weakening the
    slow-only end-to-end check that CI never runs.
    """
    assert set(GOLDEN_MLX_LM) == set(PROMPTS), "GOLDEN_MLX_LM key set != PROMPTS"
    assert set(GOLDEN_PAGED) == set(PROMPTS), "GOLDEN_PAGED key set != PROMPTS"

    for prompt in PROMPTS:
        mlx_ids = GOLDEN_MLX_LM[prompt]
        paged_ids = GOLDEN_PAGED[prompt]

        for name, ids in (("mlx_lm", mlx_ids), ("paged", paged_ids)):
            assert 0 < len(ids) <= MAX_TOKENS, (
                f"{name} golden for {prompt!r} has {len(ids)} tokens, "
                f"expected 1..{MAX_TOKENS}"
            )
            assert all(isinstance(t, int) and t >= 0 for t in ids), (
                f"{name} golden for {prompt!r} has a malformed token id: {ids}"
            )

        assert len(paged_ids) == len(mlx_ids), (
            f"Granite golden length mismatch on {prompt!r}: "
            f"mlx_lm={len(mlx_ids)} paged={len(paged_ids)}"
        )

        # Both paths are greedy; once paged diverges from mlx_lm on one
        # token, every later token is decoded from a different KV context,
        # so the meaningful invariant is the common-prefix length, capped
        # by this prompt's (EOS-shortened) length.
        required = min(_MIN_COMMON_PREFIX, len(mlx_ids))
        prefix = _common_prefix_len(mlx_ids, paged_ids)
        assert prefix >= required, (
            f"Granite golden drift on {prompt!r}: paged agrees with mlx_lm "
            f"on the first {prefix} tokens (required >= {required}). "
            f"mlx_lm={mlx_ids} paged={paged_ids}"
        )


@pytest.fixture(scope="module")
def _paged_env():
    """Single-process deterministic paged-attention env for this module.

    Respects a caller-provided ``VLLM_METAL_MEMORY_FRACTION`` (e.g. to run
    on a smaller machine) and only defaults it when unset.
    """
    import os

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        mp.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        if os.environ.get("VLLM_METAL_MEMORY_FRACTION") is None:
            mp.setenv("VLLM_METAL_MEMORY_FRACTION", "0.5")
        yield


@pytest.fixture(scope="module")
def _chat_prompts() -> list[str]:
    """Apply the Granite chat template; return prompts in PROMPTS order.

    Order-preserving (parallel to ``PROMPTS``) so the caller can zip the
    engine outputs back to the original prompts without a reverse map.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in PROMPTS
    ]


@pytest.fixture(scope="module")
def paged_outputs(_paged_env, _chat_prompts) -> dict[str, list[int]]:
    """Run vllm-metal paged inference once for all prompts."""
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=512,
        max_num_seqs=1,
        enable_prefix_caching=False,
    )

    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    assert runner._paged_attention_backend is not None, (
        "Paged attention backend not initialised"
    )
    from vllm_metal.metal_kernel_backend.paged_attention import (
        MetalKernelPagedAttentionWrapper,
    )

    attn = runner.model.model.layers[0].self_attn
    assert isinstance(attn, MetalKernelPagedAttentionWrapper), (
        "Granite layer 0 self_attn was not replaced by the paged wrapper"
    )

    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    outputs = llm.generate(_chat_prompts, sp)
    # generate() preserves input order; zip back to the original prompts.
    return {
        prompt: list(o.outputs[0].token_ids)
        for prompt, o in zip(PROMPTS, outputs, strict=True)
    }


class TestGranitePagedGolden:
    @pytest.mark.slow
    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_matches_golden(
        self, paged_outputs: dict[str, list[int]], prompt: str
    ) -> None:
        token_ids = paged_outputs[prompt]
        mlx_expected = GOLDEN_MLX_LM[prompt]
        paged_expected = GOLDEN_PAGED[prompt]

        mlx_match = token_ids == mlx_expected
        paged_match = token_ids == paged_expected

        print(f"\n  prompt: {prompt!r}")
        print(f"  ids:    {token_ids}")
        if mlx_match:
            print("  result: MATCHED mlx_lm ground truth")
        elif paged_match:
            print("  result: MATCHED paged golden")
        else:
            print("  result: NO MATCH")
            print(f"  expected (mlx_lm): {mlx_expected}")
            print(f"  expected (paged):  {paged_expected}")

        assert mlx_match or paged_match, (
            f"Granite output for {prompt!r} matched neither golden.\n"
            f"Got:               {token_ids}\n"
            f"Expected (mlx_lm): {mlx_expected}\n"
            f"Expected (paged):  {paged_expected}"
        )
