# SPDX-License-Identifier: Apache-2.0
"""Smoke test for unified prefill+decode forward pass (continuous batching).

Runs vLLM offline inference with max_num_seqs > 1 so the scheduler batches
multiple requests together, triggering the unified forward pass where prefill
and decode happen in a single model call.

Due to floating-point non-determinism when batching on Metal (MLX GEMM uses
different internal kernels for different batch sizes), exact golden-token
matching is NOT expected.  Instead, this test:
  1. Verifies all requests complete without errors.
  2. Prints the generated text for manual inspection (not gibberish).
  3. Optionally checks whether outputs still match the single-request golden.

Run:
    python -m pytest tests/test_unified_batching.py -v -s
"""

from __future__ import annotations

import pytest
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 10
MAX_NUM_SEQS = 4  # key: allow concurrent requests

PROMPTS = [
    "The capital of France is",
    "The weather today is not",
    "One plus one equals",
    "The largest planet in our solar system is",
    "Water boils at a temperature of",
    "Machine learning is",
]

# fmt: off
# Golden from max_num_seqs=1 (single-request, deterministic).
# Used only for informational comparison — NOT asserted.
GOLDEN_SINGLE = {
    "The capital of France is":                   [12095, 13, 576, 6722, 315, 9625, 374, 1083, 279, 6722],
    "The weather today is not":                   [1661, 13, 576, 9315, 374, 220, 17, 15, 12348, 13],
    "One plus one equals":                        [825, 11, 825, 5519, 825, 16819, 1378, 13, 2055, 11],
    "The largest planet in our solar system is":  [1112, 30, 362, 13, 43562, 425, 13, 48976, 356, 13],
    "Water boils at a temperature of":            [220, 16, 15, 15, 30937, 13, 3555, 374, 279, 9315],
    "Machine learning is":                        [264, 7988, 5392, 429, 702, 13791, 1506, 279, 2070, 315],
}
# fmt: on


@pytest.fixture(autouse=True, scope="module")
def _set_env():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        mp.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        mp.setenv("VLLM_METAL_MEMORY_FRACTION", "0.2")
        yield


@pytest.fixture(scope="module")
def vllm_outputs():
    """Run vLLM offline inference with concurrent batching."""
    llm = LLM(model=MODEL_NAME, max_model_len=512, max_num_seqs=MAX_NUM_SEQS)

    # Verify paged KV + attention wrapper are active
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    assert runner._paged_kv_cache is not None, "Paged KV cache not initialised"

    from vllm_metal.metal_kernel_backend.paged_attention import (
        MetalKernelPagedAttentionWrapper,
    )

    attn = runner.model.model.layers[0].self_attn
    assert isinstance(attn, MetalKernelPagedAttentionWrapper)

    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    outputs = llm.generate(PROMPTS, sp)
    return {o.prompt: o for o in outputs}


class TestUnifiedBatching:
    @pytest.mark.slow
    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_generate_coherent(self, vllm_outputs, prompt):
        """Verify output is non-empty and print for manual inspection."""
        output = vllm_outputs[prompt]
        token_ids = list(output.outputs[0].token_ids)
        text = output.outputs[0].text

        golden = GOLDEN_SINGLE.get(prompt, [])
        match = token_ids == golden

        print(f"\n  prompt:  {prompt!r}")
        print(f"  output:  {text!r}")
        print(f"  ids:     {token_ids}")
        print(f"  golden:  {golden}")
        print(f"  match:   {'YES' if match else 'no (expected with batching)'}")

        # Basic sanity: output should not be empty
        assert len(token_ids) == MAX_TOKENS, (
            f"Expected {MAX_TOKENS} tokens, got {len(token_ids)}"
        )
        assert len(text.strip()) > 0, "Generated text is empty"
