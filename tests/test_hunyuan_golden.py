# SPDX-License-Identifier: Apache-2.0
"""Deterministic golden-token test for Hunyuan on the paged attention path.

Verifies that vllm-metal's paged attention applies Hunyuan's per-head Q/K
norms.  ``hunyuan_v1_dense`` spells them ``query_layernorm`` /
``key_layernorm`` rather than the ``q_norm`` / ``k_norm`` used by Qwen3 and
Gemma4, so a name-only probe skips them silently: the model loads, serves,
and returns HTTP 200 while generating wrong tokens.  This test is the
regression guard for that failure mode, which produces no error to catch.

Two golden sets are kept:

- ``GOLDEN_MLX_LM``: tokens produced by running the model directly through
  ``mlx_lm.stream_generate`` (greedy, EOS disabled). The independent
  reference, captured outside vllm-metal.
- ``GOLDEN_PAGED``: tokens produced by vllm-metal's paged attention path on
  the same model/prompts. Captured here so small floating-point tie-break
  drifts don't cause spurious failures.

This dual-golden pattern mirrors ``tests/test_gemma4_golden.py``.  A
module-level invariant enforces that the two sets agree on at least
``MAX_TOKENS - 1`` of ``MAX_TOKENS`` token IDs for every prompt — the paged
path cannot drift from mlx_lm by more than a single token.

At capture time the two sets were **byte-identical on all five prompts**.
With the ``query_layernorm`` / ``key_layernorm`` handling removed, the paged
path diverges from ``mlx_lm`` at token 0 on four of the five prompts (and
token 1 on the fifth), so this test fails loudly without the fix.

Enable by pointing ``HUNYUAN_MODEL_PATH`` at a local MLX checkpoint:

    HUNYUAN_MODEL_PATH=/path/to/Hunyuan-1.8B-Instruct-4bit \\
        pytest tests/test_hunyuan_golden.py -v -s -m slow

Captured with mlx-community/Hunyuan-1.8B-Instruct-4bit @ 5a121aef6911,
mlx 0.32.0, mlx-lm 0.31.3, vllm 0.28.0+cpu on an Apple M2 / 8 GB.

Regenerate goldens with:
    # mlx_lm reference
    python tools/gen_hunyuan_golden.py <model-path>
    # paged-path reference (engine must run)
    VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_METAL_USE_PAGED_ATTENTION=1 \\
        python tools/gen_hunyuan_golden.py --paged <model-path>
"""

from __future__ import annotations

import gc
import os

import pytest
from vllm import LLM, SamplingParams

MAX_TOKENS = 10

MODEL_ENV = "HUNYUAN_MODEL_PATH"
MEMORY_FRACTION = "0.6"
MAX_MODEL_LEN = 512

PROMPTS = [
    "The capital of France is",
    "The weather today is not",
    "One plus one equals",
    "The largest planet in our solar system is",
    "Water boils at a temperature of",
]

# fmt: off
# mlx-community/Hunyuan-1.8B-Instruct-4bit @ 5a121aef6911, mlx-lm 0.31.3.
GOLDEN_MLX_LM = {
    "The capital of France is":                   [316, 316, 316, 316, 316, 316, 206, 206, 206, 206],
    "The weather today is not":                   [575, 575, 575, 575, 575, 575, 575, 575, 575, 575],
    "One plus one equals":                        [926, 926, 926, 926, 926, 926, 926, 926, 926, 926],
    "The largest planet in our solar system is":  [316, 316, 316, 316, 316, 316, 316, 316, 316, 316],
    "Water boils at a temperature of":            [244, 244, 244, 244, 244, 244, 244, 244, 244, 244],
}

# Identical to GOLDEN_MLX_LM at capture time (no fp tie-break drift observed).
GOLDEN_PAGED = {
    "The capital of France is":                   [316, 316, 316, 316, 316, 316, 206, 206, 206, 206],
    "The weather today is not":                   [575, 575, 575, 575, 575, 575, 575, 575, 575, 575],
    "One plus one equals":                        [926, 926, 926, 926, 926, 926, 926, 926, 926, 926],
    "The largest planet in our solar system is":  [316, 316, 316, 316, 316, 316, 316, 316, 316, 316],
    "Water boils at a temperature of":            [244, 244, 244, 244, 244, 244, 244, 244, 244, 244],
}
# fmt: on


_MIN_COMMON_PREFIX = MAX_TOKENS - 1


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if x != y:
            return i
    return len(a)


def test_golden_pairs_consistent() -> None:
    """Catch silent golden drift between the mlx_lm and paged goldens.

    Both paths are greedy; once the paged output diverges from mlx_lm on one
    token, every subsequent token is generated from a different KV context,
    so the tails rarely re-converge.  The meaningful invariant is the length
    of the common prefix — require agreement on at least
    ``_MIN_COMMON_PREFIX`` of ``MAX_TOKENS`` tokens.

    Runs as a plain pytest test (no model needed) so failures surface as
    test results instead of import-time errors.
    """
    for prompt in PROMPTS:
        mlx_ids = GOLDEN_MLX_LM[prompt]
        paged_ids = GOLDEN_PAGED[prompt]
        assert len(mlx_ids) == MAX_TOKENS, (
            f"mlx_lm golden for {prompt!r} has {len(mlx_ids)} tokens, "
            f"expected {MAX_TOKENS}"
        )
        assert len(paged_ids) == MAX_TOKENS, (
            f"paged golden for {prompt!r} has {len(paged_ids)} tokens, "
            f"expected {MAX_TOKENS}"
        )
        prefix = _common_prefix_len(mlx_ids, paged_ids)
        assert prefix >= _MIN_COMMON_PREFIX, (
            f"Hunyuan golden drift on {prompt!r}: paged agrees with mlx_lm "
            f"on the first {prefix} of {MAX_TOKENS} tokens "
            f"(required >= {_MIN_COMMON_PREFIX}). "
            f"mlx_lm={mlx_ids} paged={paged_ids}"
        )


_NO_MODEL_REASON = (
    f"Set {MODEL_ENV} to a local Hunyuan MLX checkpoint to run the "
    "end-to-end golden test."
)


@pytest.fixture(scope="class")
def _paged_env_for_golden_class():
    """Single-process deterministic paged-attention env, scoped to the
    end-to-end test class only.

    ``test_golden_pairs_consistent`` is pure data validation and does not
    need these env vars, so the fixture is NOT module-scoped.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        mp.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        yield


def _run_model() -> dict[str, list[int]]:
    model_path = os.environ[MODEL_ENV]
    if not os.path.isdir(model_path):
        pytest.skip(f"{MODEL_ENV}={model_path} is not a directory")

    from vllm_metal.config import reset_config

    previous = os.environ.get("VLLM_METAL_MEMORY_FRACTION")
    os.environ["VLLM_METAL_MEMORY_FRACTION"] = MEMORY_FRACTION
    reset_config()
    gc.collect()
    try:
        llm = LLM(
            model=model_path,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=1,
            disable_log_stats=True,
        )
        sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS, ignore_eos=True)
        outputs = llm.generate(PROMPTS, sp)
    finally:
        if previous is None:
            os.environ.pop("VLLM_METAL_MEMORY_FRACTION", None)
        else:
            os.environ["VLLM_METAL_MEMORY_FRACTION"] = previous
        reset_config()
        gc.collect()

    return {o.prompt: list(o.outputs[0].token_ids) for o in outputs}


@pytest.fixture(scope="module")
def model_outputs() -> dict[str, dict[str, list[int]]]:
    """Lazily run the model once and cache the tokens."""
    return {}


def _model_tokens(
    model_outputs: dict[str, dict[str, list[int]]],
) -> dict[str, list[int]]:
    if "hunyuan" not in model_outputs:
        model_outputs["hunyuan"] = _run_model()
    return model_outputs["hunyuan"]


@pytest.mark.skipif(not os.environ.get(MODEL_ENV), reason=_NO_MODEL_REASON)
@pytest.mark.usefixtures("_paged_env_for_golden_class")
class TestHunyuanGolden:
    @pytest.mark.slow
    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_matches_golden(
        self,
        prompt: str,
        model_outputs: dict[str, dict[str, list[int]]],
    ) -> None:
        """Paged output must match one of the two goldens.

        Without ``query_layernorm`` / ``key_layernorm`` handling in
        ``prepare_sdpa_qkv`` the Q/K norms are skipped, and the paged path
        diverges from both goldens at the first or second token.
        """
        tokens_by_prompt = _model_tokens(model_outputs)
        token_ids = tokens_by_prompt[prompt]

        mlx_expected = GOLDEN_MLX_LM[prompt]
        paged_expected = GOLDEN_PAGED[prompt]

        mlx_match = token_ids == mlx_expected
        paged_match = token_ids == paged_expected

        print(f"\n  prompt: {prompt!r}")
        print(f"  ids:    {token_ids}")
        if mlx_match:
            print("  result: MATCHED mlx_lm golden")
        elif paged_match:
            print("  result: MATCHED paged-path golden")
        else:
            print("  result: NO MATCH")
            print(f"  expected (mlx_lm): {mlx_expected}")
            print(f"  expected (paged):  {paged_expected}")

        assert mlx_match or paged_match, (
            f"Hunyuan output for {prompt!r} matched neither golden.\n"
            f"  actual:            {token_ids}\n"
            f"  expected (mlx_lm): {mlx_expected}\n"
            f"  expected (paged):  {paged_expected}\n"
            "A divergence at the first tokens usually means the per-head "
            "Q/K norms were skipped — check that prepare_sdpa_qkv still "
            "probes query_layernorm/key_layernorm alongside q_norm/k_norm."
        )
