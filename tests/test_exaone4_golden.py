# SPDX-License-Identifier: Apache-2.0
"""Deterministic golden-token test for EXAONE 4.0 on the paged attention path.

EXAONE 4.0 1.2B (``LGAI-EXAONE/EXAONE-4.0-1.2B``) is a plain GQA model: 30
layers, every layer ``full_attention`` (no sliding window), 32 attention heads /
8 KV heads, head_dim=64, tied word embeddings, llama3 RoPE scaling. It therefore
exercises the same standard "GQA (paged)" path as Llama 3 / Qwen3 — no MLA, GDN,
YOCO, or per-layer KV cache shapes. head_dim=64 is instantiated in every
paged-attention kernel (prefill tiled, per-token decode, split-KV reduce), so no
kernel work is required to bring the model up; this test only locks the result.

Two golden sets are kept (mirrors ``tests/test_gemma4_golden.py``):

- ``GOLDEN_MLX_LM``: tokens from running the checkpoint directly through
  ``mlx_lm.stream_generate`` (greedy, EOS disabled) — the independent reference
  captured outside vllm-metal.
- ``GOLDEN_PAGED``: tokens from vllm-metal's paged attention path on the same
  prompts. Captured so small floating-point tie-break drifts between the two
  paths don't cause spurious failures.

A data-only invariant (``test_golden_pairs_consistent``) enforces that the two
sets agree on at least ``_MIN_COMMON_PREFIX`` of ``MAX_TOKENS`` token IDs for
every prompt, bounding how far the in-tree paged golden may diverge from the
independent mlx_lm reference.

Regenerate goldens (after ``./install.sh`` and downloading the checkpoint):

    # mlx_lm reference:
    python tools/gen_exaone4_golden.py /path/to/exaone-4.0-1.2b-4bit
    # paged path:
    python tools/gen_exaone4_golden.py --paged /path/to/exaone-4.0-1.2b-4bit

Then paste the printed dicts into ``GOLDEN_MLX_LM`` / ``GOLDEN_PAGED`` below.

Run the end-to-end test (needs local weights):

    EXAONE4_MODEL_PATH=/path/to/exaone-4.0-1.2b-4bit \\
        pytest tests/test_exaone4_golden.py -v -s -m slow
"""

from __future__ import annotations

import gc
import os

import pytest
from vllm import LLM, SamplingParams

MODEL_ENV = "EXAONE4_MODEL_PATH"
MEMORY_FRACTION = "0.5"
MAX_MODEL_LEN = 512
MAX_TOKENS = 10

# A mix of English and Korean prompts: EXAONE 4.0 is a Korean/English bilingual
# model, so the Korean prompts also exercise multi-byte tokenization through the
# paged path. Keep these IN SYNC with tools/gen_exaone4_golden.py — the dict
# keys below are matched by prompt string.
PROMPTS = [
    "One plus one equals",
    "Two plus two equals",
    "Monday, Tuesday, Wednesday,",
    "서울은 대한민국의",
    "인공지능은",
]

# fmt: off
# ---- Goldens ------------------------------------------------------------
# Captured with mlx-community/exaone-4.0-1.2b-4bit (mlx 0.31.2, mlx-lm 0.31.3,
# transformers 5.12.1) via tools/gen_exaone4_golden.py. For this prompt set the
# paged path is bit-for-bit identical to the mlx_lm reference (no fp tie-break
# drift), so both dicts match; they are kept separate per the framework
# convention (tests/test_gemma4_golden.py) since future kernel changes may
# introduce drift on the paged path while it should keep matching mlx_lm.
GOLDEN_MLX_LM: dict[str, list[int]] = {
    "One plus one equals":         [1300, 22055, 1300, 22055, 1300, 22055, 1300, 22055, 1300, 22055],
    "Two plus two equals":         [1655, 3058, 1655, 3058, 1655, 3058, 1655, 3058, 1655, 3058],
    "Monday, Tuesday, Wednesday,": [13822, 373, 13822, 373, 13822, 373, 13822, 373, 13822, 373],
    "서울은 대한민국의":              [361, 361, 361, 361, 379, 379, 379, 379, 379, 379],
    "인공지능은":                    [3885, 3885, 3885, 3885, 3885, 720, 902, 902, 902, 902],
}
GOLDEN_PAGED: dict[str, list[int]] = {
    "One plus one equals":         [1300, 22055, 1300, 22055, 1300, 22055, 1300, 22055, 1300, 22055],
    "Two plus two equals":         [1655, 3058, 1655, 3058, 1655, 3058, 1655, 3058, 1655, 3058],
    "Monday, Tuesday, Wednesday,": [13822, 373, 13822, 373, 13822, 373, 13822, 373, 13822, 373],
    "서울은 대한민국의":              [361, 361, 361, 361, 379, 379, 379, 379, 379, 379],
    "인공지능은":                    [3885, 3885, 3885, 3885, 3885, 720, 902, 902, 902, 902],
}
# fmt: on

_MIN_COMMON_PREFIX = 8


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if x != y:
            return i
    return len(a)


def test_golden_pairs_consistent() -> None:
    """Catch silent golden drift between the mlx_lm and paged goldens.

    Both paths are greedy; once the paged output diverges from mlx_lm on one
    token, every later token is generated from a different KV context, so the
    meaningful invariant is the common-prefix length, not the total diff count.
    Skips while goldens are unpopulated (no local weights at capture time).
    """
    if not GOLDEN_MLX_LM or not GOLDEN_PAGED:
        pytest.skip("EXAONE4 goldens not populated")

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
            f"EXAONE4 golden drift on {prompt!r}: paged agrees with mlx_lm on "
            f"the first {prefix} of {MAX_TOKENS} tokens "
            f"(required >= {_MIN_COMMON_PREFIX}). "
            f"mlx_lm={mlx_ids} paged={paged_ids}"
        )


_MODEL_PATH = os.environ.get(MODEL_ENV)
_HAVE_GOLDENS = bool(GOLDEN_MLX_LM and GOLDEN_PAGED)
_SKIP_REASON = (
    f"Set {MODEL_ENV}=/path/to/exaone-4.0-1.2b-4bit and populate GOLDEN_MLX_LM "
    f"/ GOLDEN_PAGED to run the end-to-end golden test."
)


@pytest.fixture(scope="module")
def _paged_env():
    """Single-process deterministic paged-attention env for the e2e test."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        mp.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        mp.setenv("VLLM_METAL_MEMORY_FRACTION", MEMORY_FRACTION)
        yield


@pytest.fixture(scope="module")
def _paged_tokens(_paged_env) -> dict[str, list[int]]:
    """Run the paged path once and cache {prompt: token_ids} for the module."""
    from vllm_metal.config import reset_config
    from vllm_metal.v1.model_lifecycle import reset_model_cache

    model_path = os.environ[MODEL_ENV]
    reset_model_cache()
    reset_config()
    gc.collect()
    try:
        llm = LLM(model=model_path, max_model_len=MAX_MODEL_LEN, max_num_seqs=1)
        sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS, ignore_eos=True)
        outputs = llm.generate(PROMPTS, sp)
    finally:
        reset_model_cache()
        reset_config()
        gc.collect()
    return {o.prompt: list(o.outputs[0].token_ids) for o in outputs}


@pytest.mark.slow
@pytest.mark.skipif(not _MODEL_PATH or not _HAVE_GOLDENS, reason=_SKIP_REASON)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_matches_golden(prompt: str, _paged_tokens: dict[str, list[int]]) -> None:
    """The paged path must match either committed golden for every prompt."""
    token_ids = _paged_tokens[prompt]
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
        f"EXAONE4 output for {prompt!r} matched neither golden.\n"
        f"Got:               {token_ids}\n"
        f"Expected (mlx_lm): {mlx_expected}\n"
        f"Expected (paged):  {paged_expected}"
    )
