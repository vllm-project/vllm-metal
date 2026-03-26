# SPDX-License-Identifier: Apache-2.0
"""Golden token test for Qwen3.5 mlx_lm inline cache path.

Validates greedy-decoded tokens from the mlx_lm inline cache path on
Qwen3.5-0.8B against hardcoded golden IDs.  This establishes a baseline
for future paged-attention kernel validation.

Run:
    python -m pytest tests/test_qwen35_golden.py -v -s
"""

from __future__ import annotations

MODEL_NAME = "Qwen/Qwen3.5-0.8B"

# fmt: off
# Golden token IDs from MLX inline cache, greedy decoding (Qwen3.5-0.8B).
# Prompt: "The capital of France is"
GOLDEN = [760, 6511, 314, 9338, 369, 11751, 13, 198, 760, 6511, 314, 9338, 369, 11751, 13]
# fmt: on


def test_qwen35_golden_tokens():
    """mlx_lm inline cache greedy decode must match golden token IDs."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer = load(MODEL_NAME)

    prompt_tokens = GOLDEN[:5]  # "The capital of France is"
    num_generate = len(GOLDEN) - len(prompt_tokens)

    cache = make_prompt_cache(model)
    input_ids = mx.array([prompt_tokens], dtype=mx.int32)
    out = model(input_ids, cache=cache)
    mx.eval(out)

    tokens = list(prompt_tokens)
    for _ in range(num_generate):
        next_token = mx.argmax(out[:, -1, :], axis=-1)
        mx.eval(next_token)
        tid = int(next_token.item())
        tokens.append(tid)
        out = model(mx.array([[tid]], dtype=mx.int32), cache=cache)
        mx.eval(out)

    assert tokens == GOLDEN, (
        f"Golden token mismatch.\nExpected: {GOLDEN}\nGot:      {tokens}"
    )
