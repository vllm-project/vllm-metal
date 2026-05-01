#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: ``--block-size`` (== ``LLM(block_size=...)``) flows correctly
through vLLM 0.20's ``Platform.update_block_size_for_backend`` and our
``MetalBackend(MultipleOf(16))`` advertisement, on both non-hybrid and hybrid
models.

Background:
- After the vLLM 0.20 bump, ``Platform.update_block_size_for_backend`` Phase 1
  picks ``MetalBackend.get_preferred_block_size()`` (16 from ``MultipleOf(16)``,
  the Metal kernel sweet spot) unless ``cache_config.user_specified_block_size``
  is True.
- ``--block-size N`` (CLI) and ``LLM(block_size=N)`` (Python API) both set that
  flag.

For non-hybrid models the user's value survives end-to-end (Phase 2 skips),
so we assert ``actual == requested`` on the parent's view.

For hybrid models (e.g. Qwen3.5: SDPA + GDN linear attention), Phase 2's
``_align_hybrid_block_size`` *does* bump the subprocess's
``cache_config.block_size`` upward to a multiple of 16 to satisfy
mamba/attention page-size alignment, but vLLM spawns the EngineCore in a
separate process and the bump never propagates back to the parent's
``vllm_config``.  So the parent always sees the user's input value, regardless.
The meaningful verification for the hybrid case is therefore implicit: if
``MetalBackend(MultipleOf(16))`` did not flow correctly through
``_align_hybrid_block_size``, ``unify_kv_cache_spec_page_size`` would have
raised ``NotImplementedError`` in the subprocess and ``LLM(...)`` would never
have returned.  We assert that ``llm.generate`` produces non-empty output —
reaching that line proves the whole contract chain ran without crashing.

Each iteration runs in a fresh subprocess so MLX state and the EngineCore
process tree are reset cleanly between runs.

Usage:
    python tools/test_block_size_override.py
"""

from __future__ import annotations

import os
import subprocess
import sys

# (model, block_size, hybrid)
TEST_CASES: tuple[tuple[str, int, bool], ...] = (
    ("Qwen/Qwen3-0.6B", 8, False),
    ("Qwen/Qwen3-0.6B", 16, False),
    ("Qwen/Qwen3.5-0.8B", 8, True),
    ("Qwen/Qwen3.5-0.8B", 16, True),
)


def _check(actual: int, requested: int, text: str, hybrid: bool) -> tuple[bool, str]:
    """Return (passed, rule_description) for the given case."""
    has_output = bool(text)
    if hybrid:
        # Subprocess-side Phase 2 bump is invisible from the parent; engine
        # startup + non-empty generation is the real proof of contract.
        return has_output, "engine_started_and_generated"
    return actual == requested and has_output, f"actual=={requested} and has_output"


def _run_one(model: str, block_size: int, hybrid: bool) -> int:
    """Subprocess entry point: load LLM, print result, exit 0 on pass."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        block_size=block_size,
        max_model_len=512,
        max_num_batched_tokens=64,
        enforce_eager=True,
    )
    actual = llm.llm_engine.vllm_config.cache_config.block_size
    out = llm.generate(["Hello"], SamplingParams(max_tokens=3, temperature=0))
    text = out[0].outputs[0].text

    ok, rule = _check(actual, block_size, text, hybrid)
    status = "PASS" if ok else "FAIL"
    print(
        f"[block_size_override] model={model} requested={block_size} "
        f"actual={actual} hybrid={hybrid} rule=({rule}) text={text!r} [{status}]",
        flush=True,
    )
    return 0 if ok else 1


def main() -> int:
    if len(sys.argv) == 4:
        return _run_one(sys.argv[1], int(sys.argv[2]), sys.argv[3] == "True")

    env = os.environ.copy()
    env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    env.setdefault("VLLM_METAL_USE_PAGED_ATTENTION", "1")
    env.setdefault("VLLM_METAL_MEMORY_FRACTION", "0.8")

    failures: list[tuple[str, int]] = []
    for model, bs, hybrid in TEST_CASES:
        print(
            f"\n=== Loading {model} with block_size={bs} (hybrid={hybrid}) ===",
            flush=True,
        )
        proc = subprocess.run(
            [sys.executable, __file__, model, str(bs), str(hybrid)],
            env=env,
        )
        if proc.returncode != 0:
            failures.append((model, bs))

    print()
    if failures:
        print(f"FAIL: block_size override did not hold for: {failures}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
