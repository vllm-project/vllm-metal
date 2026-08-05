# SPDX-License-Identifier: Apache-2.0
"""End-to-end correctness of align-mode prefix caching on hybrid GDN models.

Runs Qwen3.5-0.8B (18 GDN + 6 SDPA layers) with ``enable_prefix_caching=True``,
which the config hook resolves to ``mamba_cache_mode="align"``: the scheduler
checkpoints GDN state in the block covering each block-aligned step end, and a
later request with the same prefix restores from that block instead of
re-running prefill.

Assertions (in code order):
  0. Config reach — the run really resolved to align mode with
     ``mamba_block_size == block_size`` (otherwise this test observes the
     none-mode path and proves nothing).
  1. Hit reach — the second shared-prefix request is admitted with
     ``num_computed_tokens`` at a positive multiple of the block size (a
     genuine restore, not a fresh prefill).
  2. Losslessness — every generation (fresh, restored, and a decode that
     crosses a block boundary, exercising state copy-forward) matches the
     caching-off reference produced in a separate child process.

Same spawn discipline as ``test_paged_prefix_caching_e2e``: each LLM body
runs in a spawned child (Metal is not fork-safe, and MLX wired buffers do
not return on gc, so cache-on and cache-off must not share a process).

Not in CI — requires local Qwen3.5-0.8B weights.
"""

from __future__ import annotations

import multiprocessing as mp
import os

import pytest

HYBRID_MODEL = os.environ.get("QWEN35_MODEL_PATH", "Qwen/Qwen3.5-0.8B")

# ~1200 tokens: two+ full blocks at the hook-raised hybrid block size (544
# for Qwen3.5-0.8B), so the second request can restore at a block boundary.
SHARED_PREFIX = (
    "The following is a detailed report about renewable energy. " * 120
).strip()
# ~525 tokens + 40 decoded tokens crosses one block boundary mid-decode.
BOUNDARY_PROMPT = ("Number the reasons carefully and explain. " * 75).strip()

SHORT_PROMPTS = [
    "The capital of France is",
    "One plus one equals",
]


def _setenv_default(key: str, default: str) -> None:
    if os.environ.get(key) is None:
        os.environ[key] = default


def _child_env() -> None:
    _setenv_default("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    _setenv_default("VLLM_METAL_USE_PAGED_ATTENTION", "1")
    _setenv_default("VLLM_METAL_MEMORY_FRACTION", "0.5")


def _run_suite(enable_prefix_caching: bool, queue) -> None:
    """Generate the fixed suite and put (results, admitted, config) on queue."""
    _child_env()

    from vllm import LLM, SamplingParams

    import vllm_metal.v1.model_runner as mr_mod

    admitted: list[tuple[str, int]] = []
    orig = mr_mod.MetalModelRunner._handle_new_requests

    def spy(self, batch, new_reqs, scheduler_output):
        for req in new_reqs:
            admitted.append((req.req_id, req.num_computed_tokens))
        return orig(self, batch, new_reqs, scheduler_output)

    mr_mod.MetalModelRunner._handle_new_requests = spy
    try:
        llm = LLM(
            model=HYBRID_MODEL,
            max_model_len=2048,
            max_num_seqs=4,
            enable_prefix_caching=enable_prefix_caching,
        )
        cache_config = llm.llm_engine.vllm_config.cache_config
        config = {
            "block_size": cache_config.block_size,
            "mamba_cache_mode": cache_config.mamba_cache_mode,
            "mamba_block_size": cache_config.mamba_block_size,
        }

        results: dict[str, list[int]] = {}
        sp_short = SamplingParams(temperature=0, max_tokens=12)
        for out in llm.generate(SHORT_PROMPTS, sp_short):
            results[f"short:{out.prompt[:20]}"] = list(out.outputs[0].token_ids)

        # Sequential shared-prefix requests: the second can only hit blocks
        # cached by the first (align defers same-step sharing by design).
        for tag, suffix in (
            ("suffix1", " Summarize the key points:"),
            ("suffix2", " List three risks:"),
        ):
            out = llm.generate([SHARED_PREFIX + suffix], sp_short)[0]
            results[f"shared:{tag}"] = list(out.outputs[0].token_ids)

        sp_long = SamplingParams(temperature=0, max_tokens=40)
        out = llm.generate([BOUNDARY_PROMPT], sp_long)[0]
        results["boundary"] = list(out.outputs[0].token_ids)

        # Concurrent same-prefix batch: align defers same-step block sharing
        # (a request cannot hit blocks cached within the same step), so this
        # exercises the deferral bookkeeping rather than the hit path.
        batch_prompts = [
            SHARED_PREFIX + f" Task {i}: name one advantage." for i in range(3)
        ]
        for i, out in enumerate(llm.generate(batch_prompts, sp_short)):
            results[f"batch:{i}"] = list(out.outputs[0].token_ids)
    finally:
        mr_mod.MetalModelRunner._handle_new_requests = orig

    queue.put((results, admitted, config))


def _run_in_child(enable_prefix_caching: bool):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_run_suite, args=(enable_prefix_caching, queue))
    proc.start()
    try:
        payload = queue.get(timeout=900)
    finally:
        proc.join(timeout=60)
        if proc.is_alive():
            proc.terminate()
    assert proc.exitcode == 0, f"child exited with {proc.exitcode}"
    return payload


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("VLLM_METAL_E2E", "1") != "1",
    reason="e2e disabled",
)
def test_hybrid_align_prefix_caching_is_lossless_and_hits() -> None:
    reference, _, ref_config = _run_in_child(enable_prefix_caching=False)
    cached, admitted, config = _run_in_child(enable_prefix_caching=True)

    # 0. Config reach.
    assert ref_config["mamba_cache_mode"] == "none"
    assert config["mamba_cache_mode"] == "align", config
    assert config["mamba_block_size"] == config["block_size"], config

    # 1. Hit reach: the second shared-prefix request restores at a
    # block-aligned position covered by the first request's checkpoints.
    block_size = config["block_size"]
    hits = [n for _, n in admitted if n > 0]
    assert hits, f"no admission carried num_computed_tokens > 0: {admitted}"
    assert any(n % block_size == 0 for n in hits), (
        f"expected a block-aligned restore (multiple of {block_size}), got {hits}"
    )

    # 2. Losslessness against the caching-off reference.
    assert set(reference) == set(cached)
    mismatches = {
        key: (reference[key], cached[key])
        for key in reference
        if reference[key] != cached[key]
    }
    assert not mismatches, f"align-mode caching changed tokens: {mismatches}"
