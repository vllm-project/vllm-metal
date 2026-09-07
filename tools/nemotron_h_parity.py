#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Nemotron-H parity tool: mlx_lm greedy vs the vllm-metal paged runtime.

Each arm runs in its own spawned process (Metal is not fork-safe and two
30B copies do not fit one budget). The paged child also asserts reach so
the gate cannot pass on a model that silently skipped its Mamba-2 layers:
``patch_model`` must return exactly the plan's attention plus state layer
count, and the Mamba-2 wrapper must run at least once per prompt per state
layer. With ``--check-state`` the mlx_lm arm also steps the first prompt by
hand and ships its layer-0 conv and SSM state, which the paged arm compares
against the pool rows of that request's slot.

A prompt passes when the greedy tokens match, or when the other arm's first
divergent token is mlx_lm's second choice within ``NEAR_TIE_ULPS`` bf16 ULPs
of its first; a divergence outside that margin fails the run.

Not in CI, requires local weights.

Usage:
    python tools/nemotron_h_parity.py
    python tools/nemotron_h_parity.py --model /path/to/checkpoint --max-tokens 16
    python tools/nemotron_h_parity.py --arm nonpaged
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
from queue import Empty

import numpy as np

MODEL_DEFAULT = os.environ.get(
    "NEMOTRON_H_MODEL_PATH", "mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit"
)
# Greedy tokens may flip where mlx_lm's own top two logits sit within a few
# bf16 ULPs; such a flip is rounding, not a state bug, and the run stays green
# when the other arm picked mlx_lm's second choice inside this margin.
NEAR_TIE_ULPS = 4
BF16_MANTISSA_BITS = 7
# Relative tolerance on the layer-0 pool rows against mlx-lm's own cache; the
# bf16 activations feeding them differ by rounding, not by state content.
STATE_REL_TOLERANCE = 1e-2
MAX_MODEL_LEN = 2048
CHILD_POLL_SECONDS = 5
PROMPTS = [
    "The capital of France is",
    "One plus one equals",
    "Monday, Tuesday, Wednesday,",
    "Water boils at a temperature of",
]
CHUNK_MNBT = 256


def _to_numpy(array) -> np.ndarray:
    """Copy an mx array out as float32 (numpy has no bf16 buffer format)."""
    import mlx.core as mx

    return np.array(array.astype(mx.float32))


def _child_env(paged: bool) -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ["VLLM_METAL_USE_PAGED_ATTENTION"] = "1" if paged else "0"
    # The non-paged MLX cache path only accepts the auto memory fraction.
    os.environ["VLLM_METAL_MEMORY_FRACTION"] = "0.5" if paged else "auto"


def _divergence(m, tokenizer, prompt: str, mlx_tokens, other_tokens) -> dict | None:
    """Rank the other arm's first divergent token in mlx_lm's own logits."""
    import mlx.core as mx

    pairs = list(zip(mlx_tokens, other_tokens, strict=False))
    diverged = [i for i, (a, b) in enumerate(pairs) if a != b]
    if not diverged:
        return None
    k = diverged[0]
    b = other_tokens[k]
    ids = tokenizer.encode(prompt) + other_tokens[:k]
    logits = m(mx.array(ids)[None])[0, -1].astype(mx.float32)
    order = mx.argsort(-logits)
    top = [(int(t), round(float(logits[t]), 4)) for t in order[:3].tolist()]
    rank = int(mx.argmax(order == b))
    best = float(logits[order[0]])
    ulp = 2.0 ** (math.floor(math.log2(abs(best))) - BF16_MANTISSA_BITS)
    margin_ulps = (best - float(logits[b])) / ulp
    return {
        "index": k,
        "top3": top,
        "other": (b, round(float(logits[b]), 4), rank),
        "near_tie": rank == 1 and margin_ulps <= NEAR_TIE_ULPS,
        "margin_ulps": round(margin_ulps, 1),
    }


def run_mlx_child(
    model: str,
    max_tokens: int,
    replay_tokens: list[int] | None,
    other_tokens: dict[str, list[int]] | None,
    queue,
) -> None:
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    m, tokenizer = load(model)
    tokenizer.eos_token_ids = set()
    sampler = make_sampler(temp=0.0)
    tokens = {
        prompt: [
            r.token
            for r in stream_generate(
                m, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler
            )
        ]
        for prompt in PROMPTS
    }
    divergence = {}
    if other_tokens is not None:
        for prompt in PROMPTS:
            report = _divergence(
                m, tokenizer, prompt, tokens[prompt], other_tokens[prompt]
            )
            if report is not None:
                divergence[prompt] = report
    state = None
    if replay_tokens is not None:
        # Replay the first prompt with the tokens the paged arm actually fed
        # (prompt plus all but its last sample), so state compares like with like.
        cache = make_prompt_cache(m)
        ids = tokenizer.encode(PROMPTS[0]) + replay_tokens[:-1]
        m(mx.array(ids)[None], cache=cache)
        mx.eval(cache[0][0], cache[0][1])
        state = (_to_numpy(cache[0][0][0]), _to_numpy(cache[0][1][0]))
    queue.put({"tokens": tokens, "state": state, "divergence": divergence})


def run_vllm_child(
    model: str,
    max_tokens: int,
    paged: bool,
    check_state: bool,
    max_num_seqs: int,
    queue,
) -> None:
    _child_env(paged)
    from vllm import LLM, SamplingParams

    reach = {
        "wrapper_calls": 0,
        "patched": None,
        "expected": None,
        "num_state": None,
        "slot": None,
    }
    captured = {}
    if paged:
        from vllm_metal.attention.impls import mamba2
        from vllm_metal.attention.runtime import hybrid

        real_call = mamba2.Mamba2PagedStateWrapper.__call__
        real_patch = hybrid.HybridPagedAttentionRuntime.patch_model

        def spy_call(self, x, mask=None, cache=None):
            reach["wrapper_calls"] += 1
            if self._mamba2_cache_idx == 0:
                ctx = mamba2.get_context()
                if ctx is not None and ctx.state_slot_mapping is not None:
                    reach["slot"] = ctx.state_slot_mapping[-1]
            return real_call(self, x, mask=mask, cache=cache)

        def spy_patch(self, model_obj):
            layers = self._hybrid_plan.layers
            reach["expected"] = layers.num_attention + layers.num_state
            reach["num_state"] = layers.num_state
            reach["patched"] = real_patch(self, model_obj)
            captured["runtime"] = self
            return reach["patched"]

        mamba2.Mamba2PagedStateWrapper.__call__ = spy_call
        hybrid.HybridPagedAttentionRuntime.patch_model = spy_patch

    llm = LLM(
        model=model,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=False,
        max_num_batched_tokens=CHUNK_MNBT,
        disable_log_stats=True,
    )
    sp = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
    tokens = {o.prompt: list(o.outputs[0].token_ids) for o in llm.generate(PROMPTS, sp)}
    state = None
    if paged and check_state:
        llm.generate([PROMPTS[0]], sp)
        cache = captured["runtime"].state_cache
        slot = reach["slot"]
        state = (
            _to_numpy(cache.conv_states[0][slot]),
            _to_numpy(cache.recurrent_states[0][slot]),
        )
    queue.put({"tokens": tokens, "reach": reach, "state": state})


def _run(target, *args) -> dict:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=target, args=(*args, queue))
    proc.start()
    while True:
        try:
            result = queue.get(timeout=CHILD_POLL_SECONDS)
            break
        except Empty:
            if not proc.is_alive():
                raise SystemExit(
                    f"child exited with {proc.exitcode} before reporting"
                ) from None
    proc.join()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--arm", choices=("paged", "nonpaged"), default="paged")
    parser.add_argument("--check-state", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=4)
    args = parser.parse_args()
    paged = args.arm == "paged"
    check_state = args.check_state and paged

    vllm_arm = _run(
        run_vllm_child,
        args.model,
        args.max_tokens,
        paged,
        check_state,
        args.max_num_seqs,
    )
    replay = vllm_arm["tokens"][PROMPTS[0]] if check_state else None
    mlx = _run(run_mlx_child, args.model, args.max_tokens, replay, vllm_arm["tokens"])

    ok = True
    for prompt in PROMPTS:
        report = mlx["divergence"].get(prompt)
        if report is None:
            verdict = "MATCH"
        elif report["near_tie"]:
            verdict = "TIE  "
        else:
            verdict = "DIFF "
            ok = False
        print(f"[{verdict}] {prompt!r}")
        print(f"    mlx_lm: {mlx['tokens'][prompt]}")
        print(f"    {args.arm}: {vllm_arm['tokens'][prompt]}")
        if report is not None:
            token, logit, rank = report["other"]
            print(
                f"    diverges at {report['index']}: mlx_lm top3={report['top3']}, "
                f"{args.arm} token {token} logit {logit} (rank {rank}, "
                f"{report['margin_ulps']} bf16 ULPs behind)"
            )
    if paged:
        reach = vllm_arm["reach"]
        print(
            f"reach: patched={reach['patched']} expected={reach['expected']} "
            f"wrapper_calls={reach['wrapper_calls']}"
        )
        ok &= reach["patched"] == reach["expected"]
        ok &= reach["wrapper_calls"] >= len(PROMPTS) * reach["num_state"]
    if check_state:
        ref_conv, ref_ssm = mlx["state"]
        conv, ssm = vllm_arm["state"]
        conv_scale = float(np.max(np.abs(ref_conv)))
        ssm_scale = float(np.max(np.abs(ref_ssm)))
        conv_rel = float(np.max(np.abs(conv - ref_conv))) / conv_scale
        ssm_rel = float(np.max(np.abs(ssm - ref_ssm))) / ssm_scale
        print(
            f"state layer 0 (replayed tokens): conv max rel diff={conv_rel:.3e} "
            f"of {conv_scale:.3g}, ssm max rel diff={ssm_rel:.3e} of {ssm_scale:.3g}"
        )
        ok &= conv_rel <= STATE_REL_TOLERANCE and ssm_rel <= STATE_REL_TOLERANCE
    print("\nPARITY OK" if ok else "\nPARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
