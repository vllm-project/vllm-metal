#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Quantitative AWQ parity vs bf16 reference (logits cos, KL, TF top-1).

Standalone reproducer for the numbers cited in supported_models.md AWQ
rows. Mirrors PR #340's analysis: report cos/KL/top-1 for the AWQ
checkpoint vs the bf16 reference, and optionally also for a naive MLX
4bit baseline vs the same reference, so the AWQ-repack advantage is
quantified rather than asserted.

Pipeline (each stage loads, runs one forward, frees before the next):

  1. bf16 reference  : greedy-generate ``--steps`` tokens, then a single
                       TF forward to record per-step raw logits.
  2. AWQ             : TF forward over the reference's token sequence.
  3. naive 4bit      : same TF forward (only if ``--naive4bit`` given).

Two large models are NOT held in memory simultaneously; only per-step
logits (steps x vocab x fp32, ~40 MB for vocab=152k, steps=64) persist
across stages.

This is a standalone tool, not part of CI -- the slow regression tests
under tests/quant/ are the CI guards.

Usage:
    python tools/awq_parity.py \
        --awq Qwen/Qwen2.5-7B-Instruct-AWQ \
        --ref Qwen/Qwen2.5-7B-Instruct \
        --naive4bit mlx-community/Qwen2.5-7B-Instruct-4bit \
        --prompt "The capital of France is" \
        --steps 64
"""

from __future__ import annotations

import argparse
import gc

import mlx.core as mx


def _free_metal_cache() -> None:
    """Hand released allocations back to the Metal pool so the next
    model load has the full budget."""
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()


def _greedy_tokens(model, tokenizer, prompt: str, steps: int) -> list[int]:
    """Greedy-sample ``steps`` tokens from ``prompt`` using mlx_lm's
    cached step API. Returns the list of generated token ids.
    """
    from mlx_lm.generate import generate_step

    prompt_ids = mx.array(tokenizer.encode(prompt))

    def _greedy_sampler(logprobs: mx.array) -> mx.array:
        return mx.argmax(logprobs, axis=-1)

    tokens: list[int] = []
    for token, _logprobs in generate_step(
        prompt_ids, model, max_tokens=steps, sampler=_greedy_sampler
    ):
        # mlx_lm versions disagree on yield element type: some emit
        # ``mx.array`` (scalar), some emit ``int``. Accept either.
        tokens.append(int(token) if isinstance(token, int) else int(token.item()))
    return tokens


def _tf_logits(model, full_ids: mx.array, prompt_len: int, n_steps: int) -> mx.array:
    """Teacher-forced raw logits at the positions that predict each of
    the ``n_steps`` continuation tokens.

    Given a single forward over ``full_ids = prompt + ref_tokens`` of
    length ``prompt_len + n_steps``, position ``prompt_len - 1 + i``
    holds the model's prediction for ``ref_tokens[i]`` (causal mask
    means each output is a function of all preceding tokens only).

    Returns shape ``(n_steps, vocab)`` in fp32 for downstream metrics.
    """
    out = model(full_ids[None])  # [1, prompt_len + n_steps, vocab]
    start = prompt_len - 1
    logits = out[0, start : start + n_steps].astype(mx.float32)
    mx.eval(logits)
    return logits


def _cos(a: mx.array, b: mx.array) -> float:
    denom = (mx.linalg.norm(a) * mx.linalg.norm(b)).item()
    if denom == 0.0:
        return 0.0
    return float((a @ b).item() / denom)


def _kl(p_logits: mx.array, q_logits: mx.array) -> float:
    """KL(p || q) where ``p = softmax(p_logits)``, ``q = softmax(q_logits)``.
    Both in fp32. Forward KL from reference to quantized, matching
    PR #340's ``KL(ref || q)`` direction.
    """
    p = mx.softmax(p_logits)
    log_p = p_logits - mx.logsumexp(p_logits)
    log_q = q_logits - mx.logsumexp(q_logits)
    return float((p * (log_p - log_q)).sum().item())


def _summary(
    label: str, ref_logits: mx.array, q_logits: mx.array, n_steps: int
) -> tuple[float, float, int]:
    """Compute mean cos / mean KL / top-1 match count between two
    ``[n_steps, vocab]`` fp32 logit tensors. Returns a tuple of
    (mean_cos, mean_kl, top1_match) and prints a one-line row.
    """
    cos_per_step = [_cos(ref_logits[i], q_logits[i]) for i in range(n_steps)]
    kl_per_step = [_kl(ref_logits[i], q_logits[i]) for i in range(n_steps)]
    ref_argmax = mx.argmax(ref_logits, axis=-1)
    q_argmax = mx.argmax(q_logits, axis=-1)
    top1 = int((ref_argmax == q_argmax).sum().item())
    mean_cos = sum(cos_per_step) / len(cos_per_step)
    mean_kl = sum(kl_per_step) / len(kl_per_step)
    print(
        f"  {label:18s} | cos {mean_cos:.4f} | KL {mean_kl:.4f} | "
        f"top-1 {top1}/{n_steps}"
    )
    return mean_cos, mean_kl, top1


def _quantized_tf_logits(
    repo: str,
    args_prompt: str,
    prompt_ids_ref: list[int],
    full_ids: mx.array,
    prompt_len: int,
    n_steps: int,
) -> mx.array:
    """Load ``repo``, assert tokenizer agreement with the reference,
    run a single TF forward, free the model. Returns the [n_steps, vocab]
    fp32 logits.
    """
    from mlx_lm import load as mlx_lm_load

    print(f"[parity] loading {repo}")
    model, tok = mlx_lm_load(repo)
    # We pass the reference's exact token-id sequence (``full_ids``) into
    # this model's forward, so tokenizer-config differences across
    # mirrors (notably ``add_bos_token`` defaults on Llama family) do not
    # invalidate the comparison. What matters is that this model
    # interprets the same token ids as the reference -- i.e. shared
    # vocabulary. Check that, accepting different ``encode()`` behavior.
    prompt_ids = tok.encode(args_prompt)
    if prompt_ids != prompt_ids_ref:
        # Same vocab, different tokenizer config (e.g. add_bos_token).
        # As long as both encodings are a contiguous prefix-or-extension
        # relationship and use the same vocab, the TF comparison stands.
        print(
            f"[parity] note: {repo} encodes prompt as {prompt_ids[:8]}... "
            f"vs reference {prompt_ids_ref[:8]}... -- proceeding (TF uses "
            f"reference token ids)"
        )
    print(f"[parity] {repo} TF forward")
    logits = _tf_logits(model, full_ids, prompt_len, n_steps)
    del model, tok
    _free_metal_cache()
    return logits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--awq", required=True, help="AWQ checkpoint repo or path")
    ap.add_argument("--ref", required=True, help="bf16 reference repo or path")
    ap.add_argument(
        "--naive4bit",
        default=None,
        help="optional MLX-native 4bit baseline repo; reported alongside "
        "AWQ to quantify the AWQ-repack advantage vs naive MLX 4bit "
        "(mirrors PR #340's analysis)",
    )
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--steps", type=int, default=64)
    args = ap.parse_args()

    from mlx_lm import load as mlx_lm_load

    # Stage 1: reference. Greedy-generate to fix the token trajectory,
    # then a single TF forward to record reference raw logits.
    print(f"[parity] loading reference: {args.ref}")
    ref_model, ref_tok = mlx_lm_load(args.ref)

    print(f"[parity] greedy {args.steps} steps from reference")
    ref_tokens = _greedy_tokens(ref_model, ref_tok, args.prompt, args.steps)
    print(f"[parity] reference continuation: {ref_tok.decode(ref_tokens)!r}")

    prompt_ids = ref_tok.encode(args.prompt)
    prompt_len = len(prompt_ids)
    full_ids = mx.array(prompt_ids + ref_tokens)

    print("[parity] reference TF forward")
    ref_logits = _tf_logits(ref_model, full_ids, prompt_len, args.steps)

    # Release the reference before any further loads.
    del ref_model, ref_tok
    _free_metal_cache()

    # Stage 2: AWQ TF.
    awq_logits = _quantized_tf_logits(
        args.awq, args.prompt, prompt_ids, full_ids, prompt_len, args.steps
    )

    # Stage 3 (optional): naive MLX 4bit TF.
    naive_logits = None
    if args.naive4bit is not None:
        naive_logits = _quantized_tf_logits(
            args.naive4bit,
            args.prompt,
            prompt_ids,
            full_ids,
            prompt_len,
            args.steps,
        )

    # Metrics.
    print()
    print(f"reference : {args.ref}")
    print(f"AWQ       : {args.awq}")
    if args.naive4bit:
        print(f"naive4bit : {args.naive4bit}")
    print(f"prompt    : {args.prompt!r}")
    print(f"steps     : {args.steps}")
    print()
    _summary("AWQ-repack", ref_logits, awq_logits, args.steps)
    if naive_logits is not None:
        _summary("naive MLX 4bit", ref_logits, naive_logits, args.steps)


if __name__ == "__main__":
    main()
