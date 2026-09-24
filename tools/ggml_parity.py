# SPDX-License-Identifier: Apache-2.0
"""Parity check: vllm-metal ggml engine vs. HF transformers reference logits.

Exercises the paths the vLLM runner relies on:
  * single-shot prefill (all positions),
  * chunked prefill + token-by-token decode (paged KV + recurrent state carry),
  * a mixed batch (decode + prefill groups, shuffled block tables, several
    recurrent-state slots) in one step.

Usage:
  python tools/ggml_parity.py Qwen/Qwen3.5-0.8B
  python tools/ggml_parity.py google/gemma-4-E2B-it --long 700
"""

from __future__ import annotations

import argparse
import random
import sys

import numpy as np
import torch

BLOCK = 16


class Seq:
    """Host-side bookkeeping mirroring what the vLLM runner tracks."""

    def __init__(self, tokens: list[int], slot: int, free_blocks: list[int]):
        self.tokens = tokens
        self.slot = slot
        self.done = 0
        need = (len(tokens) + BLOCK - 1) // BLOCK
        self.blocks = [free_blocks.pop() for _ in range(need)]


def run_step(engine, vocab: int, work: list[tuple[Seq, int, int]]) -> list[np.ndarray]:
    """work: (seq, n_new_tokens, n_logits). Returns logits per seq."""
    toks, q, c, slots, reset, nl = [], [], [], [], [], []
    max_blocks = max(len(s.blocks) for s, _, _ in work)
    bt = np.zeros((len(work), max_blocks), np.int32)
    for i, (s, n, k) in enumerate(work):
        toks += s.tokens[s.done : s.done + n]
        q.append(n)
        c.append(s.done + n)
        slots.append(s.slot)
        reset.append(1 if s.done == 0 else 0)
        nl.append(k)
        bt[i, : len(s.blocks)] = s.blocks
    rows = sum(nl)
    out = np.empty((max(rows, 1), vocab), np.float32)
    a = lambda x: np.asarray(x, np.int32)  # noqa: E731
    got = engine.forward(a(toks), a(q), a(c), bt, a(slots), a(reset), a(nl), out)
    assert got == rows, (got, rows)
    res, r = [], 0
    for s, n, k in work:
        s.done += n
        res.append(out[r : r + k])
        r += k
    return res


def compare(name: str, ours: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    ours = torch.from_numpy(ours).double()
    ref = torch.from_numpy(ref).double()
    top1 = (ours.argmax(-1) == ref.argmax(-1)).double().mean().item()
    lp, lq = torch.log_softmax(ref, -1), torch.log_softmax(ours, -1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    maxdiff = (ours - ref).abs().max().item()
    print(
        f"  {name:<34} top1={top1:6.3f}  mean_KL={kl.mean().item():.2e}  "
        f"max_KL={kl.max().item():.2e}  max|dlogit|={maxdiff:.3f}"
    )
    return top1, kl.mean().item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", default="metal")
    ap.add_argument(
        "--long",
        type=int,
        default=0,
        help="extra long-prompt length (sliding window test)",
    )
    ap.add_argument("--ref-dtype", default="float32", choices=["float32", "bfloat16"])
    # Mean-KL gate. For scale: HF's own bf16 inference of gemma-4-E2B sits at
    # mean KL ~6.5e-3 / top1 0.98 against its fp32 run on a 700-token prompt;
    # ggml matmuls round activations to bf16 for bf16 weights, so per-position
    # max KL on high-entropy positions is not a meaningful gate.
    ap.add_argument("--kl-tol", type=float, default=5e-3)
    ap.add_argument("--top1-tol", type=float, default=0.95)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    from vllm_metal.ggml import load_extension

    path = snapshot_download(args.model)
    tok = AutoTokenizer.from_pretrained(path)
    cfg = AutoConfig.from_pretrained(path)
    dtype = getattr(torch, args.ref_dtype)
    loader = (
        AutoModelForImageTextToText
        if hasattr(cfg, "text_config")
        else AutoModelForCausalLM
    )
    ref_model = loader.from_pretrained(path, dtype=dtype).eval()

    @torch.no_grad()
    def ref_logits(model, ids: list[int]) -> np.ndarray:
        out = model(input_ids=torch.tensor([ids]), use_cache=False)
        return out.logits[0].float().numpy()

    texts = [
        "The capital of France is Paris. The capital of Germany is",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
        "Once upon a time, in a land far, far away, there lived a",
    ]
    prompts = [tok(t, add_special_tokens=True)["input_ids"] for t in texts]
    if args.long:
        rng = random.Random(0)
        body = " ".join(
            rng.choice(["alpha", "beta", "gamma", "delta", "river", "stone"])
            for _ in range(args.long)
        )
        prompts.append(
            tok(body + " The end.", add_special_tokens=True)["input_ids"][: args.long]
        )

    refs = [ref_logits(ref_model, p) for p in prompts]
    del ref_model

    ext = load_extension()
    eng = ext.Engine(path, args.device)
    info = eng.info()
    vocab = info["vocab_size"]
    num_blocks = 1 + sum((len(p) + BLOCK - 1) // BLOCK for p in prompts) * 2
    eng.init_cache(num_blocks, BLOCK, 8)
    print(
        f"engine: {info['arch']} on {info['backend']}, weights {info['weight_bytes'] / 2**30:.2f} GiB"
    )

    ok = True

    def check(name, ours, ref):
        nonlocal ok
        top1, kl = compare(name, ours, ref)
        ok &= kl < args.kl_tol and top1 >= args.top1_tol

    def fresh_blocks():
        blocks = list(range(1, num_blocks))
        random.Random(1).shuffle(blocks)
        return blocks

    # 1. single-shot prefill, logits for every position
    print("single-shot prefill:")
    free = fresh_blocks()
    for i, (p, r) in enumerate(zip(prompts, refs, strict=True)):
        s = Seq(p, i, free)
        (lg,) = run_step(eng, vocab, [(s, len(p), len(p))])
        check(f"prompt{i} (len={len(p)})", lg, r)

    # 2. chunked prefill then per-token decode
    print("chunked prefill + decode:")
    free = fresh_blocks()
    for i, (p, r) in enumerate(zip(prompts, refs, strict=True)):
        s = Seq(p, i, free)
        chunk = max(3, len(p) // 3)
        split = min(len(p) - 4, 2 * chunk)
        rows = []
        (lg,) = run_step(eng, vocab, [(s, chunk, chunk)])
        rows.append(lg)
        (lg,) = run_step(eng, vocab, [(s, split - chunk, split - chunk)])
        rows.append(lg)
        while s.done < len(p):
            (lg,) = run_step(eng, vocab, [(s, 1, 1)])
            rows.append(lg)
        check(f"prompt{i} chunk={chunk}", np.concatenate(rows), r)

    # 3. mixed batch: seq0/seq1 decoding while seq2 prefills, one step each
    print("mixed batch (decode group + prefill group):")
    free = fresh_blocks()
    seqs = [Seq(p, i, free) for i, p in enumerate(prompts[:3])]
    got = {i: [] for i in range(3)}
    # prime seq0/seq1 up to 4 tokens before the end
    for i in (0, 1):
        n = len(prompts[i]) - 4
        (lg,) = run_step(eng, vocab, [(seqs[i], n, n)])
        got[i].append(lg)
    s2 = seqs[2]
    p2_first = len(prompts[2]) - 4
    step_work = [(seqs[0], 1, 1), (seqs[1], 1, 1), (s2, p2_first, p2_first)]
    outs = run_step(eng, vocab, step_work)
    for (s, _, _), lg in zip(step_work, outs, strict=True):
        got[prompts.index(s.tokens)].append(lg)
    while any(s.done < len(s.tokens) for s in seqs):
        work = [(s, 1, 1) for s in seqs if s.done < len(s.tokens)]
        for (s, _, _), lg in zip(work, run_step(eng, vocab, work), strict=True):
            got[prompts.index(s.tokens)].append(lg)
    for i in range(3):
        check(f"prompt{i} in mixed batch", np.concatenate(got[i]), refs[i])

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
