#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-27B-4bit e2e decode A/B: paged GQA vs old per-token/split-KV.

Same process, same weights. ``num_decode_requests=0`` misses the GQA gate
and is the old kernel; the production path is GQA when the 16k occupancy
gate fires. Prefill is unchanged by this PR; we still record TTFT.

Usage (from repo root, venv active):

    VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_METAL_COMPILED_MLP=1 \\
      VLLM_METAL_NATIVE_SAMPLING=1 python tools/e2e_qwen38.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import mlx.core as mx

MODEL = os.environ.get(
    "GQA_E2E_MODEL",
    "/Users/suntp/.cache/modelscope/models/mlx-community--Qwen3.8-27B-4bit/snapshots/master",
)
OUT_DIR = os.getcwd()  # logs/results land where you invoke it
TAG = os.environ.get("GQA_E2E_TAG", "e2e")
LOG = (
    os.path.join(OUT_DIR, f"e2e-qwen38-{TAG}.log")
    if TAG != "e2e"
    else os.path.join(OUT_DIR, "e2e-qwen38.log")
)
JSON_OUT = (
    os.path.join(OUT_DIR, f"e2e-qwen38-{TAG}.json")
    if TAG != "e2e"
    else os.path.join(OUT_DIR, "e2e-qwen38.json")
)
FAST_MIN_TFLOPS = float(os.environ.get("GQA_E2E_MIN_TFLOPS", "100"))
FAST_WAIT_S = int(os.environ.get("GQA_E2E_FAST_WAIT_S", "1200"))
REQUIRE_FAST = os.environ.get("GQA_E2E_REQUIRE_FAST", "0") == "1"

GEN_TOKENS = 32
MAX_MODEL_LEN = 110_000
MAX_NUM_SEQS = 4

# (ctx, concurrency). Gate fires when conc * ctx >= 16384 (pure decode).
CASES: list[tuple[int, int]] = [
    (4096, 1),
    (8192, 1),
    (16384, 1),
    (32768, 1),
    (65536, 1),
    (102400, 1),
    (8192, 2),
    (16384, 2),
    (32768, 2),
    (4096, 4),
    (8192, 4),
    (16384, 4),
]


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line)


def probe_tflops() -> float:
    a = mx.random.normal((2048, 2048)).astype(mx.bfloat16)
    b = mx.random.normal((2048, 2048)).astype(mx.bfloat16)
    for _ in range(5):
        mx.eval(a @ b)
    ts = []
    for _ in range(8):
        t0 = time.perf_counter()
        mx.eval(a @ b)
        ts.append(time.perf_counter() - t0)
    best = min(ts)
    return (2 * (2048**3)) / best / 1e12


def wait_for_fast(min_tflops: float, timeout_s: int) -> float:
    """Idle between probes so the GPU can thermally recover.

    Sustained GEMM keeps clocks down; a short burst every 20s is enough
    to notice when the chip comes back.
    """
    deadline = time.time() + timeout_s
    best = 0.0
    while True:
        tps = probe_tflops()
        best = max(best, tps)
        state = "fast" if tps >= min_tflops else "slow"
        log(f"probe_tflops={tps:.1f} best={best:.1f} state={state}")
        if tps >= min_tflops:
            return tps
        if time.time() >= deadline:
            return tps
        time.sleep(20.0)


def install_gqa_switch() -> callable:
    """Return a setter(disable: bool). Old path = num_decode_requests=0."""
    import vllm_metal.attention.impls.sdpa as sdpa_mod

    state = {"disable": False}
    real_get_ops = sdpa_mod.get_ops

    class _Ops:
        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def paged_attention_primitive(self, *args, **kwargs):
            if state["disable"]:
                kwargs["num_decode_requests"] = 0
            return self._inner.paged_attention_primitive(*args, **kwargs)

    cache: dict = {}

    def _get_ops():
        inner = real_get_ops()
        proxy = cache.get(id(inner))
        if proxy is None:
            proxy = _Ops(inner)
            cache[id(inner)] = proxy
        return proxy

    sdpa_mod.get_ops = _get_ops  # type: ignore[method-assign]
    return lambda disable: state.update(disable=disable)


@dataclass
class Row:
    ctx: int
    conc: int
    mode: str
    gate_expect: str
    prompt_tokens: int
    gen_tokens: int
    ttft_s: float
    decode_s: float
    decode_tok_s: float
    e2e_s: float
    prefill_tok_s: float


def _metrics_times(req) -> tuple[float, float, float]:
    """Return (ttft, decode_s, e2e_s) from vLLM request metrics + wall fallback."""
    m = getattr(req, "metrics", None)
    if m is None:
        return float("nan"), float("nan"), float("nan")
    arrival = getattr(m, "arrival_time", None)
    first = getattr(m, "first_token_time", None)
    last = getattr(m, "finished_time", None) or getattr(m, "last_token_time", None)
    if not arrival or not first or not last:
        return float("nan"), float("nan"), float("nan")
    return first - arrival, last - first, last - arrival


def _make_prompts(tokenizer, ctx: int, conc: int):
    """Exact-length token prompts (no decode/encode round-trip shrink)."""
    from vllm.inputs import TokensPrompt

    filler = tokenizer.encode(
        "The history of mathematics is a long story of counting, measuring, and reasoning. ",
        add_special_tokens=False,
    )
    if not filler:
        raise RuntimeError("tokenizer produced empty filler")
    prompts = []
    for i in range(conc):
        salt = tokenizer.encode(f" #{i}", add_special_tokens=False) or [filler[0]]
        n_body = max(1, ctx - len(salt))
        body: list[int] = []
        while len(body) < n_body:
            body.extend(filler)
        ids = body[:n_body] + salt
        ids = ids[:ctx]
        if len(ids) < ctx:
            ids.extend([filler[0]] * (ctx - len(ids)))
        prompts.append(TokensPrompt(prompt_token_ids=ids))
    return prompts


def _generate(llm, prompts, max_tokens: int):
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    n_gen = [len(o.outputs[0].token_ids) for o in outs]
    n_prompt = [len(o.prompt_token_ids or []) for o in outs]
    ttfts, decs = [], []
    for o in outs:
        ttft, dec, _ = _metrics_times(o)
        ttfts.append(ttft)
        decs.append(dec)
    ttft = max(ttfts) if ttfts and ttfts[0] == ttfts[0] else float("nan")
    decode_s = max(decs) if decs and decs[0] == decs[0] else float("nan")
    return outs, wall, n_prompt, n_gen, ttft, decode_s


def run_pair(llm, tokenizer, setter, ctx: int, conc: int) -> list[Row]:
    """Prefill once (fills prefix cache), then decode-only old vs GQA.

    Prefix caching is content-keyed on the prompt, so the two decode legs
    both skip prefill and are comparable. The first generate of 1 token is
    the prefill measurement.
    """
    prompts = _make_prompts(tokenizer, ctx, conc)
    agg = ctx * conc
    gate = "ON" if agg >= 16384 else "off"

    setter(False)
    _, prefill_wall, n_prompt, _, ttft_prefill, _ = _generate(llm, prompts, 1)
    prompt_tokens = sum(n_prompt) if n_prompt and n_prompt[0] else ctx * conc
    prefill_s = ttft_prefill if ttft_prefill == ttft_prefill else prefill_wall
    prefill_tok_s = prompt_tokens / prefill_s if prefill_s > 0 else float("nan")
    log(
        f"{ctx:7d} {conc:4d} prefill {gate:>4} "
        f"{prefill_s:8.3f}s  {prefill_tok_s:8.1f} tok/s  (prompt_tokens={prompt_tokens})"
    )

    rows = []
    for mode in ("old", "gqa"):
        setter(mode == "old")
        _, wall, _, n_gen, ttft, decode_s = _generate(llm, prompts, GEN_TOKENS)
        total_gen = sum(n_gen)
        # Prefer engine decode span; if metrics are missing, the prefix-cache
        # hit makes the whole wall ≈ decode, which is what we want.
        if decode_s != decode_s or decode_s <= 0:
            decode_s = wall
        decode_tok_s = total_gen / decode_s if decode_s > 0 else float("nan")
        row = Row(
            ctx=ctx,
            conc=conc,
            mode=mode,
            gate_expect=gate,
            prompt_tokens=prompt_tokens,
            gen_tokens=total_gen,
            ttft_s=ttft if ttft == ttft else prefill_s,
            decode_s=decode_s,
            decode_tok_s=decode_tok_s,
            e2e_s=wall,
            prefill_tok_s=prefill_tok_s,
        )
        rows.append(row)
        log(
            f"{row.ctx:7d} {row.conc:4d} {row.mode:<4} {row.gate_expect:>4} "
            f"{row.ttft_s:8.3f} {row.decode_s:8.3f} {row.decode_tok_s:9.2f} "
            f"{row.prefill_tok_s:13.1f} {row.e2e_s:8.2f}"
        )
    return rows


def main() -> int:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_METAL_COMPILED_MLP", "1")
    os.environ.setdefault("VLLM_METAL_NATIVE_SAMPLING", "1")

    if os.path.exists(LOG):
        os.remove(LOG)

    from vllm_metal.metal import get_ops

    ops = get_ops()
    log(f"model={MODEL}")
    log(f"tag={TAG} log={LOG}")
    log(f"has_gqa={ops.has_gqa_decode_kernel()} min_seq={ops.GQA_DECODE_MIN_SEQ_LEN}")
    log(
        f"compiled_mlp={os.environ.get('VLLM_METAL_COMPILED_MLP')} native_sampling={os.environ.get('VLLM_METAL_NATIVE_SAMPLING')}"
    )
    log(
        f"gen_tokens={GEN_TOKENS} max_model_len={MAX_MODEL_LEN} max_num_seqs={MAX_NUM_SEQS}"
    )
    log(
        f"require_fast={REQUIRE_FAST} min_tflops={FAST_MIN_TFLOPS} wait_s={FAST_WAIT_S}"
    )
    tflops = (
        wait_for_fast(FAST_MIN_TFLOPS, FAST_WAIT_S) if REQUIRE_FAST else probe_tflops()
    )
    log(
        f"probe_tflops={tflops:.1f} state={'fast' if tflops >= FAST_MIN_TFLOPS else 'slow'}"
    )
    if REQUIRE_FAST and tflops < FAST_MIN_TFLOPS:
        log(
            f"FAST_UNREACHABLE after {FAST_WAIT_S}s (best {tflops:.1f} < {FAST_MIN_TFLOPS})"
        )
        log("DONE")
        return 2

    setter = install_gqa_switch()

    from transformers import AutoTokenizer
    from vllm import LLM

    log("loading LLM…")
    t_load = time.perf_counter()
    llm = LLM(
        model=MODEL,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        trust_remote_code=True,
        disable_log_stats=False,
        enable_prefix_caching=True,
    )
    log(f"loaded in {time.perf_counter() - t_load:.1f}s")
    tflops_loaded = probe_tflops()
    log(
        f"probe_tflops_loaded={tflops_loaded:.1f} state={'fast' if tflops_loaded >= FAST_MIN_TFLOPS else 'slow'}"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    rows: list[Row] = []
    log(
        f"{'ctx':>7} {'conc':>4} {'mode':<4} {'gate':>4} "
        f"{'ttft_s':>8} {'dec_s':>8} {'dec_tok/s':>9} {'prefill_tok/s':>13} {'e2e_s':>8}"
    )
    for ctx, conc in CASES:
        try:
            rows.extend(run_pair(llm, tokenizer, setter, ctx, conc))
        except Exception as exc:  # noqa: BLE001
            log(f"{ctx:7d} {conc:4d} FAIL {type(exc).__name__}: {exc}")

    # Speedup table
    log("\n=== decode tok/s speedup (gqa / old) ===")
    by = {(r.ctx, r.conc, r.mode): r for r in rows}
    for ctx, conc in CASES:
        old = by.get((ctx, conc, "old"))
        gqa = by.get((ctx, conc, "gqa"))
        if not old or not gqa:
            continue
        sp = gqa.decode_tok_s / old.decode_tok_s if old.decode_tok_s else float("nan")
        log(
            f"ctx={ctx:<6} conc={conc}  old={old.decode_tok_s:7.2f} tok/s  "
            f"gqa={gqa.decode_tok_s:7.2f} tok/s  speedup={sp:5.2f}x  gate={gqa.gate_expect}"
        )

    tflops_end = probe_tflops()
    log(f"probe_tflops_end={tflops_end:.1f}")
    payload = {
        "tflops": tflops,
        "tflops_loaded": tflops_loaded,
        "tflops_end": tflops_end,
        "model": MODEL,
        "tag": TAG,
        "rows": [asdict(r) for r in rows],
    }
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    log(f"wrote {JSON_OUT}")
    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
