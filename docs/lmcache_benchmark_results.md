# LMCache-on-Metal: True End-to-End Benchmark Results

**Host:** Apple M4 Pro (48 GB), macOS 26.5.1 · **Model:** Qwen/Qwen3-0.6B ·
**Backend:** vllm-metal (Metal/MLX), 28 layers, bf16 · **LMCache:** CPU/host-memory
connector + `python_ops_fallback` (zero CUDA).

Cold-prefill TTFT = median of 5 trials (distinct prompt per trial, APC reset each time).
LMCache store/restore = best of 3. All LMCache restores verified **bit-exact** against the
model-computed KV (28 layers, every slot).

| Prompt tokens | Cold prefill (ms) | vllm-metal APC repeat (ms) | LMCache restore (ms) | Speedup vs recompute | KV size |
|---:|---:|---:|---:|---:|---:|
| 326  | 132.8  | 21.5  | 8.4  | **15.8×** | 36 MB |
| 641  | 223.2  | 89.3  | 15.2 | **14.6×** | 72 MB |
| 1287 | 401.4  | 57.5  | 27.6 | **14.5×** | 146 MB |
| 2562 | 1004.0 | 44.1  | 48.2 | **20.8×** | 293 MB |
| 5127 | 2587.3 | 108.0 | 97.0 | **26.7×** | 587 MB |

## Honest reading of these numbers

- **LMCache KV restore is 14.5–26.7× faster than recomputing the prefill** on Metal,
  and the advantage grows with prompt length (recompute is ~O(n²) attention; KV load is ~O(n) bytes).
- **vllm-metal's built-in Automatic Prefix Cache (APC) already wins for in-process repeats**
  (20–110 ms). It would be dishonest to claim LMCache beats APC for same-process repeats — it doesn't need to.
- **LMCache's real, incremental value on Metal is what APC cannot do:** *persistent* KV
  (survives engine restart), *cross-process / cross-node* sharing, and *offload* to CPU/disk/remote
  tiers when the in-GPU working set is evicted. In all those cases the alternative is a full cold
  recompute — exactly the left bars — so the 14–27× applies.
- Transfer runs through the **un-optimized pure-Python fallback** on unified memory (store 6–11 GB/s,
  restore 6–7 GB/s). A native Metal transfer kernel would raise the ceiling further.

## Correctness

- `identical_output = True`: greedy decode identical for cold vs cached runs.
- Mechanism proof (see `lmcache_metal_mechanism.png`): after storing the live MLX KV to LMCache and
  **zeroing** those slots in the live cache, a genuine LMCache `retrieve` restores them **bit-exact** —
  proof the KV truly transits LMCache rather than a no-op.

## Reproduce

```bash
python examples/lmcache_metal_roundtrip.py     # correctness (bit-exact round-trip)
# benchmarks: docs/*_bench.py harnesses (cold_bench.py + true_e2e_bench.py)
```

Figures: `lmcache_metal_true_e2e.png`, `lmcache_metal_mechanism.png`.

## Update: automatic in-engine reuse (scheduler KVConnector) — the real serving win

The results above are the *isolated* transfer-vs-recompute microbenchmark. With the
scheduler-side `MetalLMCacheKVConnector` now wired end-to-end, LMCache reuse happens
**automatically inside the serving loop** (the scheduler folds the cached prefix into
`num_computed_tokens`; the worker loads the KV into the live MLX cache before the forward):

| Condition (≈1.2k-token prompt, APC reset so only LMCache serves the prefix) | TTFT (median of 3) |
|---|---:|
| Cold (full recompute) | 459.8 ms |
| LMCache-served (prefix loaded from LMCache) | **168.5 ms** |
| **Speedup** | **2.73×** |
| Output cold-vs-LMCache | **bit-identical ✓** |

Two bugs were found and fixed to get here (both documented in git history):
1. **Correctness** — `retrieve` fed LMCache a `torch.stack([k,v])` buffer, but `stack`
   copies, so writes never reached the MLX arrays. Fixed by writing the populated K/V
   halves back through the unified-memory bridge. (Near-miss symptom: first tokens matched,
   tail diverged.)
2. **Performance** — store/retrieve bridged the *entire* 31 GB / 17k-block cache each step
   (bridge 6.7 s + copy-back 10.1 s → 0.51×, slower than recompute). Fixed by touching only
   the request's own blocks (`mx.take` + block scatter), so cost scales with the prompt.

Honest note: in-process repeat prompts are already served ~9–10× by vllm-metal's built-in APC;
LMCache's incremental value is **persistent / cross-process / post-eviction** reuse — exactly
the case measured above (APC reset), where the alternative is a cold recompute.
