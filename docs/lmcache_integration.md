# LMCache KV Caching on Apple Silicon (vllm-metal)

This document describes how to run LMCache KV caching with the vLLM Metal
backend on Apple Silicon, for local development. It is the reproducible dev-env
recipe behind the LMCache ↔ vllm-metal integration.

> Status: **working local-dev path**, verified end-to-end (KV round-trips
> through LMCache bit-exact on a live MLX cache). The automatic in-engine
> prefix-reuse connector is a follow-on; this path exposes explicit
> store/retrieve so the KV movement is fully observable and benchmarkable.

## Why this is non-trivial

vllm-metal keeps its paged KV cache as **MLX arrays in Apple unified memory**
(`MetalPagedKVCache`: per-layer `key_caches[i]` / `value_caches[i]`, each
`[num_blocks, block_size, num_kv_heads, head_dim]`). It does **not** drive
vLLM's standard v1 `KVConnector` hooks, so LMCache's stock
`LMCacheConnectorV1` (which binds to vLLM's torch paged tensors) has nothing
to attach to.

The bridge works because:

1. On Apple Silicon an MLX array and a `torch` CPU tensor can **share the same
   bytes** (unified memory). vllm-metal's `tensor_bridge.mlx_to_torch` exposes
   this near-zero-copy, and it is **bidirectional** — writing the torch view
   writes the MLX cache. (Verified.)
2. LMCache's CPU path is fully functional without CUDA: on a CUDA-less host
   `lmcache.c_ops` resolves to the vectorized, device-agnostic
   `python_ops_fallback.multi_layer_kv_transfer`.
3. The per-layer stacked view `[2, nb, bs, nh, hd]` is exactly LMCache's
   `NL_X_TWO_NB_BS_NH_HS` (flash-attention NHD) format.

## Prerequisites (dev env)

- Apple Silicon Mac (verified on **M4 Pro**, macOS 26.5.1).
- **Homebrew Python 3.12** — *not* a fbcode/`+meta` Python. The fbcode Python's
  `sysconfig` bakes in a broken macOS SDK link path and breaks native
  extension builds (pybind11 `undefined symbols`). Homebrew's
  `LDSHARED = clang -bundle -undefined dynamic_lookup` is correct.
- Rust toolchain (vllm-metal's `_rs` pyo3 extension is built with maturin).
- If behind a proxy whose `no_proxy` contains a bracketed IPv6 literal
  (`[::1]`), strip it — httpx/reqwest URL parsers crash on it.

## Install

```bash
# from a checkout of vllm-metal on Homebrew py3.12:
./install.sh                      # builds vLLM core + the metal plugin
# (Metal .metallib shaders ship prebuilt in the release wheel; if you lack the
#  full-Xcode Metal toolchain, copy them from the release wheel into
#  vllm_metal/metal/.)

# LMCache (host-memory / CPU build — no CUDA kernels needed):
NO_GPU_EXT=1 uv pip install -e /path/to/lmcache --no-build-isolation
```

## Run: force the Metal platform, then use LMCache

```python
# CRUCIAL: importing vllm can cache CpuPlatform before the metal plugin loads,
# silently running on CPU torch instead of MLX. Bootstrap first.
from vllm_metal.bootstrap import bootstrap_metal_platform
bootstrap_metal_platform(require_metal=True)   # sets VLLM_ENABLE_V1_MULTIPROCESSING=0

from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen3-0.6B", max_model_len=2048, gpu_memory_utilization=0.15)
```

Then, inside the worker (e.g. via `llm.llm_engine.collective_rpc`), use
`vllm_metal.lmcache_integration.MetalLMCacheConnector` against
`model_runner._paged_attention_runtime.kv_cache` to `store` / `retrieve` /
`lookup` KV. See `examples/lmcache_metal_roundtrip.py`.

## Verified results (M4 Pro, Qwen3-0.6B, 28 layers, MLX bf16)

- **Correctness:** live MLX KV stored to LMCache, zeroed in place, then restored
  from LMCache **bit-exact** (500 tokens, all 28 layers). Zero CUDA.
- **Benchmark (LMCache reuse vs recompute):**

  | prefix tokens | recompute prefill | LMCache retrieve | speedup |
  |---:|---:|---:|---:|
  | 256  | 120.6 ms | 6.4 ms  | 18.7× |
  | 512  | 241.3 ms | 9.2 ms  | 26.2× |
  | 1024 | 482.5 ms | 16.8 ms | 28.7× |
  | 2048 | 965.1 ms | 39.5 ms | 24.4× |

  LMCache store 6–11 GB/s, retrieve 4.5–7 GB/s through the pure-Python fallback
  transfer on unified memory (a native Metal transfer kernel would be faster).

## Known issues fixed along the way

- **Platform-cache race** (`bootstrap.py`): `import vllm` can cache
  `CpuPlatform` before the OOT metal plugin resolves → silent CPU fallback.
- **mlx-lm × transformers ≥5** (`compat.py`): `AutoTokenizer.register("Newline
  Tokenizer", …)` passes a string key; transformers 5.x reads `key.__module__`
  → `AttributeError`. Patched to tolerate string keys.
