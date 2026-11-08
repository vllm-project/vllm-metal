# TurboQuant KV Cache Compression

vllm-metal supports TurboQuant-based KV cache compression. Keys use per-block
affine quantization; values use a Walsh–Hadamard rotation followed by per-block
Lloyd-Max quantization. Quantize/dequantize runs natively on Apple Silicon via
MLX and Metal kernels. Quantization is lossy; model-quality impact depends on
the model, bit widths, context length and workload.

## Quick Start

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --additional-config '{"turboquant": true, "k_quant": "q8_0", "v_quant": "q3_0"}'
```

TurboQuant is controlled via vLLM's `--additional-config` JSON, not a separate environment variable.

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `turboquant` | `false` | Enable TurboQuant KV cache compression |
| `k_quant` | `"q8_0"` | Key quantization type (see table below) |
| `v_quant` | `"q3_0"` | Value quantization type (Lloyd-Max) |

### Supported Key Quant Types

K uses per-block affine quantization without a Walsh–Hadamard rotation.

| `k_quant` | Bits | Notes |
|-----------|------|-------|
| `q8_0`, `int8`, `uint8` | 8 | Higher-precision key option |
| `q5_0` | 5 | Good quality / size trade-off |
| `q4_0`, `int4`, `uint4` | 4 | Lower-memory key option; validate model quality |
| `int2`, `uint2` | 2 | Aggressive; noticeable quality loss |

### Supported Value Quant Types

V uses Lloyd-Max (non-uniform) quantization with a Walsh–Hadamard rotation. Values are mapped to precomputed centroids per bitwidth.

| `v_quant` | Bits |
|-----------|------|
| `q2_0` | 2 |
| `q3_0` | 3 |
| `q4_0` | 4 |
| `q5_0` | 5 |
| `q8_0` | 8 |

## Compression

Measured on a Qwen3-0.6B-shaped KV cache (28 layers, 4 KV heads, head_dim=128, block_size=16) vs fp16:

| Config | Compression | K mse | V mse |
|--------|-------------|-------|-------|
| `k_quant=q8_0`, `v_quant=q3_0` (default) | **2.56x** | 0.00002 | 0.03241 |
| `k_quant=q5_0`, `v_quant=q3_0` | 3.37x | 0.00154 | 0.03241 |
| `k_quant=q4_0`, `v_quant=q3_0` | 3.76x | 0.00658 | 0.03241 |
| `k_quant=uint2`, `v_quant=q3_0` | 4.92x | 0.16639 | 0.03241 |

At `max_model_len=32768` on Llama-3.2-1B, the default `q8_0/q3_0` configuration frees roughly 2.5x more context for the same KV memory budget.

## Requirements and Caveats

- **MHA and hybrid (SDPA + GDN linear attention) models are supported.** In hybrid models, only the SDPA layers are compressed; GDN recurrent state retains its configured state dtypes and is not quantized by TurboQuant.
- **MLA models are not supported.** Enabling `turboquant` on an MLA model raises `NotImplementedError` at startup rather than silently falling back.
- **Head dim must be 64, 128, 256, or 512** — sizes supported by the FWHT Metal kernel. Models outside this set are not supported yet.
- Quality is model-dependent. For production use, spot-check perplexity with your target config before rolling out aggressive settings (`int2`, `q2_0`).

Three-bit V selects among eight non-uniform centroids for each rotated
coordinate, with per-block scales. Rotation and Lloyd-Max centroids reduce
distortion; they do not make compression lossless. K errors affect attention
weights, while V errors affect their weighted sum. Validate both bit widths
against a BF16-cache baseline with the same model weights and workload.

## Known Quality Floors

The historical observations below are workload-specific, not guarantees for
every model. Compression ratios use the geometry and scale metadata from the
table above. In particular, the lowest-bit settings have produced severe output
degradation and should not be treated as interchangeable serving presets.

| Config | Compression | Quality guidance |
|--------|-------------|------------------|
| `q8_0` / `q3_0` | 2.56x | Default bit widths; validate against a BF16 cache on the target workload |
| `q8_0` / `q2_0` | 2.78x | A fluency dip has been observed; validate before using it to save memory |
| `q4_0` / `q3_0` | 3.76x | Lower-memory option with model-dependent quality loss |
| `int2` / `q3_0` | 4.92x | **Degraded**: topic drift and numeric artefacts have been observed; capacity benchmarks only |
| `int2` / `q2_0` | 5.82x | **Broken output observed**: degenerate repetition loops; not for serving |

## Examples

### Normal Compression

Use the default bit widths after validating quality on the target workload:

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --additional-config '{"turboquant": true, "k_quant": "q8_0", "v_quant": "q3_0"}'
```

### Aggressive Compression

For memory-bound workloads where the measured quality loss is acceptable:

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --additional-config '{"turboquant": true, "k_quant": "q4_0", "v_quant": "q3_0"}'
```

## Prefill Acceleration

Eligible TurboQuant prefills materialize the referenced KV pages and use the
existing NAX or tiled attention kernel. `VLLM_METAL_TQ_PREFILL=auto` enables this
when NAX is available (M5; `VLLM_METAL_DISABLE_NAX=1` makes it behave as `0`);
`1` opts into tiled prefill on other GPUs, and `0` disables it. The calibration
below covers M5 Pro, M3 Air and M1 Pro. Broader hardware and shape validation
is still required before enabling tiled prefill by default on M1–M4.

Eligibility uses the **new query tokens in the current scheduler chunk**:
The default minimum is
`max(128, head_dim // 2, ceil(256 * num_kv_heads / num_query_heads))`.
For signed K8 (`q8_0` or its equivalent `int8`) with V3 (`q3_0`), hd128,
Q/KV heads 8/2, 8/8, 32/8 or 16/2 and total visible KV context at least 8192,
the calibrated minimum is 64. Other K/V encodings retain the default formula,
including unsigned `uint8` keys. Each request uses its own context
and threshold; the cached prefix counts toward context but not new query tokens.
The lane supports single/multiple requests, mixed prefill/decode batches, and
prefix caching on or off. After a prefix hit, only the uncached suffix contributes
query tokens; the attention still reads the relevant cached history. Decode,
short suffixes, speculative verification, FP32, sliding-window and image-block
attention keep their existing paths. Only full-attention SDPA layers are
accelerated; GDN/linear layers are unchanged. Attention sinks remain unsupported
with TurboQuant.

The fused materializer reads the original strided packed cache and writes K/V
directly in FP16/BF16. Unpacking stays in registers; scale math and inverse FWHT
use FP32 arithmetic. There are no context-sized packed gathers or FP32 arrays.
It uses `mx.fast.metal_kernel`, JIT-compiled on first use for each dtype,
geometry and quantization-format specialization; warmed timings exclude this
initial compilation cost.

Dequantized pages are temporary. Shared physical prefixes are decoded once
per layer **per scheduler step**; each prefill chunk re-materializes its
referenced history. Scheduler-owned storage, block tables and their lifetime
remain authoritative.

The worker resolves the workspace allowance once, before KV allocation, and
passes that same value to each forward. Routing plans remain cached in the
existing per-forward, per-KV-group metadata. A fully selected batch reuses the
original sequence metadata without constructing query-reordering indices.

`VLLM_METAL_TQ_PREFILL_MAX_MIB=auto` reserves 2% of the device's recommended
working set, rounded up to 64 MiB, with a 256 MiB floor and 2 GiB ceiling before
applying a model-specific cap. For non-speculative serving, the cap covers all
eligible independent histories allowed by `max_model_len`, `max_num_seqs` and
`max_num_batched_tokens`, including page padding and mixed-batch routing
copies. It can reduce the reservation below 256 MiB for small configurations.
A scheduler chunk limits new queries, not the historical KV they can read.
Layers reuse one allowance.

The cap uses `max_model_len` when the worker plans the KV budget, before vLLM
auto-fits the context to available memory. Later auto-fit reductions do not
recompute or reclaim the reservation. Set an explicit `--max-model-len` to cap
the reservation using a shorter context at planning time.

Speculative configurations retain the device-based allowance. A number
explicitly overrides the allowance in MiB without applying the model cap;
`0` disables materialization. Set these variables before worker startup.
The existing cache planner subtracts the allowance **once, inside
`gpu_memory_utilization`**, before allocating KV blocks. This is a fixed
allowance, not permission to borrow currently free memory.

For example, on M5 Pro 64 GB with K8/V3, `max_model_len=4104`,
`max_num_seqs=1` and `max_num_batched_tokens=2048`:

| Model | Device-only reservation | Model-capped reservation | Additional KV budget |
|---|---:|---:|---:|
| Qwen3-0.6B-4bit | 1,088 MiB | 16.13 MiB | 1,071.87 MiB |
| Qwen3.5-0.8B BF16 | 1,088 MiB | 10.71 MiB | 1,077.29 MiB |

Both retain the accelerated path for their eligible prefills. The released
budget goes to the existing KV planner within the same memory-utilization
limit; these capacity figures do not imply a further TTFT speedup.

Admission counts final K/V, page indices, block tables and any mixed-batch
query/output copies. Independent histories add their sizes; shared physical
pages count once. An entire batch that fits avoids the split-copy charge.
Oversized histories fall back before materialization, while smaller requests
can still qualify. Each accelerated layer evaluates its output with `mx.eval`
and drains the GPU stream with `mx.synchronize` before returning. This adds a
host synchronization boundary in each scheduler step, so temporary K/V from
successive layers cannot accumulate. Normal model buffers remain in the
existing profiled execution budget.

The worker logs the reserved allowance, first lane activation and first budget
fallback. Unsupported activation dtypes, head dimensions and cache layouts
also report their fallback reason once. Debug logs include selected/fallback
request counts, gathered tokens and estimated bytes. Larger histories still need
more scratch space: this is bounded materialization, not constant-memory
streaming attention.

## Head-Dimension 128 Crossover

M5 Pro 64 GB, K8/V3, head dimension 128, eight query heads, FP16/BF16,
8K/32K KV histories and TF32 disabled. The table reports the range across both
precisions and history lengths. Ratios are compressed time divided by
materialized time; values below 1 mean materialization is slower. These are
warmed single-layer production-wrapper timings, including projection, cache
writes, planning, materialization and synchronization, not model TTFT.

| Backend | Q/KV heads | 16 new tokens | 32 | 64 | 96 | 128 | 256 |
|---|---|---:|---:|---:|---:|---:|---:|
| NAX | GQA 8/2 | 0.65–0.69× | 1.06–1.11× | 2.35–2.69× | 2.70–3.10× | 4.22–5.23× | 7.38–9.97× |
| NAX | MHA 8/8 | 0.56–0.60× | 0.90–0.95× | 1.95–2.28× | 2.32–2.61× | 3.56–4.31× | 6.00–8.07× |
| Tiled | GQA 8/2 | 0.52–0.61× | 0.90–0.95× | 1.67–1.72× | 2.37–2.51× | 3.07–3.25× | 4.24–4.96× |
| Tiled | MHA 8/8 | 0.47–0.51× | 0.77–0.80× | 1.35–1.51× | 1.89–2.18× | 2.50–2.91× | 3.89–4.37× |

On this device, the observed NAX GQA crossover is between 16 and 32 new tokens;
the advantage at 32 tokens remains small. NAX MHA and both tiled shapes
cross between 32 and 64 tokens. These measurements do not establish thresholds
for other GPUs or model shapes. Some configurations
still show timing variation of several percent; the shortest cases can vary
by about 10–13%, so small gains near the crossover need confirmation.

### M3 calibration and bounded admission

M3 Air, 24 GiB, tiled attention, K8/V3, FP16/BF16, 8K/32K total KV context
and TF32 disabled. Ratios use the same compressed/materialized median
definition as the M5 table above.

| Q/KV heads | 16 new tokens | 32 | 64 | 96 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|
| 8/2 | 0.72–0.88× | 1.37–1.70× | 2.69–3.29× | 3.86–4.72× | 2.98–3.54× | 4.07–4.78× |
| 8/8 | 0.67–0.72× | 1.18–1.33× | 2.19–2.61× | 3.16–3.76× | 2.68–3.10× | 3.73–4.40× |
| 32/8 (Qwen3-4B) | 1.36–1.59× | 2.62–2.94× | 3.81–4.42× | 4.42–5.12× | 4.81–5.37× | 5.70–6.59× |
| 16/2 (MiniCPM5-2B) | 1.42–1.68× | 2.56–3.15× | 2.88–3.68× | 4.33–5.26× | 4.19–5.13× | 5.25–6.61× |

The two model rows use their attention head geometry in the single-layer
fixture; they are not whole-model latency measurements. Fresh 64-token prompts
yielded only 0.81–0.93× across these geometries and precisions: materialization
was slower. A global 64-token cutoff would therefore regress this workload.
The 8K context gate is conservative, not a measured optimum; 2K contexts also
favored materialization in this matrix.

On M5 Pro 64 GB, the additional model geometries have these NAX ratios for
FP16/BF16 and 8K/32K contexts, with the same K8/V3 comparison:

| Q/KV heads | 64 new tokens | 96 | 128 |
|---|---:|---:|---:|
| 32/8 (Qwen3-4B) | 6.30–8.29× | 7.80–9.54× | 8.86–10.22× |
| 16/2 (MiniCPM5-2B) | 4.09–5.38× | 4.89–6.16× | 7.44–10.44× |

Admission uses 64 new query tokens only for signed K8/V3, hd128 with Q/KV
heads 8/2, 8/8, 32/8 or 16/2 and total visible KV context of at least 8192
tokens. FP16/BF16 in these tables describes activation precision, not additional
K/V encodings. Shorter contexts, other K/V encodings and other geometries
retain the original
`max(128, head_dim // 2, ceil(256 * KV_heads / Q_heads))` threshold. The context
condition is evaluated per request, so mixed batches may contain both policies.
The workspace cap uses the same encoding-aware policy and the minimum threshold
reachable up to `max_model_len`, including the extra independent histories
admitted by the lower cutoff.
Default hardware rollout is unchanged: M3 still requires
`VLLM_METAL_TQ_PREFILL=1`. Contexts above 32K were not calibrated, and ordinary
workspace rejection still applies before materialization.

```bash
# Actual model geometries, with the same long-context protocol:
PYTHONPATH=. VLLM_METAL_BUILD_FROM_SOURCE=1 MLX_ENABLE_TF32=0 \
  python tools/benchmark/tq_lane_verify.py --suite crossover-hd128 --tiled \
  --head-pairs 32:8 16:2 --query-tokens 16 32 64 96 128 192 256 \
  --reps 31 --warmup 5
# Context 0 means a fresh prompt (total KV length equals the query length):
PYTHONPATH=. VLLM_METAL_BUILD_FROM_SOURCE=1 MLX_ENABLE_TF32=0 \
  python tools/benchmark/tq_lane_verify.py --suite crossover-hd128 --tiled \
  --head-pairs 8:2 8:8 32:8 16:2 --query-tokens 64 96 128 192 \
  --context-tokens 0 256 2048 --reps 15 --warmup 3
```

The `crossover-hd128` suite forces only the query-count threshold in the
benchmark to compare both algorithms below the production cutoff. It retains
shape validation, the workspace limit and the normal synchronization boundary,
and rejects a sample if materialization did not run or any output is nonfinite.
Each record also reports `production_lane_selected`, separately from the
forced measurement. The older `crossover` suite continues to measure ordinary
routing, including fallbacks.

```bash
PYTHONPATH=. VLLM_METAL_BUILD_FROM_SOURCE=1 MLX_ENABLE_TF32=0 \
  python tools/benchmark/tq_lane_verify.py --suite crossover-hd128 --reps 31 --warmup 5
# The same matrix through tiled attention:
PYTHONPATH=. VLLM_METAL_BUILD_FROM_SOURCE=1 MLX_ENABLE_TF32=0 \
  python tools/benchmark/tq_lane_verify.py --suite crossover-hd128 --tiled --reps 31 --warmup 5
```

### M1 Pro tiled prefill

On an M1 Pro with 16 GPU cores and 32 GiB (macOS 26.6.2), the single-layer
crossover depends on how much KV history the new tokens attend to. The table
uses the Qwen3-4B geometry (32 query heads, 8 KV heads, head dimension 128)
with K8/V3. Each cell gives the range across FP16 and BF16 of the median
compressed time divided by the median tiled time. Total KV includes the new
tokens. The benchmark forces the tiled path even below its admission threshold.

| Total KV tokens | 16 new tokens | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|
| Fresh prompt | 0.90–0.96× | 0.95–1.28× | 1.08–1.81× | 1.60–2.49× | 2.62–4.04× |
| 8,192 | 1.89–3.15× | 3.63–6.15× | 4.40–5.97× | 4.91–7.87× | 5.65–8.47× |
| 32,768 | 2.14–3.42× | 4.35–6.89× | 5.12–7.99× | 5.41–8.28× | 5.75–8.75× |

```bash
PYTHONPATH=. MLX_ENABLE_TF32=0 python tools/benchmark/tq_lane_verify.py \
  --suite crossover-hd128 --tiled --head-pairs 32:8 --query-tokens 16 32 64 128 256 \
  --context-tokens 0 8192 32768 --reps 31 --warmup 5
```

With Qwen3-4B-4bit and BF16 activations, the paired whole-model benchmark
tested fresh prompts on an already loaded model. At 2,048 input tokens, median
time to first token was 15.09 s on the compressed path and 6.53 s on the tiled
path. Peak MLX allocation above the start of each request was 672 MiB and
258 MiB, respectively. All three measured pairs produced the same eight greedy
tokens.

### M3 whole-model policy comparison

Qwen3-4B-4bit (32 Q / 8 KV heads, hd128, 36 layers), BF16 activations and
K8/V3, with exactly 8192 cached tokens. Both arms used the same workspace
allowance and KV budget. The table reports in-process, instrumented vLLM TTFT
medians with engine multiprocessing disabled, without HTTP or serving queues.

| Uncached queries | Old TTFT (ms) | New TTFT (ms) | Old/new medians | Median paired ratio | Materialized layers old/new |
|---|---:|---:|---:|---:|---:|
| 32 | 907.97 | 1008.10 | 0.901× | 0.919× | 0/0 |
| 63 | 1859.21 | 1990.24 | 0.934× | 0.982× | 0/0 |
| 64 | 2151.62 | 539.12 | 3.991× | 3.886× | 0/36 |
| 65 | 2297.26 | 773.19 | 2.971× | 2.979× | 0/36 |
| 96 | 3205.37 | 760.99 | 4.212× | 4.045× | 0/36 |
| 127 | 4211.42 | 995.32 | 4.231× | 4.231× | 0/36 |
| 128 | 1028.18 | 1026.26 | 1.002× | 1.002× | 36/36 |
| 129 | 1229.25 | 1196.16 | 1.028× | 1.028× | 36/36 |
| 256 | 1929.08 | 1925.55 | 1.002× | 1.000× | 36/36 |

First-token outputs matched, with exact cache hits and the expected layer
dispatch. Each query reuse occupied one prefill scheduler step. The 64-token
materialization used 32.38 MiB within a 33.20 MiB allowance. These are gains
from the scoped policy change, unlike the forced single-layer algorithm
ratios above.

The unchanged compressed controls at 32 and 63 tokens were 11.0% and 7.05%
slower on the new side despite identical routing; the cause is unresolved.
Unchanged materialized controls at 128/129/256 were approximately flat.
The 65-token point varied substantially, with a new-side latency as high as
1094.78 ms and paired speedups of 2.062–3.043×. These ranges are observations,
not confidence intervals or a general serving-throughput/SLO claim.

The mixed probe used three independent 8192-token cached histories with
64/9/1 uncached tokens and eight generated tokens per request. Every measured
batch had the actual cumulative queries `[0, 64, 73, 74]` in one prefill
scheduler step. All 36 layers selected one request and retained two compressed
fallbacks under the new policy; the old policy retained all three fallbacks.
Cache hits were exact and eight-token continuations matched. Batch TTFT
(maximum of the three request TTFTs) was
2719.64 ms old versus 1030.55 ms new, a 2.639× median ratio; individual paired
ratios were 2.460–2.708×. Actual materialization workspace stayed at 34.12 MiB
within the shared 145.36 MiB allowance. Batch generation wall time was
3413.05 ms old versus 1818.65 ms new (1.877×). The one-query-token request is
a prefix-reuse fallback; this does not establish overlap with an already
running decode request or concurrent HTTP throughput.

### M5 Pro whole-model policy comparison

M5 Pro 64 GB, Qwen3-4B-4bit, NAX, BF16 activations, K8/V3 and an 8192-token
cached prefix. The in-process comparison controls workspace and KV capacity
between the old/new policies, with engine multiprocessing disabled. TTFT
values are medians; first-token outputs match.

| Uncached queries | Old TTFT (ms) | New TTFT (ms) | Old/new | Materialized layers old/new |
|---|---:|---:|---:|---:|
| 32 | 224.89 | 224.73 | 1.001× | 0/0 |
| 64 | 424.68 | 105.58 | 4.022× | 0/36 |
| 96 | 624.03 | 136.34 | 4.577× | 0/36 |
| 128 | 124.72 | 124.30 | 1.003× | 36/36 |

### M5 Pro HTTP serving comparison

The same model, precision and 8K prefix workload through streamed HTTP
completions, with one request at a time and eight generated tokens. The server
uses the standard multiprocessing topology (`VLLM_ENABLE_V1_MULTIPROCESSING=1`).
TTFT uses the upstream serving benchmark's request function and ends at the
first generated token received by the client. Both policies use the same
workspace allowance and KV capacity. Values are medians.

| Uncached queries | Old TTFT (ms) | New TTFT (ms) | Old/new | Materialized layers old/new |
|---|---:|---:|---:|---:|
| 32 | 246.56 | 246.64 | 1.000× | 0/0 |
| 64 | 446.45 | 127.82 | 3.493× | 0/36 |
| 96 | 646.07 | 158.46 | 4.077× | 0/36 |
| 128 | 147.14 | 146.36 | 1.005× | 36/36 |

Worker observations confirm the exact cache hit, uncached query length and
expected layer routing. Eight-token continuations match. Paired TTFT speedups
span 3.48–3.52× at 64 queries and 4.04–4.09× at 96; both unchanged controls
stay within 1% in their medians.
These HTTP results include serving overhead and should be kept separate from
the in-process results above. They cover single-request latency; concurrent
throughput, queueing under load and SLO goodput remain unmeasured. Agreement
on this fixed workload does not establish corpus-level model quality.

## Validation and Reproduction

`tests/attention/test_turboquant_prefill.py` checks the production wrapper using
upstream-allocated storage, real cache writes and native attention. Coverage
includes formats/dtypes, padded and translated pages, shared/independent
histories, mixed-output ordering, fallbacks, admission and cross-layer memory
bounds. Fused dequantization is compared with the independent Python decoder.

The attention microbenchmark uses the same fixture, with interleaved timings,
numerical error, actual dispatch and peak additional MLX memory. Suites are
`crossover`, `crossover-hd128`, `geometry` and `long`; add `--tiled` to test
the tiled backend:

```bash
PYTHONPATH=. VLLM_METAL_BUILD_FROM_SOURCE=1 MLX_ENABLE_TF32=0 \
  python tools/benchmark/tq_lane_verify.py --suite crossover
```

For whole-model TTFT, run both TQ paths in one warmed model with identical
quantization. The reference disables only the prefill planner. Prefix caching
is off so repeated prompts execute prefill:

```bash
PYTHONPATH=. MLX_ENABLE_TF32=0 VLLM_METAL_TQ_PREFILL=1 python tools/benchmark/tq_e2e_arm.py \
  --model /path/to/model --arm paired --prompt-tokens 8192 16384 \
  --max-tokens 1 --reps 3 --warmup 1 --output latency.json
```

JSON records vLLM's `first_token_latency`, wall time, actual layer dispatch,
workspace, MLX memory and runtime versions. TTFT summaries are medians of the
measured repetitions, excluding warmup. Missing TTFT or an inactive requested
TQ lane fails explicitly. This offline tool excludes HTTP and concurrent serving
queues.

For threshold calibration, the prefix probe accepts an exact cached history and
the number of query tokens that must remain uncached. The history must be a
multiple of the scheduler block size. The probe seeds one extra token so the
entire requested history can be cached, then verifies both the cache-hit count
and the remaining query count on every reuse. A partial hit fails the probe.

```bash
PYTHONPATH=. VLLM_METAL_BUILD_FROM_SOURCE=1 MLX_ENABLE_TF32=0 \
  VLLM_METAL_TQ_PREFILL=1 python tools/benchmark/tq_e2e_arm.py \
  --model /path/to/hd128-model --prefix-probe --prefix-tokens 8192 \
  --query-tokens 32 63 64 65 96 127 128 129 256 \
  --max-tokens 1 --reps 3 --warmup 1 --output prefix-8k.json
```

Use `--prefix-tokens 32768` for a longer history when the model and device can
hold it. These explicit prefix/query lengths determine the required model
context limit; `--prompt-tokens` is for the ordinary prompt-length sweep.
Without `--prefix-tokens`, the prefix probe retains its two-block history;
default query lengths are 1, 9 and 257. Query lengths are not appended-suffix
lengths: vLLM must retain at least one query to compute logits. Query lengths
must fit within `--batch-tokens` so the measured reuse occupies one scheduler step.

Each arm gets a fresh prefix seeded through the same production path, keeping
cached hidden states independent of the measured arm. Warmup and measured pairs
alternate arm order, and summaries report median reuse TTFT, actual lane calls
and greedy token agreement. Dispatch checks use the loaded model's production threshold,
so short seeds and MHA thresholds do not inherit a hard-coded GQA cutoff.
An eligible probe with no materialized layers fails rather than reporting a
fallback as a materialized result.

The hd128 microbenchmark can run on M3 with `--tiled`, and on M5 with both
backends. Use the same source and dependency versions for calibration. The
microbenchmark measures the two algorithms below the policy cutoff; the prefix
probe measures compressed attention against the current production policy.
To compare the previous threshold with the current production policy, add
`--compare-policies`. Both arms reset the cache and seed it through the old
policy, then execute normal production admission for the measured reuse. They
share a workspace reservation large enough for either policy and the same KV
budget; this isolates admission latency rather than measuring a capacity change.
The records include per-request cache hits, context-aware thresholds, actual
selected/fallback counts and prompt/output equality. `--max-tokens` may exceed
one in this mode to check greedy continuation; seeds still generate one token.

```bash
PYTHONPATH=. VLLM_METAL_BUILD_FROM_SOURCE=1 MLX_ENABLE_TF32=0 \
  VLLM_METAL_TQ_PREFILL=1 python tools/benchmark/tq_e2e_arm.py \
  --model /path/to/hd128-model --prefix-probe --compare-policies \
  --prefix-tokens 8192 --query-tokens 32 63 64 65 96 127 128 129 256 \
  --max-tokens 1 --reps 3 --warmup 1 --output policies-8k.json
# Distinct cached histories must be co-scheduled in full; verify the mixture
# of newly eligible prefill, short fallback and one uncached query token:
PYTHONPATH=. VLLM_METAL_BUILD_FROM_SOURCE=1 MLX_ENABLE_TF32=0 \
  VLLM_METAL_TQ_PREFILL=1 python tools/benchmark/tq_e2e_arm.py \
  --model /path/to/hd128-model --mixed-prefix-probe --compare-policies \
  --prefix-tokens 8192 --query-tokens 64 9 1 --max-tokens 8 \
  --reps 3 --warmup 1 --output policies-mixed-8k.json
```

Mixed summaries report the maximum per-request TTFT for each submitted batch;
each request's own TTFT remains in the raw records. Planner-observed steps
exclude later decode steps. These are offline instrumented probes, without
HTTP or external serving queues. Failed probes preserve their output and exit
nonzero; choose a new output path for a repeat. Enabling non-M5 devices by
default remains a separate rollout decision.

For HTTP latency and throughput, follow the
[macOS serving benchmark guide](https://github.com/vllm-project/vllm-metal/blob/main/docs/benchmarking-macos.md).
To compare the whole prefill lane with compressed attention, start the same
model with prefix caching off:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=1 MLX_ENABLE_TF32=0 \
  VLLM_METAL_TQ_PREFILL=1 vllm serve /path/to/model \
  --host 127.0.0.1 --served-model-name tq-prefill --dtype bfloat16 \
  --max-model-len 9216 --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --gpu-memory-utilization 0.7 --no-enable-prefix-caching --generation-config vllm \
  --additional-config '{"turboquant": true, "k_quant": "q8_0", "v_quant": "q3_0"}'
```

Run the upstream serving benchmark against that endpoint, then repeat with
`VLLM_METAL_TQ_PREFILL=0` for the compressed reference. Set `--max-concurrency 4`
to include concurrent requests; compare both arms at the same concurrency.

```bash
vllm bench serve --model /path/to/model --served-model-name tq-prefill \
  --backend openai --endpoint /v1/completions --dataset-name random \
  --random-input-len 8192 --random-output-len 32 --random-range-ratio 0 \
  --num-prompts 8 --num-warmups 1 --max-concurrency 1 \
  --ignore-eos --temperature 0 --seed 853 --save-result --save-detailed
```

The client reports median TTFT and throughput over the full HTTP workload,
including server queueing. These differ from the in-process TTFT probe above.

For the hd128 admission-policy comparison, use the
[HTTP reproduction package](https://gist.github.com/suntp/31fb918815b34ecf48313ea2087bed1e).
It provides a parameterized client and worker observer, the exact old-policy
switch, original serving records and a script that verifies the published
results. Both services keep the lane enabled and use the same model and
capacity, with exact 8K prefix hits and returned-token validation. Download the
package and follow its README; the in-process probe above does not replace
this HTTP comparison.

For teacher-forced perplexity, use a fixed corpus and score the same windows in
both paths. This isolates the prefill implementation:

```bash
PYTHONPATH=. MLX_ENABLE_TF32=0 python tools/benchmark/tq_e2e_arm.py \
  --model /path/to/model --quality-text /path/to/wikitext-test.txt \
  --quality-window 1024 --quality-windows 16 --output quality.json
```

Repeat with `--arm bf16` for an uncompressed cache, or `--arm tq --v-quant q4_0`
for another TQ format, writing separate output files. Keep weights, tokenization
and windows fixed; verify corpus/token-ID hashes match. Defaults score 16,368
tokens from the first 16 × 1,024-token windows without extra special tokens.
Paired runs report a window-bootstrap NLL interval. Record corpus source, split
and revision with results. TQ-vs-TQ parity does not measure quantization loss
relative to BF16; short-window perplexity does not establish long-context task
accuracy. Different attention arithmetic also means greedy outputs can differ.

For a whole-model long-context execution and memory check:

```bash
PYTHONPATH=. MLX_ENABLE_TF32=0 python tools/benchmark/tq_e2e_arm.py \
  --model /path/to/model --arm tq --prompt-tokens 131072 --max-tokens 32 \
  --reps 1 --warmup 0 --progress-interval 30 --output long-context.json
```

This reports progress, actual dispatch, peak active MLX allocation and peak above
the pre-request allocation. A single run establishes neither paired speedup nor
long-context retrieval quality.
