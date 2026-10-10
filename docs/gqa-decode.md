# GQA decode routing

The default paged attention path selects between 256- and 512-token GQA
partitions for ordinary single- and multi-request decode in the geometries below.
Each partition has FP16/BF16 specializations for head dimensions 128/256
with kernel block16, plus head256 with kernel block32. These form 12 active
shader specializations.

This is a Metal-backend split-KV occupancy heuristic over precompiled
partitions. It uses a fixed work budget, counts only complete partitions,
and tries 512, then 256. A measured M3 preference below chooses P256 for
some long-context batches that also qualify for P512. This is the same class
of backend-internal scheduling choice
as [FA3 split selection](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/heuristics.h),
[FlashInfer planning](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/scheduler.cuh),
and [ROCm partition choices](https://github.com/vllm-project/vllm/blob/main/csrc/rocm/attention.cu),
not an equivalent cost model. FA3 also favors fewer splits near its estimated
best efficiency; this implementation does not adopt its wave-efficiency model.

## Automatic eligibility and partition selection

The supported `(query heads, KV heads, head dimension)` geometries are
`(32,8,128)`, `(24,4,256)`, `(16,2,128)`, and `(16,2,256)`.
Kernel block16 is supported for each; block32 is additionally supported for
`(16,2,256)`. The existing scope is not inferred from a model-name list.
Geometry and kernel-page admission are checked at the native dispatch gate
before occupancy planning, including when a private test forces a partition.
One C++ table owns this domain. Numerical and routing tests read that table
through `_gqa_decode_config_for_test()` instead of maintaining another list;
golden threshold values and unsupported-input expectations remain independent.

Let `C` be the detected GPU core count, `Q` the query-head count, and `L_i`
each request's KV length including its current decode token. Both partitions
use the same empirical budget of 33 SIMD groups per core. First select the largest
`P` satisfying `Q * sum(floor(L_i / P)) >= 33 * C`; otherwise use the
established path. Flooring each length separately avoids counting partial
tails as complete work or treating short requests as copies of the longest.

The measured 10-core M3 (`applegpu_g15g`) uses an additional short-context
admission guard. A six-request `32/8/128` batch stays on the established path while
all KV lengths are below 704; adjacent batch sizes keep the common rule.
This guard conservatively excludes the measured short-context regression;
other admitted head128 batches retain their existing partition selection.

On that M3, multi-request head256 batches that qualify for P512 prefer P256
when the longest actual KV length reaches
4,096 tokens with kernel block16, or 8,192 with block32. This preference
keeps the work gate: it does not admit an otherwise ineligible batch. It
also requires P256's dynamic statistics plus its 64-byte static workspace to
fit 32 KiB, using the allocation upper bound as well as the actual lengths;
an oversized bound retains P512. Dispatch additionally checks the compiled
pipeline's actual static allocation. These are
device and layout calibration limits, independent of model names. Outside
the short guard, shorter contexts and head128 keep the common rule.
Single requests and other devices are unchanged.
The preference removes measured M3 P512 regressions without disabling
the positive long-context GQA cases. These are empirical policy limits, not a claim of the fastest route for every
prompt or a cross-device speedup claim.

The same complete-partition budget applies across batch sizes. The GQA
planner does not reject a batch solely because its unsplit query-head grid
exceeds a core-count threshold. The established per-token fallback retains
its own split-KV policy.

The single-request decision is unchanged: each partition starts at
`P * ceil(33 * C / Q)`. Admission and promotion share one rule, with no
head-ratio correction or per-model budget. This is an empirical policy, not a
hardware identity or a prediction of the fastest partition at every position.

For one request on a 40-core GPU this gives:

| Q/KV/head dimension | Start P256 | Start P512 |
|---|---:|---:|
| 32/8/128 | 10,752 | 21,504 |
| 24/4/256 | 14,080 | 28,160 |
| 16/2/128 or 256 | 21,248 | 42,496 |

These values scale with the detected core count; they are not per-device or
per-model context tables in the implementation. Core-count scaling does not
establish cross-device performance or guarantee a speedup at every boundary.
The budget is shared by both partitions; it has no per-model exceptions.

For example, two `32/8/128` requests on that GPU start P256 at 5,376 tokens
each and P512 at 10,752 each. Lengths `[1, 10752]` instead select P256:
the short request supplies no complete partition. Ten `32/8/128` requests
start P256 at 1,280 tokens each and P512 at 2,560 each on that GPU.

Only complete partitions count toward selection. The producer, temporary
buffers and reducer still use `ceil(KV_length / P)` so the final partial
partition is processed. The first ordinary decode layer builds an immutable
CPU length plan once per forward, shared across layers and KV groups. A single
pass records the request count, maximum length and complete-partition totals
for P256/P512. Each native layer copies this fixed-size value and applies its
geometry's policy without re-copying or re-scanning the request list. A new
forward gets a new plan; deferred execution retains the original value.
Single- and multi-request selection use the same planner. There is no GPU
readback, startup benchmark or per-request performance probe.
`gqa_decode_partition_size` exposes the default decision for tests;
`gqa_decode_shape_eligible` is true when it selects a nonzero partition.
`gqa_decode_batch_partition_size` takes a list of per-request lengths and
applies the M3 preference using the executing GPU's architecture by default.
Its optional `gpu_arch` is a read-only simulation input; it cannot change
actual dispatch. `max_seq_len` optionally supplies an allocation upper bound.
These queries check geometry, work and the scratch budget, not all dispatch
conditions. Direct primitive callers can pass `gqa_length_plan` from
`gqa_decode_length_plan(context_lens)`, or the original `gqa_context_lens` list;
the two inputs are mutually exclusive and must agree with GPU `seq_lens`.

Kernel block size is the view after hybrid-cache translation: a 1056-token
scheduler page selects block32, while a 784- or 528-token page selects
block16. Translated page IDs address their kernel-sized subpages even when
upstream K/V storage remains unreshaped.

The GQA producer walks pages rather than tokens. Lane 0 preloads the next
valid block-table entry and `simd_shuffle` broadcasts it within each SIMD
group. A 4-token inner step computes independent QK dots and combines their
online-softmax update; a scalar remainder handles the final 1-3 tokens.
Every tier reads K/V directly, without threadgroup staging or barriers.
Online-softmax state stays in registers.

The producer writes the same log2-space `(max, exp-sum)` plus
epsilon-normalized `tmp_out` contract as split-KV. The shared
`paged_attention_v2_reduce` therefore merges GQA partials unchanged:
the GQA producer does not require a separate reduction algorithm.

Every eligible call additionally requires:

- Ordinary decode rows with exactly one query row per request: either a
  whole pure-decode batch, or the leading decode rows that the native split
  separates from a mixed batch (see below). One request in a pure-decode
  batch retains the existing `num_decode_requests=1` or omitted-count
  convention. Multiple requests require matching `num_decode_requests` and
  `num_decode_tokens`, plus their CPU `gqa_context_lens` or
  `gqa_length_plan` metadata; a mixed-batch decode prefix always requires a
  `gqa_length_plan` describing exactly its rows.
- A verification window of at most 1.
- Matching FP16/BF16 query, key-cache and value-cache types.
- At most **512 MiB total GQA scratch per attention call**, including all
  rectangular padding: `B * Q * ceil(max_seq_len / P) * (2 * head_dim + 8)`
  bytes for the FP16/BF16 partial output and two FP32 statistics. Larger calls
  fall back before allocation. This ceiling covers the measured serving
  windows; it is not a reservation or a bound on total process memory.
- A kernel page size allowed above and sufficient reducer shared memory:
  aligned dynamic statistics plus the compiled pipeline's static allocation.
  This is checked before allocating GQA scratch or encoding the producer;
  an oversized `max_seq_len` allocation bound falls back even when the actual
  KV lengths pass the work budget. The reducer reuses the checked pipeline.
- No TurboQuant, attention sinks, logit soft-capping or sliding window.
- A known, positive GPU core count.

Other calls, including mixed batches without a decode-prefix plan, missing
batch lengths, expanded verification rows and unknown core counts, use the
established attention family. Model context limits, cache capacity and
primitive resource limits still apply. The 16/4/256 geometry remains
excluded from default routing.

## Disable switch

Set `VLLM_METAL_DISABLE_GQA_DECODE=1` before starting the server to keep
eligible requests on the established path. This switch provides an A/B and
operational fallback; it cannot enable an otherwise ineligible call.

The switch is captured once when each forward's `PagedAttentionContext` is
created and shared by its layers. The native module reports its public ABI
through one `paged_attention_capabilities()` query, also cached per forward.
Kernel availability is distinct from a supported production GQA route.
The `gqa_disabled` keyword is sent only when disabling GQA on a native build
that reports control support. Pre-GQA builds receive no new keyword; an
unrecognized GQA build without disable support requires a rebuild rather than
silently ignoring the switch. The structured query is part of the required
extension ABI; an artifact too old to export it is unsupported and fails at
the query instead of probing older entry points.
The `gqa_length_plan` capability advertises the per-forward plan. Builds with
only `gqa_batch_context_lens` receive the original optional length list;
an older structured query that omits it keeps its existing batch routing.

## Routing contract

Kernel dtype, page layout, query-row shape, feature and resource checks determine
whether an operation can run. Default admission is a separate measured policy:
ordinary decode, the listed geometries, the two-tier work budget and the
scoped M3 partition preference.
The C++ boundary enforces both, including for direct primitive callers.

The native planner consumes post-append KV lengths, query/KV head geometry
and detected core count; its result is fallback or P256/P512. For a whole
ordinary decode batch, SDPA forwards the scheduler's existing CPU lengths.
The primitive captures its own copy and includes it in lazy-node equivalence.
Direct callers must supply lengths consistent with `seq_lens`; shape and
positive-length bounds are checked eagerly, without reading GPU array data.

All requests use the same chosen partition. Scratch uses the maximum number
of partitions as its row stride. Each producer skips partitions beyond its
own KV length, and each reducer reads only that request's valid partials.
Neither padding nor stale scratch from a shorter row contributes to its result.

Mixed batches use an explicit decode sub-batch. When the native split
separates the leading ordinary decode rows from the prefill kernel (#851 for
tiled prefill, and NAX on M5), `sdpa_forward` passes a `GqaDecodeLengthPlan`
built from only those rows' context lengths (capability
`gqa_mixed_decode_plan`). The prefix call then treats rows and sequences
`0..D-1` as a pure decode batch: the GQA grid, reducer and scratch use `D`
requests and the plan's maximum length, not the batch-wide count or
`max_seq_len`. Without that plan, or with GQA disabled, the prefix keeps the
per-token kernel, whose split-KV partitions are likewise bounded by
`max_decode_context_len` rather than `max_seq_len` (which also covers the
prefill rows and may be an allocation bound). Admission counts only the
decode rows: a long prefill chunk never admits a short decode prefix. The
binding rejects a prefix plan whose request count differs from the decode
rows or whose maximum exceeds `max_decode_context_len`.

## Validation

`tests/test_gqa_decode_routing.py` evaluates the public primitive and checks
`last_paged_dispatch()` alongside the numerical reference. Positive cases
must actually report `gqa_decode`; partition tests additionally check
the executed 256/512 specialization through
`last_gqa_partition_size()`. Boundary, multi-request, verification, feature,
and disabled cases must report the appropriate established family, including
upstream mixed prefill/decode.
References include independent grouped CPU FP32 attention and native MLX SDPA.
All four geometries and both cache dtypes are checked at 128K, 192K,
and 256K, including a partial final partition.
These extended matrices are marked `slow` and excluded from regular CI;
all specialization, tail and routing-boundary checks remain in regular CI.
Run the long-context matrices explicitly with
`python -m pytest tests/test_gqa_paged_decode.py tests/test_gqa_decode_routing.py -m slow`.
Shared-storage tests use upstream-allocated
K/V views, non-contiguous page tables, native writes, prefix-page copying,
source-page clearing and a subsequent decode write. Dominant attention rows
make missing writes observable even in a long context.
The 1056-token upstream-page case also exercises the real block-table
translation and unreshaped K/V storage: the block32 kernel addresses each
translated page with its 32-token stride.
Remainder tests cover block16/block32 pages and final page lengths that
exercise both the 4-token loop and the scalar tail. Long-context page
tables use a bounded permutation to avoid inflating the physical cache.
Resource-limit tests reject oversized forced partitions before Metal
encoding, then check that a larger partition computes the same history.
`tests/test_attention_sdpa.py` checks that the environment switch and scheduler
decode count reach the primitive.
`tests/test_gqa_batched_decode.py` covers independent batch-planning boundaries,
ragged lengths, every shipped specialization and dtype, shared prefix pages,
translated scheduler pages, lazy writes, metadata lifetime and excluded routes.
`tests/test_gqa_m3_policy.py` checks fixed device/page boundaries, unchanged
single-request and other-device decisions, the reducer allocation limit and
executed numerical results while lazy batches change membership and partition.
The Python tests keep expanded verification metadata out of batch planning
and pass only the decode prefix of a mixed batch, regenerating the plan when
a later forward changes decode membership, order or lengths. Mixed-prefix
tests pin the GPU core count and require decode rows bit-identical to the
same rows as a pure GQA decode batch and prefill rows bit-identical to the
unsplit prefill kernel, for every shipped geometry, both cache dtypes and
translated scheduler pages; a short decode row next to a long prefill stays
per-token, and an allocation-wide `max_seq_len` does not size the prefix.

Default-policy positive route tests need a GPU core count and a sufficient
partition grid. Hosts whose IORegistry does not report cores inject a
test-only count through `_override_detected_gpu_core_count_for_test` so
the default selector still runs; the unknown-core fallback test forces
that count to zero. Production serving never calls the override.
`tests/test_gqa_paged_decode.py` tests kernel correctness separately through the private
`_gqa_paged_attention_for_test` entry, which accepts one query row per request
and selects an explicit partition
without changing global state or the public primitive's routing API.
Its partition is captured in the lazy primitive and its equivalence key.
These tests execute both partitions, both dtypes, every shipped shader
specialization, tails and upstream shared-page writes/copies even when CI
cannot report GPU cores. Numerical parity does not establish performance
eligibility. Feature/dtype/page-size fallback tests remain independent.
The library-availability check loads all 12 active GQA specializations and
their matching reducers even when the reported core count is unavailable.
Validate the positive path on a capable GPU before reporting GQA coverage.

Dispatch diagnostics are disabled by default. The ordinary dispatch path
reads one relaxed atomic flag and returns without writing diagnostic state.
Serial tests and isolated benchmark workers explicitly enable recording:

```python
import mlx.core as mx
from vllm_metal.metal import get_ops

ops = get_ops()
mx.synchronize()  # Toggle only while evaluation is idle.
previous = ops._set_paged_dispatch_diagnostics(True)
try:
    # Run and evaluate the attention operation in this process.
    ...
    family = ops.last_paged_dispatch()
    partition = ops.last_gqa_partition_size()
    requests = ops.last_gqa_num_requests()
finally:
    mx.synchronize()
    ops._set_paged_dispatch_diagnostics(previous)
```

Enable recording in the worker that executes attention, before its measured
requests; enabling it only in the HTTP client does not observe the worker.
Both toggles clear the last observation. While disabled, the getters return
an empty family and zero partition/request count. Recording changes neither routing nor
attention computation. These are process-wide diagnostics, not per-request
telemetry for concurrent serving. Evaluate the operation before reading its
record, because MLX builds graphs lazily.

For performance validation, rebuild native artifacts from the tested
revision and use the real `vllm serve` process topology. Record the model,
runtime versions, hardware, warmup, KV lengths, repetitions, and background
GPU activity; compare enabled and disabled arms with actual worker-side
dispatch checks. Separate primitive timing from HTTP decode throughput,
and keep noisy measurements visible. The
[macOS benchmarking guide](benchmarking-macos.md)
explains why in-process engine measurements and short probes are not
interchangeable with serving results.

## Follow-up work (separate PRs)

These are directions for evaluation after this scoped change, not additional
enablement or performance claims of this two-tier routing policy:

1. **GQA without split-KV:** evaluate a grouped kernel that writes its final
   output directly, independently of context partitioning. Measure its
   tradeoff against P256/P512 at larger batch sizes.
2. **Mixed prefill/decode:** the decode prefix split by
   [#851](https://github.com/vllm-project/vllm-metal/pull/851) (and its NAX
   counterpart) now uses GQA through a decode-prefix length plan (see the
   mixed-batch paragraph above). Its threshold (`kMixedDecodeMinContext`) is still
   the tiled-era 4096 tokens and has not been re-tuned for NAX or GQA.
3. **TurboQuant decode:** evaluate consuming packed KV/scales in the GQA
   kernel. This could complement the prefill optimization in
   [#853](https://github.com/vllm-project/vllm-metal/pull/853), which now handles
   bounded TurboQuant prefill. Avoid assuming that materializing the whole
   history on every decode step is cheap; validate format, memory and quality
   effects.
4. **Sliding windows:** bound loads and split selection by the actual visible
   KV window. This is relevant to Gemma/Mistral-style attention, but their
   geometries and other features such as sinks or soft-capping must also
   satisfy the supported kernel contract.
5. **Speculative verification:** evaluate sharing KV across both grouped heads
   and multiple query rows. The existing verification kernel already shares
   KV across rows; compare against it while preserving causal masks and
   controlling register pressure. A larger gain is possible, not established.
6. **Cooperative K/V loading:** evaluate shared staging against direct
   loads on each device, accounting for cache reuse, synchronization and
   occupancy. Context length alone does not establish a benefit.
