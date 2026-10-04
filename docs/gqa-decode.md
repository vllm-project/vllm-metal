# GQA decode routing

The default paged attention path selects between 256- and 512-token GQA partitions for the measured single-request geometries below.
Each partition has FP16/BF16 specializations for head dimensions 128/256
with kernel block16, plus head256 with kernel block32. These form 12 active
shader specializations.

This is a Metal-backend split-KV occupancy heuristic over precompiled
partitions. It uses a fixed work budget, counts only complete partitions,
and greedily tries 512, then 256: among eligible tiers it prefers
fewer splits. This is the same class of backend-internal scheduling choice
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

Let `C` be the detected GPU core count and `Q` the query-head count. Both
partitions use the same empirical budget of 33 SIMD groups per core.
Select the largest P satisfying `floor(KV_length / P) * Q >= 33 * C`;
otherwise use the established path. Equivalently, each partition starts at
`P * ceil(33 * C / Q)`. Admission and promotion share one rule, with no
head-ratio correction or separate admission budget. This is an empirical
policy, not a hardware identity or a prediction of the fastest partition at
every position.

On a 40-core GPU this gives:

| Q/KV/head dimension | Start P256 | Start P512 |
|---|---:|---:|
| 32/8/128 | 10,752 | 21,504 |
| 24/4/256 | 14,080 | 28,160 |
| 16/2/128 or 256 | 21,248 | 42,496 |

These values scale with the detected core count; they are not per-device or
per-model context tables in the implementation. Core-count scaling does not
establish cross-device performance or guarantee a speedup at every boundary.
The budget is shared by both partitions; it has no per-model exceptions.

Only complete partitions count toward selection. The producer, temporary
buffers and reducer still use `ceil(KV_length / P)` so the final partial
partition is processed. Selection is stateless, uses at most two integer
comparisons, and adds no startup benchmark or per-request performance probe.
`gqa_decode_partition_size` exposes the default decision for tests;
`gqa_decode_shape_eligible` is true when it selects a nonzero partition.

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

- One pure-decode request, with `num_decode_requests` equal to 1 or omitted.
- A verification window of at most 1.
- Matching FP16/BF16 query, key-cache and value-cache types.
- A kernel page size allowed above and sufficient reducer shared memory.
- No TurboQuant, attention sinks, logit soft-capping or sliding window.
- A known, positive GPU core count.

Other calls, including multi-request batches and unknown core counts, use the
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

## Routing contract

Kernel dtype, page layout, query-row shape, feature and resource checks determine
whether an operation can run. Default admission is a separate measured policy:
one ordinary decode request, the listed geometries and the two-tier grid budget.
The C++ boundary enforces both, including for direct primitive callers.

The single-request planner uses post-append KV length, query/KV head geometry
and detected core count; its result is fallback or P256/P512. Execution sizes
its temporary buffers and reducer from that selected partition and the actual
query rows. Future batch planning must consider per-request lengths and the
parallelism supplied by the batch, while retaining this single-request case.

Mixed-batch integration must use an explicit decode sub-batch: active decode
request count, query/output row mapping, KV lengths and page-table mapping.
The whole batch count is not the decode count. Until that integration has its
own validation, mixed batches use their existing routes, including #851's
split where applicable. Removing the single-request guard alone is insufficient.

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

Default-policy positive route tests need a GPU core count and a sufficient
partition grid. Hosts whose IORegistry does not report cores inject a
test-only count through `_override_detected_gpu_core_count_for_test` so
the default selector still runs; the unknown-core fallback test forces
that count to zero. Production serving never calls the override.
`tests/test_gqa_paged_decode.py` tests kernel correctness separately through the private
`_gqa_paged_attention_for_test` entry, which selects an explicit partition
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
finally:
    mx.synchronize()
    ops._set_paged_dispatch_diagnostics(previous)
```

Enable recording in the worker that executes attention, before its measured
requests; enabling it only in the HTTP client does not observe the worker.
Both toggles clear the last observation. While disabled, the getters return
an empty family and partition zero. Recording changes neither routing nor
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

1. **Multi-request decode:** extend the grid, reduction and selection policy
   for different request lengths and batch sizes. This broadens the useful
   workload range; batches that already fill the GPU may need different
   choices from the single-request policy.
2. **Mixed prefill/decode:** integrate with the decode prefix split by merged
   [#851](https://github.com/vllm-project/vllm-metal/pull/851), after validating
   multi-request decode. Preserve row offsets, page tables and the prefill
   path; benchmark continuous batching rather than inferring its benefit
   from isolated decode. The integration point is clear, but correctness
   and admission need their own tests.
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
