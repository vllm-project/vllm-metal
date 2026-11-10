# Grouped NAX prefill

The MLX NAX dispatcher can process three or six query heads from one six-head
KV group in a BQ16 threadgroup. `VLLM_METAL_NAX_GQA_PREFILL` defaults to `1`;
set it to `0` and restart workers to retain upstream NAX.

Eligibility is based on tensor geometry and attention features, not a model
name: BF16, head dimension 256, Q/KV head ratio 6, kernel page 16, one ordinary
causal sequence, no mixed suffix offset, sliding window, softcap or sinks.
Kernel pages may be translated from larger physical cache pages. Other shapes
retain their established route. Each dispatch executes one selected kernel.

Both layouts preserve the per-head calculation. h3 uses 96 threads and h6 uses
192; neither is always fastest. Complete query/KV tiles omit bounds checks,
while partial tiles and causal masking remain guarded. Whole-request alignment
is not required.

## Admission and calibration

`nax_prefill_work` counts the KV tiles visited by complete and partial query
blocks. It keeps issued work, useful work, guarded-query work and isolated KV
edges separate. `nax_gqa_work_score` compares each eligible grouped layout with
the actual upstream pipeline, using query/KV heads and detected GPU cores to
limit credit for parallel work. These counts are source-work proxies, not DRAM
traffic, register allocation, measured occupancy or predicted milliseconds.

The coefficients are manual conservative calibration weights. They were
calibrated with paired native-kernel comparisons on M5 (10 GPU cores), M5 Pro
(20) and M5 Max (40), varying Q, prefix length, Q/KV head count, page alignment
and execution order. The retained comparison includes upstream, h3 and h6;
beating upstream alone does not establish the fastest layout.

| Score component | Weights in the implementation | What the measurements constrain |
|---|---|---|
| Optimized upstream: padding, tail critical path, full work and guarded-Q savings | 13, 8, 2.5, 15 | A smaller Q tile can remove wasted/tail work, but grouping still costs extra KV streams and state |
| Optimized upstream: stream/state and guarded-Q cost | 10/h, 3.5, 16 | Do not assume shared KV makes all per-head loads disappear |
| Guarded upstream: partial work, critical path, reuse and non-tail guarded work | 4, 0.875, 3.5, 4 | Long-prefix short queries differ from dense cold-prefill blocks |
| Guarded upstream: lost grid capacity and additional group state | 10.5, 32, 3, 3.5, 0.75 | Wider groups must supply enough independent work for the GPU |
| Fixed/core cost, KV edges and guarded Q tails | 11.5, 2, 5 | Preserve a margin for launch/edge/tail work rather than crediting every complete tile as a win |
| Admission margin | 0.5 plus capacity-scaled tail reserve | A positive score alone is insufficient |

The source-work count is checked against an independent per-query-block loop
in `tests/test_block_size_translation.py`. It compares all work fields with
the compiled C++ helper, including every prefix residue modulo 32 and Q residue
modulo 16/64, both split modes, and longer full/tail/edge combinations. Correct
work counts do not make empirical coefficients universal.
Head dimension is an eligibility boundary here; this score is not a
cross-dimension performance model.

The six-head candidate normally requires at least one complete group per core
and enough complete SIMD-group work or KV work. A dense-work allowance requires
at least 32 complete h6 groups per core and at least 95% full-tile work. The
additional guarded h6 allowance is restricted to the measured 10-core range;
scaling that allowance by group/core counts alone produced regressions.

An earlier calibration run (base `93562f1`) produced the following
paired native measurements. They explain why the wider guarded allowance was not
extended to 20/40 cores. These are kernel timings, not model TTFT:

| GPU cores | Workload | Candidate/reference | Two-round latency change | Decision |
|---:|---|---|---|---|
| 10 | H24/K4, Q144/P31360 | h6 / retained h3 | -12.51%, -11.83% | Retain the 10-core guarded allowance |
| 20 | H48/K8, Q144/P24577 | broader h6 / retained h3 | +36.29%, +36.65% | Reject broad guarded admission |
| 40 | H96/K16, Q144/P24577 | broader h6 / retained h3 | +33.15%, +36.63% | Reject broad guarded admission |
| 80 | No available NAX execution host | Not measured | No performance result | Do not infer performance from CPU policy tests |

These rows have different head geometries and must not be pooled into a
cross-device speedup. Full results, exact measured artifacts, final-source
regressions and serving measurements belong in the accompanying PR evidence.

Known limitations remain: some odd-K selections trail historical experimental
layouts even while beating upstream; D128/D512 and other ratios need their own
compiled layouts and resource validation. The separate GQA8 implementation is
not part of this change. Other models with matching geometry have operator
coverage, not a blanket whole-model performance guarantee.

## Context length

There is no hard 32K total-KV admission limit. Removing the earlier calibration
cap was checked on M5/Pro/Max with 100 paired operator case-runs across the
32K boundary, 64K and approximately 128K KV. Outputs were bitwise equal and
no admitted path showed a repeated significant loss above 2% against actual
upstream or h3 in that matrix. This is operator evidence, not 128K model E2E
or a speedup guarantee at arbitrary lengths. Complete-work density, tail cost
and core capacity still gate admission.

## Serving measurements

The cached-request measurements in PR #1225 used feature commit `dadde59` on
base `14554be`. The later test/diagnostic follow-up leaves the shader sources and
admission policy unchanged; it is not a new serving measurement. The earlier
cold-input run below used base `93562f1` and is reported separately.

Both runs used the same local Qwen3.8-27B-4bit checkpoint files on M5 (10 cores),
M5 Pro (20) and M5 Max (40): H24/K4/D256, BF16 attention, APC enabled, batch size
one, 16 output tokens, MLX 0.32.4 and vLLM 0.32.0+cpu. GPU-memory fractions were
.75 on M5 and .50 on Pro/Max. ABBA used two independent workers per arm.

### Cached continuation, base `14554be`

The main protocol had one warmup and two formal requests per case/worker, with
31360 cached tokens. Q144 TTFT changes were -18.76% / -13.90% / -11.48% on
M5/Pro/Max. All 56 main requests per device passed complete 16-token-ID, APC,
CPU geometry and observed final-prefill-family checks. Four trailing Q1328
requests checked route/output only and were excluded from timings.

The boundary controls below retain small slower observations. Only M5 Q64
selects grouped h3; the other listed boundary routes stay upstream.

| GPU | Q63 TTFT change | Q64 TTFT change | Q65 TTFT change |
|---|---:|---:|---:|
| M5 (10 cores) | +0.13% | -10.83% | -0.02% |
| M5 Pro (20 cores) | +1.09% | -0.27% | +0.35% |
| M5 Max (40 cores) | +0.85% | -0.24% | -0.18% |

The separate `vllm bench serve` protocol used 16 formal requests per case/worker
(128 per device), one server warmup per case and the prefix retained after the
main protocol. The first CLI sample was retained. These samples are not pooled
with the main protocol or earlier runs with distinct cache salts.

| GPU | Q | Upstream CLI TTFT ms | Enabled CLI TTFT ms | TTFT change | E2E change | Zero-swap-in CLI worker pairs |
|---|---:|---:|---:|---:|---:|---:|
| M5 (10 cores) | 64 | 677.07 | 607.60 | -10.26% | -2.58% | 2/2 |
| M5 (10 cores) | 144 | 1550.02 | 1222.90 | -21.10% | -9.02% | 1/2 |
| M5 Pro (20 cores) | 64 | 397.78 | 394.40 | -0.85% | -0.28% | 0/2 |
| M5 Pro (20 cores) | 144 | 714.48 | 620.81 | -13.11% | -5.54% | 0/2 |
| M5 Max (40 cores) | 64 | 274.67 | 275.88 | +0.44% | +0.28% | 0/2 |
| M5 Max (40 cores) | 144 | 444.17 | 396.80 | -10.66% | -4.72% | 0/2 |

### Cold 32K input, earlier base `93562f1`

These are full-prompt serving requests, with actual APC hits of zero. Each
worker submitted two formal 32768-token requests with distinct salts (eight
requests per device), through `vllm bench serve`. The scheduler used
20 x 1568 + 784 + 624 query tokens, so this is complete prompt processing
through chunked prefill, not one 32768-row kernel launch. Native observations
covered the chunks/layers, and the enabled requests used grouped layouts.
Decoded texts matched across arms; this CLI protocol did not assert complete
output token IDs.

| GPU | Upstream cold TTFT s | Enabled cold TTFT s | TTFT change | E2E change | Zero-swap-in CLI worker pairs |
|---|---:|---:|---:|---:|---:|
| M5 (10 cores) | 165.666 | 155.836 | -5.93% | -5.87% | 0/2 |
| M5 Pro (20 cores) | 78.631 | 77.058 | -2.00% | -1.98% | 0/2 |
| M5 Max (40 cores) | 45.442 | 43.740 | -3.75% | -3.70% | 0/2 |

These earlier cold numbers are not a cold-serving retest on `14554be`. The
final-source regression checked cold priming outputs but made no new cold
performance claim. Neither run includes 128K full-model E2E.

### Measurement limits and regression coverage

Max main Q144 has a zero-swap-in subset covering only one of two worker pairs;
both complete pairs favor the candidate, but the full mean is not a fully
paging-free result. CLI windows include client startup and global swap-ins;
the cold run has no clean worker pair. No valid run added swap-outs or raised
OS memory pressure. Samples were retained without attributing global swap-ins
to this model, and no across-worker confidence interval is claimed.

Feature serving observed the final prefill family after natural sampling and
CPU chunk/layer geometry, without extra GPU fences. It did not collect per-layer
native function/grid/width events. Native-MLX short references recorded
EXACT=2, TOP_K_MATCH=0, FAIL=0 for each checked baseline/default worker.

Each device covered 1075 related cases (1066 initial passes, then the exact nine
MPS cases after building the missing extension) and 107+107 native parity cases
against tiled attention. FP32 was used for error calculation, not a complete
independent FP32 attention implementation for every case.

The feature owns its native h3/h6 and 784-to-16 page-translation regression
tests in `tests/test_nax_split.py`, independently of event tracing. They assert
the executed width after pipeline-capacity fallback and compare translated,
interleaved physical storage against compact kernel-page storage and upstream
NAX. `tools/nax_prefill_parity.py` also requires both widths under fixed host
planning, with poisoned padding and real physical-page translation. Geometry
cases that intentionally allow fallback are labelled separately. These are
correctness tests; simulated core counts are not hardware performance data.
