# Earlier grouped NAX measurements

These are historical measurements of the original h3/h6 and bounded h4
implementations. They do not validate the current adaptive selector.
See [current behavior and validation](gqa-prefill.md).

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
compiled layouts and resource validation. Other models with matching geometry have operator
coverage, not a blanket whole-model performance guarantee.

## Context length

There is no hard 32K total-KV admission limit. For GQA6, removing the earlier
calibration cap was checked on M5/Pro/Max with 100 paired operator case-runs
across the 32K boundary, 64K and approximately 128K KV. Outputs were bitwise equal and
no admitted path showed a repeated significant loss above 2% against actual
upstream or h3 in that matrix. This is operator evidence, not 128K model E2E
or a speedup guarantee at arbitrary lengths. Complete-work density, tail cost
and core capacity still gate admission.

## Earlier bounded GQA8 admission

The bounded GQA8 implementation added the page32/h4 instance to the same grouped shader. The ratio names
the model's Q/KV-head ratio; the threadgroup handles four of those query heads.
It reuses the existing work counter and guarded-pipeline score with no extra
group-state growth. The GQA6 h3/h6 policy remains unchanged.

Only the guarded upstream pipeline is eligible. A candidate is rejected if
it reduces the useful group count available to the detected GPU cores:
`min(cores, grouped.groups) < min(cores, baseline.groups)`. The remaining score
must exceed `0.5 + 32 * cores * grouped.tail_stream / baseline.issued`.
Upstream split/optimized requests retain their existing pipeline. Thus long
cold prompts need not use GQA8 grouping, while cached short queries can qualify.
Head counts, Q/prefix work and GPU capacity drive this decision; there is no
model-name or chip-name allowlist and no fixed total-KV length limit.

The guarded score weights were calibrated with the earlier GQA8 implementation
(base `2a2a2d0`) on 10/20/40-core M5 devices. They are empirical work weights,
not predicted milliseconds or a guarantee for unseen shapes. Earlier h8
experiments retained some wins over h4; this bounded implementation does not
claim to be the fastest of every experimental layout. There is no 80-core GPU
measurement. That implementation's results are reported separately from that
earlier calibration and from the GQA6 serving measurements below.

## Bounded GQA8 validation and serving, commit `3fe4b06`

These measurements use feature commit `3fe4b06` on main `5e09bb9`. Negative
latency changes are faster. Operator timings and model serving are separate.

The paired native matrix includes upstream at that base, the retained experimental
h4/h8 layouts and GQA6 controls, in both execution orders. All 442 case-runs
were bitwise equal. No admitted path had a repeated significant loss above 2%
against upstream at that base or old h4 in this matrix. The canonical rows below
use H16/K2/D256, BF16, page32 and prefix31680. Admission varies by geometry and
core count, so the total reference comparisons differ between devices.

| GPU / memory | Q64 kernel change (two orders) | Q128 kernel change (two orders) | Paired case-runs |
|---|---|---|---|
| M5, 10 cores / 32 GiB | -32.32% / -31.77% | -45.00% / -39.00% | 142 |
| M5 Pro, 20 cores / 64 GiB | -17.16% / -16.73% | -31.05% / -29.02% | 150 |
| M5 Max, 40 cores / 48 GiB | -16.40% / -16.63% | -15.72% / -14.63% | 150 |

The bounded h4 addition is not the fastest of every experimental layout.
Retained repeated old-h8 wins are:

| GPU | Geometry | h4 latency change versus old h8 |
|---|---|---|
| M5 | H128/K16, Q208/P17409 | +8.15% / +7.84% |
| M5 Pro | H128/K16, Q32/P255 | +24.89% / +38.97% |
| M5 Max | H96/K12, Q96/P255 | +27.04% / +29.04% |

Serving uses the same Qwen3.6-35B-A3B-OptiQ-4bit checkpoint, MLX 0.32.4,
vLLM 0.33.0+cpu, BF16 attention, APC enabled, batch size one, 16 output
tokens and GPU-memory fraction .65. Each run has two independent workers per
arm; baseline explicitly disables `VLLM_METAL_NAX_GQA_PREFILL`, while the
enabled arm uses its default. M5 has operator coverage only; this checkpoint
was not forced into its 32 GiB serving budget.

Each run checks 52 main requests, with one warmup and two formal requests per
case/worker. Complete 16-token output IDs, Q63/64/65/128 work and native
dispatch events match the expected routes. Cached cases hit 31680 tokens;
priming requests hit zero.
Q64/Q128 select h4; Q63/Q65 retain upstream. Main TTFT changes are:

| GPU / order | Q63 main TTFT | Q64 main TTFT | Q65 main TTFT | Q128 main TTFT |
|---|---|---|---|---|
| M5 Max / ABBA | +2.63% | -9.55% | +1.02% | -8.40% |
| M5 Max / BAAB | -0.32% | -7.20% | +0.84% | -8.22% |
| M5 Pro / ABBA | +2.06% | -4.51% | +1.03% | -12.88% |

The independent `vllm bench serve` protocol has 16 formal samples per
case/worker (128 per run), after a distinct-salt prime and server warmup.
The first CLI sample is retained. Saved input IDs and decoded output texts
are checked completely across workers for each case; complete CLI output
token IDs are not asserted. These samples are not pooled with the main client.

| GPU / order | New Q tokens | Upstream CLI TTFT ms | Enabled CLI TTFT ms | TTFT change | E2E change | Zero-swap-in worker pairs |
|---|---|---|---|---|---|---|
| M5 Max / ABBA | 64 | 183.37 | 162.61 | -11.32% | -5.70% | 1/2 |
| M5 Max / ABBA | 128 | 191.74 | 177.49 | -7.43% | -3.94% | 1/2 |
| M5 Max / BAAB | 64 | 175.63 | 167.35 | -4.71% | -2.38% | 0/2 |
| M5 Max / BAAB | 128 | 183.09 | 172.54 | -5.76% | -2.90% | 2/2 |
| M5 Pro / ABBA | 64 | 234.51 | 218.84 | -6.68% | -3.01% | 2/2 |
| M5 Pro / ABBA | 128 | 253.87 | 230.15 | -9.34% | -4.39% | 2/2 |

Cold requests have 32768 input tokens, distinct salts and actual APC hits of
zero, with two requests per worker (eight per run). Native observations
cover every chunk/layer: 31 x 1056 + 32 query tokens, all upstream NAX, with
no GQA8 launch. Decoded texts match across arms.

| GPU / order | Upstream cold TTFT s | Enabled cold TTFT s | TTFT change | Zero-swap-in worker pairs |
|---|---|---|---|---|
| M5 Max / ABBA | 11.294 | 11.928 | +5.61% | 0/2 |
| M5 Max / BAAB | 12.591 | 11.897 | -5.52% | 0/2 |
| M5 Pro / ABBA | 18.066 | 18.053 | -0.07% | 2/2 |

Max cold results change sign with execution order; no cold-input speedup is
claimed. Max's first Q63 main-control slowdown does not repeat in the reverse
run, while its Q65 remains about 1% slower. Pro's Q63/Q65 main controls are
2.06%/1.03% slower. Reverse-run Max CLI Q128 includes a
worker pair that is 1.75% slower despite the lower overall mean. Keep these
observations rather than claiming every request or worker improves.

No valid run added swap-outs or raised OS memory pressure. Global swap-ins
are recorded without attribution to this model; the CLI table gives the number
of worker pairs with zero swap-ins across the whole CLI process. In particular,
not every CLI result has two such pairs. No across-worker confidence interval
or GPU-frequency/occupancy explanation is claimed.

The native MLX short reference is checked separately from the timed requests.
Both serving hosts record EXACT=1, TOP_K_MATCH=1, FAIL=0 per checked
upstream/enabled worker, with identical serving outputs in both arms;
the differing native prompt has
tied top-two reported reference scores at the first divergence. This is not
a claim of exact native-MLX output identity for both prompts.

Each device passes 532 related tests plus 118 enabled and 118 disabled native
parity cases. The CPU policy comparison preserves 20000 GQA6 decisions against
main and 20000 GQA8 decisions against the earlier policy. The independent
work-count oracle includes GQA8 head geometries. Mutation checks force upstream
fallback and zero GQA8 output: the new route/output tests fail, then pass after
exact restoration. The affected upstream RoPE/attention suites also pass 129
tests on both serving hosts. The source at that commit passed Ruff/format and mypy.
The new GQA8 GPU matrix and model requests are at most 32K; longer KV lengths
have CPU policy/work-count coverage here, without a new long-KV GQA8 hardware
performance claim. There is no 128K model E2E or 80-core GPU measurement.

PR #1171 was still open at validation. It changes upstream pipeline admission
and can reduce GQA8 eligibility; these results do not claim additive gains
against that proposal. D128/D512, other GQA ratios and h8 admission remain
separate work.

## Earlier GQA6 serving measurements

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

The feature owns native h3/h4/h6 regression and physical-page translation
tests (784-to-16 and 544-to-32) in `tests/test_nax_split.py`, independently of
event tracing. They assert the executed width after pipeline-capacity fallback
and compare translated,
interleaved physical storage against compact kernel-page storage and upstream
NAX. `tools/nax_prefill_parity.py` also requires all three widths under fixed host
planning, with poisoned padding and real physical-page translation. Geometry
cases that intentionally allow fallback are labelled separately. These are
correctness tests; simulated core counts are not hardware performance data.
