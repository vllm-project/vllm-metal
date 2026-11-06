# Grouped NAX prefill

`VLLM_METAL_NAX_GQA_PREFILL` defaults to `1`. Set it to `0` and restart workers
to retain single-head upstream NAX. Each dispatch executes one selected kernel.

The compiled candidate sets are **GQA6: h2, h3, h6** and **GQA8: h2, h4, h8**.
Here `h` is the number of query heads sharing one threadgroup, not the model's
Q/KV ratio. Grouped kernels use BQ16 and `32 * h` threads; upstream uses BQ64
and 128 threads. Wider groups share KV loads but reduce independent groups
and increase logical state. None is uniformly fastest.

## Supported requests

Automatic grouping requires MLX NAX, BF16, D256, one ordinary causal sequence,
no mixed suffix, sliding window, softcap or attention sinks. GQA6 uses kernel
page16; GQA8 uses page32. These are geometry constraints, not model names.
Larger physical pages may translate to either kernel-page size. Complete Q/KV
tiles omit bounds checks; partial tiles and causal masking remain guarded.
Neither the whole request nor the final physical page must be complete.

The adaptive replacement additionally requires exact CPU Q/KV lengths and the
calibrated dense interleaved KV strides. The model runner builds one immutable
length plan per forward and shares it across layers. Calls without that plan,
or with other valid strides, retain the previous h3/h6/h4 policy. Unsupported
geometry and insufficient pipeline capacity retain upstream. There is no
fixed 32K total-KV admission cap.

## Selection and work accounting

Selection has three layers:

1. Check features, compiled layouts and available device capacity.
2. Count each sequence's actual causal tile visits for upstream and all three
   grouped widths. Include issued/useful work, Q tails, KV edges, QK/PV
   operations, potential KV streams, logical state, threads and group counts.
3. Compare relative costs with the actual upstream pipeline **and** the retained
   h3/h6/h4 choice. Replace the retained choice only when the predicted benefit
   exceeds a margin. Finally check the selected pipeline's thread limit.

For a layout, the calibrated score combines bulk work, the longest tile stream
and potential KV-stream cost with a smooth fourth-power maximum, plus launch
cost. Dimensions scale matrix work and logical state; the kernel page and
compiled width determine the layout rate. GPU cores normalize available work.
These are source-operation estimates, not measured register occupancy, cache
traffic or milliseconds. Parameterizing dimensions does not qualify D128/D512:
they require compiled kernels and separate performance validation.

The minimum replacement margin is 15%, with an additional reserve of
`0.30 * min(1, cores / full_upstream_SIMD_groups) * full_work_fraction`.
This is a **predicted** margin, not a promise of 15% observed speedup. It prevents
complete but underfilled grids from receiving the same confidence as dense
work. Existing profitable h3/h6/h4 routes remain when predictions are close.
The retained GQA6 policy still contains its earlier 10-core guarded-h6
allowance; the new common comparison has no model or chip-name allowlist.

`nax_prefill_resources` sums sequences independently; it never combines two
partial requests into a complete block. It also accounts for the starting Q
remainder of mixed suffixes and value slicing. Independent block-loop tests
cover D128/D256/D512 accounting, all widths, both split modes, translated pages,
every Q16/Q64 and prefix32 residue, and longer boundaries. Multi-sequence and
mixed automatic admission remains disabled pending its own performance study.

## Calibration and independent validation

The adaptive coefficients were fitted on base `5e09bb9` using 64 predefined
geometries on M5 (10 GPU cores), M5 Pro (20) and M5 Max (40). The matrix varies
Q, prefix, absolute Q/KV heads and page tails without model names. It contains
2,304 paired grouped-versus-NAX observations (both upstream pipeline controls)
and 768 direct width comparisons, all in both execution orders. The robust
relative-cost fit weights direct width comparisons twice; order disagreement
and median absolute deviation reduce observation weight. Coefficients and the
15% plus capacity reserve were frozen before independent validation.

The independent set has 40 additional geometries, including Q/KV head counts
absent from calibration. All predictions were recorded before timing. It used
700 direct pairs against actual main-default routes and retained h3/h6/h4,
8 warmups and 16 paired cycles per order, with identical logical tensors and
one shader library across all hosts. Outputs were bitwise equal before and
after timing; logical-page reconstruction, input immutability and actual
pipeline/grid/thread identities were checked.

| GPU cores | Independent shapes | Changed from retained route | Main-relative kernel gains over 5% | Repeated losses over 2% versus main or retained route |
|---:|---:|---:|---:|---:|
| 10 | 40 | 31 | 32 | 0 |
| 20 | 40 | 24 | 24 | 0 |
| 40 | 40 | 12 | 13 | 0 |

A repeated loss requires both order medians above 2% and both paired-cycle
confidence intervals above zero. These intervals do not measure uncertainty
between independent server workers. The counts above are kernel evidence,
not serving or TTFT results; main already includes GQA6 optimization.

Native regression exercises every width with real physical-page translation,
including 784-to-16 and 1056-to-32, and compares interleaved physical storage
against compact storage and upstream output. Test-only core overrides establish
route correctness, not performance on the simulated core count. The parity
tool also passes exact CPU lengths and requires h2/h3/h4/h6/h8 launches.

## Serving validation

The baseline is actual main `5e09bb9`, with its default GQA6 optimization
enabled; the adaptive arm also uses the default-on flag. This is an incremental
comparison against main, not against forcibly disabled GQA6. Both arms use
MLX 0.32.4, vLLM 0.33.0+cpu, BF16 attention, APC, batch size one, identical
input IDs and 16 generated tokens. Models are Qwen3.6-35B-A3B-OptiQ-4bit
(GQA8, H16/K2) and Qwen3.8-27B-4bit (GQA6, H24/K4). GPU cores / RAM are
M5 10 / 32 GiB, Pro 20 / 64 GiB and Max 40 / 48 GiB.

Each comparison uses two independent workers per arm in ABBA order; Max 35B
also has a BAAB repeat because its cold result changed materially. Main
streaming measurements use one warmup and two formal requests per case/worker.
The separate official `vllm bench serve` uses eight formal cached requests
per case/worker, including the first CLI sample, after distinct-salt priming
and server warmup. There are three cached CLI groups for 35B and two for 27B.
Cold CLI runs use two requests per worker. Negative changes mean faster.

Explicit KV budgets are 1 GiB for 35B and 4 GiB for 27B. Every worker has
the expected APC capacity: 49 physical pages of 1056 tokens for 35B,
83 pages of 784 for 27B. The actual cache hits are 31,680 and 31,360 tokens,
respectively. Kernel pages are 32 and 16. Cold inputs have zero cache hits.

The main-client TTFT changes below keep all boundary controls, including
unchanged upstream routes. They are separate from official CLI measurements.

| GPU / model / order | Q16 | Q32 | Q63 | Q64 | Q65 | Q128 | Q144 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M5 / 35B¹ | -10.00% | -9.05% | -0.04% | -7.96% | -8.95% | -18.31% | — |
| M5 Pro / 35B | -11.46% | -8.98% | +0.22% | -5.11% | +0.12% | -9.85% | — |
| M5 Max / 35B ABBA | -14.99% | -16.43% | +1.29% | -6.09% | +1.46% | -7.60% | — |
| M5 Max / 35B BAAB | -13.10% | -14.47% | +1.02% | -7.56% | +1.40% | -7.52% | — |
| M5 / 27B | -0.12% | -0.07% | +0.03% | -0.12% | -22.27% | — | +0.00% |
| M5 Max / 27B | -1.80% | -0.09% | -0.79% | -1.08% | -2.13% | — | -0.53% |

M5 27B Q65 changes from upstream to h3: 1099.73 → 854.79 ms (−22.27%).
Q144 retains h6. Other measured 27B cases are approximately flat against
main's already-enabled GQA6, rather than a new broad GQA6 serving gain.

| GPU / model / order | New Q tokens | CLI TTFT main → adaptive (ms) | TTFT change | E2E change | Zero-swap-in worker pairs |
| --- | --- | --- | --- | --- | --- |
| M5 / 35B¹ | 16 | 352.12 → 274.61 | -22.01% | -9.78% | 2/2 |
| M5 / 35B¹ | 64 | 409.21 → 391.78 | -4.26% | -3.71% | 1/2 |
| M5 / 35B¹ | 128 | 537.92 → 426.50 | -20.71% | -12.25% | 1/2 |
| M5 Pro / 35B | 16 | 179.55 → 158.75 | -11.58% | -4.31% | 1/2 |
| M5 Pro / 35B | 64 | 235.26 → 226.92 | -3.54% | -0.76% | 2/2 |
| M5 Pro / 35B | 128 | 250.99 → 229.14 | -8.71% | -4.39% | 1/2 |
| M5 Max / 35B ABBA | 16 | 161.32 → 142.12 | -11.90% | -5.88% | 0/2 |
| M5 Max / 35B ABBA | 64 | 204.37 → 180.65 | -11.61% | -6.62% | 0/2 |
| M5 Max / 35B ABBA | 128 | 205.71 → 168.60 | -18.04% | -10.10% | 1/2 |
| M5 Max / 35B BAAB | 16 | 164.96 → 139.73 | -15.30% | -7.42% | 2/2 |
| M5 Max / 35B BAAB | 64 | 189.88 → 189.69 | -0.10% | -0.07% | 1/2 |
| M5 Max / 35B BAAB | 128 | 192.17 → 168.85 | -12.14% | -6.31% | 1/2 |
| M5 / 27B | 64 | 587.24 → 589.69 | +0.42% | +0.04% | 0/2 |
| M5 / 27B | 144 | 1182.32 → 1171.67 | -0.90% | -0.28% | 0/2 |
| M5 Max / 27B | 64 | 288.54 → 283.86 | -1.62% | -0.80% | 0/2 |
| M5 Max / 27B | 144 | 410.08 → 399.24 | -2.65% | -1.36% | 1/2 |

¹ For M5 35B, the official client runs on Pro through SSH forwarding to the
M5 loopback server. The model, main streaming client and route checks remain
on M5. Both arms use identical transport, runtime, tokenizer and input IDs;
do not compare or pool these absolute CLI latencies with local-client runs.
The CLI mean includes client startup and transport variability; the separate
M5 main-client Q16/Q128 changes are −10.00%/−18.31%.

Cold 32K full-input controls are retained separately:

| GPU / model / order | Cold 32K TTFT main → adaptive (s) | Change | Zero-swap-in worker pairs |
| --- | --- | --- | --- |
| M5 / 35B¹ | 37.170 → 36.910 | -0.70% | 0/2 |
| M5 Pro / 35B | 17.956 → 17.956 | +0.00% | 0/2 |
| M5 Max / 35B ABBA | 11.121 → 12.057 | +8.42% | 0/2 |
| M5 Max / 35B BAAB | 11.582 → 11.108 | -4.09% | 1/2 |
| M5 / 27B | 155.095 → 155.041 | -0.04% | 0/2 |
| M5 Max / 27B | 45.539 → 45.292 | -0.54% | 0/2 |

There is **no stable cold 32K improvement**. Max 35B changes sign with execution
order (+8.42%, −4.09%), so neither a cold speedup nor a causal regression is
established. Its Q63/Q65 main-client fallback controls are approximately
1–1.5% slower in both orders. M5 27B official Q64 is +0.42%. These observations
are retained rather than discarded as noise.

Across the six comparisons, 464 main requests pass complete 16-token output,
APC and actual native dispatch checks; 512 official cached and 48 cold
requests pass their audit. CLI input IDs and complete decoded texts are
checked, but exact CLI output IDs are not asserted. Native MLX top-5 checks
report 1 EXACT / 1 TOP_K_MATCH / 0 FAIL per checked 35B worker, and
2 EXACT / 0 TOP_K_MATCH / 0 FAIL per checked 27B worker. The M5 35B native
reference was generated on Pro with identical full-checkpoint file hashes.

All completed workers have zero new swap-outs and no OS pressure event.
Global swap-ins are reported per window, without source attribution. The
tables retain all samples and show how many independent worker pairs also
have a zero-swap-in window; that subset is not a substitute for the full data.
Most cold controls have no clean pair. There is no across-worker confidence
interval, and only two workers per arm.

Two earlier M5 35B attempts were stopped and excluded from performance
results: native reference loading added 36 swap-out pages; starting a local
official CLI added 16,780 pages and reached OS pressure level 2. Moving only
the official CLI to Pro completed the controlled comparison. Pro's 16 client
windows were independently checked for memory pressure and against its
300-second quiet-gated resource lease. These failures remain part of the
evidence; the finished run does not show that local CLI + model fits M5 RAM.

## Regression checks and remaining scope

Max passed 382 affected tests; M5 and Pro each passed 137 native/SDPA tests.
All three passed 121 enabled and 121 disabled parity cases. Two final CPU
benchmark-view tests also pass. Independent work-count, batch-tail,
mixed-Q-offset, build-invalidation, margin and wrong-view mutation checks
fail with the injected defect and pass after restoration. Ruff, formatting
and mypy pass. The final production Python formatting change has an identical
AST; the tested native artifacts and serving harness remain unchanged.

The conservative selector does not always choose the measured fastest width.
For example, M5 GQA8 H16/K2 Q128/P31680 has a direct h2-over-h4 gain near 17%,
but this cost model does not predict enough replacement margin and retains h4.
80-core GPUs have no hardware measurement. Routine model requests are limited
to 32K; longer-length CPU/resource coverage is not a new long-context serving
claim. Other dimensions, ratios, strides and batch layouts need independent
qualification before broadening adaptive admission. Pro 27B was not run.

PR #1171 was still open during this calibration. Both upstream pipeline
controls were measured, but the serving baseline is main `5e09bb9`; no additive
serving gain against that proposal is claimed. [Earlier measurements](gqa-prefill-history.md)
retain the original GQA6 and bounded GQA8 results, separately from this selector.
