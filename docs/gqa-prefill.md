# Grouped NAX prefill

`VLLM_METAL_NAX_GQA_PREFILL` defaults to `1`. Set it to `0` and restart workers
to use ordinary NAX. One threadgroup can share KV across the following numbers
of query heads. Each invocation runs one selected layout.

| Q heads per KV head | Kernel page (tokens) | Query heads per threadgroup |
| --- | --- | --- |
| 6 | 16 | 2, 3, 6 |
| 8 | 32 | 2, 4, 8 |

Adaptive selection requires MLX NAX, BF16 attention, head dimension 256, one
ordinary causal sequence, exact CPU lengths and dense interleaved KV storage.
Larger physical pages may translate to the listed kernel page. Partial tiles
retain bounds and causal checks. Sliding windows, softcap, attention sinks and
mixed suffixes use their existing routes.

The selector compares each compiled width with the upstream route for the
request. A switch needs at least a 10% **predicted cost reduction**. Complete
query work adds a capacity reserve when the candidate has fewer independent
groups than detected GPU cores. Compact KV and calls without exact lengths retain
upstream selection, including established GQA6 h3/h6 routes. Unknown GPU
capacity or an insufficient pipeline thread limit falls back to ordinary NAX.

For long split-prefill calls the score includes the KV scan that the last
partially filled query tile must finish after the busy part of the launch.
For example, 1,056 new rows fill 66 grouped tiles of 16. Ordinary NAX fills
16 tiles of 64, then uses a guarded KV scan for its remaining 32 rows.
For each KV head, the score limits how far additional query tiles can reduce
the estimated cost of a wide group's state.

## Cost calibration

Paired measurements covered 10-, 20- and 40-core M5 GPUs, query and KV tails,
different head counts, and translated physical pages. A measurement panel contained
60 shapes per GPU. The 10-core measurements completed after the cost was
frozen and served as additional prospective checks. Two disjoint panels of
32 further shapes per GPU initially qualified a 15% margin.

| Minimum predicted gain | Calibration cases slower in both orders by over 2% | New margin-panel cases with reproducible losses over 2% |
| --- | ---: | ---: |
| 5% | 11 | 2 |
| 10% | 2 | 0 |
| 15% | 0 | 0 |

That study's final 96 device/shape combinations changed ten routes. Nine improved
by over 5% in both orders, with zero new reproducible losses over 2% against the
previous selector or the actual upstream route. Reproducible losses in that
study require both paired confidence intervals to exclude zero. The coefficients
are empirical estimates for the supported layouts.

## Ten-percent margin validation

A further experiment held the cost formula, capacity reserve and shaders fixed
and compared the 10% margin with the qualified 15% policy (`b8ae48b`). A fresh
32-shape panel per GPU changed 16 of 96 device/shape choices. All 16 improved in
both orders; 11 improved over 5% in both. This panel was not used to fit the cost.

A separate scan densely sampled two known tail-risk regions: Q8–39 with
64 query / 8 KV heads and prefix 24,577, and Q784–815 with 36 query / 6 KV
heads and prefix 9,408. Of 192 device/shape cases, 53 changed route. Twenty
improved over 5% in both orders. Twenty-four had timing medians over 2% slower
in both orders, all on the 40-core GPU; 21 also had both paired-cycle confidence
intervals exclude zero. No case was over 5% slower in both orders. The largest
single-order slowdown was 6.20%, with 2.80% in its reverse order. This deliberate
boundary sample is not an estimate of production request frequencies.

Serving used four fresh workers per model/device in 15%, 10%, 10%, 15% order,
with local clients pinned to the same version, prefix caching, a 2,048-token
scheduler budget and 16 output tokens. Each worker had two timed cold requests
and/or 12 timed requests per cached Q. Every first timed sample was retained.
Unique suffixes kept cache hits fixed even when Q reached a physical page.
The following changes are relative to the 15% policy; negative means faster.

**Qwen3.6-35B-A3B-OptiQ-4bit, M5 Pro (20 cores)**

| Cached prefix / new Q | Two-order TTFT change |
| --- | ---: |
| 0 / 32,768 | -2.24% / -2.11% |
| 31,680 / 65 | -2.14% / -0.92% |
| 31,680 / 769 | -4.03% / -4.05% |
| 31,680 / 1,040 | -2.87% / -2.86% |
| 31,680 / 1,056 | -3.05% / -2.92% |
| 31,680 / 1,057 | -2.78% / -2.86% |

Cold 32K changed 30 ordinary NAX chunks to h4. Q1,057 was actually dispatched
as Q1,056 plus a final single-token decode chunk. On M5 Max, cold layouts were
unchanged (TTFT +5.75% / -0.68%); the newly admitted Q769 improved only
1.31% / 0.14%. These observations do not establish a uniform serving benefit.

**Qwen3.8-27B-4bit**

Cached requests reuse 15,680 tokens.

| GPU / new Q | Two-order TTFT change |
| --- | ---: |
| M5 Pro / 671 | -0.66% / -0.73% |
| M5 Pro / 769 | -1.22% / -1.21% |
| M5 Max / 671 | -1.67% / -1.25% |
| M5 Max / 769 | -0.72% / +0.49% |

These cases changed ordinary NAX to h3. M5's tested
cached layouts were unchanged; its timing differences are controls, not gains
attributable to the margin. The complete validation covers 10/20/40-core M5
hardware and contexts up to 32K. A 35B startup on the 32 GiB M5 triggered the
system memory guard before any request, so no M5/35B serving result is claimed.

The 10% margin accepts bounded kernel regressions to enable these additional
layouts. It is not a guarantee of faster execution for every admitted shape.
Short cached serving remains sensitive to the first request of a new client.
An earlier same-h2 Max/Q64 control gave -1.43% / +3.70% mean changes versus
-1.26% / -0.18% for the median over 30 requests. No first requests were removed;
the specific source of that variation remains unresolved. Serving observations
were excluded from coefficient fitting.
