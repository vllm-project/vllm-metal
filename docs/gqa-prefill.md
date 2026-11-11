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
request. A switch needs at least a 15% **predicted cost reduction**. Complete
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

## Calibration and validation

Paired measurements covered 10-, 20- and 40-core M5 GPUs, query and KV tails,
different head counts, and translated physical pages. The new measurement panel contained
60 shapes per GPU. The 10-core measurements completed after the cost was
frozen and served as additional prospective checks. Two disjoint panels of
32 further shapes per GPU checked the admission margin and the final policy.

| Minimum predicted gain | Calibration cases slower in both orders by over 2% | New margin-panel cases with reproducible losses over 2% |
| --- | ---: | ---: |
| 5% | 11 | 2 |
| 10% | 2 | 0 |
| 15% | 0 | 0 |

The final 96 device/shape combinations changed ten routes. Nine improved by
over 5% in both orders, with zero new reproducible losses over 2% against the
previous selector or the actual upstream route. Reproducible losses require
both paired confidence intervals to exclude zero. Measurements cover contexts
up to 32K. The coefficients are empirical estimates for the supported layouts.

Short cached serving remains sensitive to the first request of a new client.
On M5 Max, Q64 used the same h2 shader before and after recalibration. Initial
five-request batch means were 10.09% / 10.74% slower. A fixed 30-request control
gave -1.43% / +3.70% for the mean and -1.26% / -0.18% for the median. Every first
request was retained. The server's prefill measurements also varied, and the
specific source of that first-request variation remains unresolved. These
serving observations were excluded from coefficient fitting.
