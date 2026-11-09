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
