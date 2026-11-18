# Grouped NAX prefill

MLX enables grouped NAX prefill by default for a single ordinary causal BF16
sequence with head dimension 256. The compiled widths (query heads sharing one
KV head per threadgroup) are:

| Query heads per KV head | Kernel page size | Grouped widths |
| --- | --- | --- |
| 6 | 16 tokens | 2, 3, 6 |
| 8 | 32 tokens | 2, 4, 8 |

Ordinary NAX processes 64 query rows per head. Grouped NAX processes 16 rows
and shares KV reads across heads. Wider groups reduce repeated reads but give
the GPU fewer independent groups and require more state per group.

One rule scores ordinary NAX and every supported width. It estimates the
larger of total tile work per GPU core and the longest serial KV scan.
Grouped execution requires a predicted cost more than 10% below ordinary
NAX. The lowest eligible cost wins. Compact and interleaved caches use the
same rule, including larger physical pages translated to the kernel page size.

The selector uses the existing query shape and host KV length bound. Serving
supplies the sequence length. A loose bound in a direct native call can change
the width estimate. The shader reads the actual lengths and guards partial
tiles. Mixed batches, sliding windows, softcap, sinks and unsupported shapes
keep their existing paths. Unknown GPU capacity or an insufficient pipeline
thread limit selects ordinary NAX.

Set `VLLM_METAL_NAX_GQA_PREFILL=0` and restart workers to disable grouped
prefill. The [serving results](https://github.com/vllm-project/vllm-metal/pull/1300)
report measured gains and regressions.
