# Grouped NAX prefill

MLX enables grouped NAX prefill by default. Set `VLLM_METAL_NAX_GQA_PREFILL=0`
and restart workers to disable it.

Grouped kernels support a single causal BF16 sequence with query/key and value
head dimensions of 256. The width counts query heads sharing one KV head per
threadgroup.

| Query heads per KV head | Kernel page size | Grouped widths |
| --- | --- | --- |
| 6 | 16 tokens | 2, 3, 6 |
| 8 | 32 tokens | 2, 4, 8 |

One cost rule scores ordinary NAX and every grouped width for both compact and
interleaved caches. Ordinary NAX runs unless a grouped width is predicted to be
more than 10% cheaper. The lowest-cost qualifying width wins.

Larger physical pages must be multiples of the kernel page size. Mixed batches,
nonzero query-block offsets, sliding windows, softcap, sinks, and unsupported
geometries follow their existing paths. An unknown GPU core count or an
insufficient pipeline thread limit selects ordinary NAX. A loose KV length
bound in a direct native call can change the width estimate. The shader uses
the actual sequence length.
