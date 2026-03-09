# Metal Kernel Sources

Metal paged-attention shaders vendored from [mistral.rs](https://github.com/EricLBuehler/mistral.rs) (MIT license).

## `kernels_v1/` — active

Compiled and dispatched by `paged_ops.cpp` via MLX's native Metal command encoder.

| File | Purpose |
|------|---------|
| `utils.metal` | shared types and helpers (bfloat16 polyfill) |
| `float8.metal` | FP8 E4M3/E5M2 encode/decode helpers |
| `pagedattention.metal` | paged attention v1/v2 kernels (with sink support) |
| `reshape_and_cache.metal` | write projected K/V into block cache |
| `copy_blocks.metal` | block-level cache copy kernel |
| `gather_kv_cache.metal` | gather KV from non-contiguous blocks into contiguous tensors |
| `kv_scale_update.metal` | KV scale update for quantised caches |

## Deprecation plan

This kernel set will be superseded once we introduce first-class variable-length kernel support, which is a prerequisite for:

- Continuous batching
- Chunked prefill
- MQA Scorer speculative decoding
