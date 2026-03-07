# Metal Kernel Sources

This directory contains two sets of Metal paged-attention shaders, both vendored from [mistral.rs](https://github.com/EricLBuehler/mistral.rs) (MIT license).

## `kernels/` — active (current)

Drop-in replacement for the HuggingFace kernels-community paged-attention shaders, originally vendored from an older version of mistral.rs. This is what `paged_ops.cpp` compiles and dispatches today via MLX.

| File | Purpose |
|------|---------|
| `utils.metal` | bfloat16 polyfill, operator overloads |
| `float8.metal` | FP8 E4M3/E5M2 encode/decode helpers |
| `attention/paged_attention.metal` | paged attention v1/v2 kernels |
| `cache/reshape_and_cache.metal` | write projected K/V into block cache |
| `cache/copy_blocks.metal` | block-level cache copy kernel |
| `convert_fp8.metal` | FP8 precision conversion kernel |

### Reference only (not compiled, kept for context)

- `paged_attention.mm` — PyTorch MPS dispatch (Obj-C++), replaced by `paged_ops.cpp`
- `cache.mm` — PyTorch MPS cache ops (Obj-C++), replaced by `paged_ops.cpp`

## `kernels_v1/` — next-generation (not yet wired up)

Latest Metal kernels from the mistral.rs repo. More mature than `kernels/`, with preliminary scaffolding for variable-length sequences and gpt-oss sink attention support.

| File | Purpose |
|------|---------|
| `utils.metal` | shared types and helpers |
| `float8.metal` | FP8 encode/decode helpers |
| `pagedattention.metal` | paged attention kernel (restructured) |
| `reshape_and_cache.metal` | K/V cache reshape kernel |
| `copy_blocks.metal` | block-level cache copy kernel |
| `gather_kv_cache.metal` | *new* — gather KV from non-contiguous blocks |
| `kv_scale_update.metal` | *new* — KV scale update for quantised caches |

## Deprecation plan

Neither kernel set will persist long-term. Both are slated for deprecation once we introduce first-class variable-length kernel support, which is a prerequisite for:

- Continuous batching
- Chunked prefill
- MQA Scorer speculative decoding
