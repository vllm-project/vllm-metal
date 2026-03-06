# Metal Kernel Sources

Vendored from `kernels-community/paged-attention` (HuggingFace).

## Active (used by `paged_ops.cpp` via MLX dispatch)

- `utils.metal` — bfloat16 polyfill, operator overloads
- `float8.metal` — FP8 E4M3/E5M2 encode/decode helpers
- `attention/paged_attention.metal` — paged attention v1/v2 kernels
- `cache/reshape_and_cache.metal` — write projected K/V into block cache
- `cache/copy_blocks.metal` — block-level cache copy kernel
- `convert_fp8.metal` — FP8 precision conversion kernel

## Reference only (not compiled, kept for future use)

- `paged_attention.mm` — PyTorch MPS dispatch (Obj-C++), replaced by `../paged_ops.cpp`
- `cache.mm` — PyTorch MPS cache ops (Obj-C++), replaced by `../paged_ops.cpp`
