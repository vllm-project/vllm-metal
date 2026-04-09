# Configuration

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_METAL_MEMORY_FRACTION` | `auto` | `auto` allocates just enough memory plus a minimal KV cache, or `0.?` for fraction of memory |
| `VLLM_METAL_USE_MLX` | `1` | Use MLX for compute (1=yes, 0=no) |
| `VLLM_MLX_DEVICE` | `gpu` | MLX device (`gpu` or `cpu`) |
| `VLLM_METAL_BLOCK_SIZE` | `16` | KV cache block size |
| `VLLM_METAL_USE_PAGED_ATTENTION` | `0` | Enable experimental paged KV cache |
| `VLLM_METAL_DEBUG` | `0` | Enable debug logging |
| `VLLM_USE_MODELSCOPE` | `False` | Set True to change model registry to <https://www.modelscope.cn/> |
| `VLLM_METAL_MODELSCOPE_CACHE` | None | Specify the absolute path of the local model |
| `VLLM_METAL_PREFIX_CACHE` | (unset) | Set to enable prefix caching for shared prompt reuse |
| `VLLM_METAL_PREFIX_CACHE_FRACTION` | `0.05` | Fraction of MLX working set for prefix cache (0, 1] |

## Paged KV vs MLX KV Memory Settings

- MLX path (`VLLM_METAL_USE_PAGED_ATTENTION=0`): `VLLM_METAL_MEMORY_FRACTION` must be `auto`.
- Paged KV path (`VLLM_METAL_USE_PAGED_ATTENTION=1`): `VLLM_METAL_MEMORY_FRACTION` can be `auto` or a numeric fraction in `(0, 1]`.
- For paged KV with `VLLM_METAL_MEMORY_FRACTION=auto`, vllm-metal uses a default fraction of `0.9`.

| `VLLM_METAL_MEMORY_FRACTION` | `VLLM_METAL_USE_PAGED_ATTENTION` | Valid? | Notes |
|--|--|--|--|
| `auto` | `0` | Yes | MLX path (default) |
| `auto` | `1` | Yes | Paged KV path; defaults to 0.9 internally |
| `0.7` | `1` | Yes | Paged KV path with explicit memory budget |
| `0.7` | `0` | No | Explicit fraction without paged KV is invalid |
