# Configuration

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_METAL_MEMORY_FRACTION` | `auto` | `auto` allocates just enough memory plus a minimal KV cache, or `0.?` for fraction of memory |
| `VLLM_METAL_USE_MLX` | `1` | Use MLX for compute (1=yes, 0=no) |
| `VLLM_MLX_DEVICE` | `gpu` | MLX device (`gpu` or `cpu`) |
| `VLLM_METAL_USE_PAGED_ATTENTION` | `1` | Enable experimental paged KV cache |
| `VLLM_METAL_KV_SHARING_FAST_PREFILL` | `0` | Enable experimental Gemma4 YOCO fast prefill on the paged KV path |
| `VLLM_METAL_DEBUG` | `0` | Enable debug logging |
| `VLLM_METAL_MULTIMODAL_MODE` | `auto` | Multimodal serve mode: `auto`, `text-only-compat`, or `multimodal-native` |
| `VLLM_USE_MODELSCOPE` | `False` | Set True to change model registry to <https://www.modelscope.cn/> |
| `VLLM_METAL_MODELSCOPE_CACHE` | None | Specify the absolute path of the local model |
| `VLLM_METAL_PREFIX_CACHE` | (unset) | Set to enable prefix caching for shared prompt reuse |
| `VLLM_METAL_PREFIX_CACHE_FRACTION` | `0.05` | Fraction of MLX working set for prefix cache (0, 1] |

## Multimodal Serve Modes

- `auto`: use native multimodal loading by default, but fall back to the text-only compatibility path for known-incompatible checkpoints such as Gemma4 and Qwen3.5/Qwen3.6 FP8 conditional-generation wrappers.
- `text-only-compat`: force the text-only compatibility path only for known-safe checkpoints such as Gemma4 and Qwen3.5/Qwen3.6 FP8 conditional-generation wrappers. Other multimodal checkpoints stay on the native multimodal loader.
- `multimodal-native`: disable the compatibility fallback and keep the native multimodal path active when validating or developing real multimodal support.

## Gemma4 YOCO Fast Prefill

`VLLM_METAL_KV_SHARING_FAST_PREFILL=1` enables an experimental default-off optimization for eligible Gemma4 text models on the paged KV path. It runs the YOCO KV-shared decoder layers only on the selected logits positions during prefill, then scatters those hidden states back before the final norm and LM head.

This path requires `VLLM_METAL_USE_PAGED_ATTENTION=1` and is currently limited to Gemma4/Gemma4 text models with KV-shared layers. If the loaded model is not eligible, vllm-metal logs a warning and continues without fast prefill.

## Paged KV vs MLX KV Memory Settings

- MLX path (`VLLM_METAL_USE_PAGED_ATTENTION=0`): `VLLM_METAL_MEMORY_FRACTION` must be `auto`.
- Paged KV path (`VLLM_METAL_USE_PAGED_ATTENTION=1`): `VLLM_METAL_MEMORY_FRACTION` can be `auto` or a numeric fraction in `(0, 1]`.
- For paged KV with `VLLM_METAL_MEMORY_FRACTION=auto`, vllm-metal uses a default fraction of `0.9`.

| `VLLM_METAL_MEMORY_FRACTION` | `VLLM_METAL_USE_PAGED_ATTENTION` | Valid? | Notes |
|--|--|--|--|
| `auto` | `0` | Yes | MLX path |
| `auto` | `1` | Yes | Paged KV path (default); defaults to 0.9 internally |
| `0.7` | `1` | Yes | Paged KV path with explicit memory budget |
| `0.7` | `0` | No | Explicit fraction without paged KV is invalid |
