# Configuration

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_METAL_MEMORY_FRACTION` | `auto` | `auto` allocates just enough memory plus a minimal KV cache, or `0.?` for fraction of memory |
| `VLLM_METAL_USE_MLX` | `1` | Use MLX for compute (1=yes, 0=no) |
| `VLLM_MLX_DEVICE` | `gpu` | MLX device (`gpu` or `cpu`) |
| `VLLM_METAL_USE_PAGED_ATTENTION` | `1` | Enable experimental paged KV cache |
| `VLLM_METAL_DEBUG` | `0` | Enable debug logging |
| `VLLM_METAL_MULTIMODAL_MODE` | `auto` | Multimodal serve mode: `auto` / `text-only-compat` use the compatibility allowlist; `multimodal-native` disables overrides |
| `VLLM_USE_MODELSCOPE` | `False` | Set True to change model registry to <https://www.modelscope.cn/> |
| `VLLM_METAL_MODELSCOPE_CACHE` | None | Specify the absolute path of the local model |
| `VLLM_METAL_GDN_LAZY_KERNELS` | `1` | Enable lazy GDN kernels for eligible hybrid batches. Set to `0` to force the eager conv / C++ recurrent fallback path. |
| `VLLM_METAL_MLA_KERNEL` | `0` | Enable the experimental absorbed-MLA single-pass Metal decode kernel ([RFC #360](https://github.com/vllm-project/vllm-metal/issues/360)). Off by default; the MLA wrapper falls back to the MLX SDPA per-request slow path. Set to `1` to route absorbed-MLA decode through the kernel when the workload matches the instantiated specialization (`kv_lora_rank=512`, `qk_rope_head_dim=64`, `block_size ∈ {16, 32}`, fp16/bf16, decode-only). |
| `VLLM_METAL_VISIBLE_DEVICES` | — | Set automatically by the Ray executor per worker (the device-control var); not user-configurable. See [Distributed](distributed.md). |
| `VLLM_METAL_RING_BASE_PORT` | `32323` | Base TCP port for the MLX ring data plane under pipeline parallelism; stage *r* binds `base + r` (so the default is `32323`/`32324` for two stages). Set the **same** value on every node to move the ring off a busy port — e.g. when an `mlx.launch` job, a restart still in `TIME_WAIT`, or another PP job holds the default. See [Distributed](distributed.md#pipeline-parallelism). |

## Multimodal Serve Modes

- `auto`: use the text-only compatibility path for checkpoints on the compatibility allowlist, such as Gemma4 and Qwen3.5/Qwen3.6 FP8 conditional-generation wrappers.
- `text-only-compat`: use the same compatibility allowlist as `auto`.
- `multimodal-native`: disable the compatibility fallback and keep the native multimodal path active when validating or developing real multimodal support.

## Speculative Decoding

Pass `--speculative-config` with a JSON object to enable speculative decoding.
Use `--no-async-scheduling` (required for all spec-decode methods on Metal).
See [Speculative Decoding](speculative_decoding.md) for supported methods,
model pairing, and memory considerations.

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
