# Metal Kernel Sources

Native paged-attention and linear-attention Metal shaders for the MLX backend.

All sources live in `kernels_v2/`. The legacy `kernels_v1/` path has been
removed (PR #378); v2 supersedes it on every axis (online-softmax `exp2`
trick, variable-length support, MLA, GDN, TurboQuant).

**License / provenance** (per the file headers themselves): Apache-2.0.
Portions of `utils.metal` and `pagedattention.metal` are adapted from
Apple's [MLX](https://github.com/ml-explore/mlx) framework (Apache-2.0,
© 2023 Apple Inc.); `pagedattention.metal` also adapts portions of the
[vLLM project](https://github.com/vllm-project/vllm) (Apache-2.0).
`turboquant.metal`, `pagedattention_tiled.metal`, and `mla.metal` are
vLLM-project Apache-2.0 sources.

## How the shaders are compiled

The shader set is defined explicitly in Python — the `.metal` files are
concatenated by name, never globbed, so source order matters (e.g.
`float8.metal` must precede `utils.metal`, which `#include`s it; the loader
strips local `#include "…"` directives). There are **four** library outputs
across **two** compile mechanisms:

### 1. C++ `_paged_ops` extension — `__init__.py`

`get_ops()` builds the nanobind extension, then initializes three JIT
Metal libraries from concatenated source:

| Library | Builder → init | Concatenated sources (in order) |
|---------|----------------|----------------------------------|
| **v2 paged attention** | `_build_v2_paged_attention_source` → `init_v2_library` | `#define VLLM_METAL_PARTITION_SIZE` · `float8.metal` · `utils.metal` · `turboquant.metal` · `pagedattention.metal` · `pagedattention_tiled.metal` |
| **GDN linear attention** | `_build_gdn_source` → `init_gdn_library` | `utils.metal` · `gdn_linear_attention.metal` |
| **MLA** | `_build_mla_paged_attention_source` → `init_mla_library` | `utils.metal` · `mla.metal` |

### 2. `mx.fast.metal_kernel` snippets — `attention/impls/gdn_lazy.py`

The lazy GDN decode fast path compiles two shaders directly through MLX
(not the C++ extension), via `_read_v2_metal_source`:

| Shader | Compiled kernel name |
|--------|----------------------|
| `gdn_conv1d_silu_decode.metal` | `gdn_conv1d_silu_decode_v2` |
| `gdn_recurrent_decode.metal` | `gdn_recurrent_v2` |

## File reference

| File | Role |
|------|------|
| `float8.metal` | FP8 E4M3/E5M2 encode/decode helpers. Concatenated first (no local includes). |
| `utils.metal` | Generic vector types and shared helpers; `#include`s `float8.metal`. Adapted from Apple MLX. |
| `pagedattention.metal` | Per-token paged-attention kernel with online softmax and sink support. Adapted from Apple MLX + vLLM. |
| `pagedattention_tiled.metal` | Tiled Flash-Attention-style kernel using simdgroup 8×8 MMA; independent of the per-token kernel, same library. |
| `turboquant.metal` | TurboQuant KV-cache compression (K: asymmetric uniform int8 / sub-8-bit; V: 3-bit Lloyd-Max + FWHT). Must be concatenated **after** `pagedattention.metal` declares `Vec<>`. |
| `mla.metal` | Paged Multi-head Latent Attention kernel (RFC #360). Single-pass mode is wired today; partitioned 2-pass mode is scaffolded but inactive. |
| `gdn_linear_attention.metal` | GDN (gated delta-net) linear-attention kernel for hybrid models (prefill / chunked path). |
| `gdn_conv1d_silu_decode.metal` | Lazy GDN decode: causal conv1d + SiLU. Compiled via `mx.fast.metal_kernel`. |
| `gdn_recurrent_decode.metal` | Lazy GDN decode: recurrent state update. Compiled via `mx.fast.metal_kernel`. |

## Unwired leftover shaders

These files exist in `kernels_v2/` but are **not referenced** by any build
or dispatch path (`__init__.py`, `paged_ops.cpp`, `build.py`,
`gdn_lazy.py`, or any shader `#include`). They are v2-directory
copies of old v1 kernels whose functionality is now handled MLX-natively.
PR #378 only removed `kernels_v1/`, so it does not touch these:

- `copy_blocks.metal`
- `gather_kv_cache.metal`
- `kv_scale_update.metal`
- `reshape_and_cache.metal`
