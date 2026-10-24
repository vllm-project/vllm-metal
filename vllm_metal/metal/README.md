# Metal Kernel Sources

Native paged-attention and linear-attention Metal shaders for the MLX backend.

All shader sources live in `kernels_v2/`.

**License / provenance:** Portions of `utils.metal` and `pagedattention.metal`
are adapted from Apple's [MLX](https://github.com/ml-explore/mlx) framework
(Apache-2.0, © 2023 Apple Inc.); `pagedattention.metal` also adapts portions of
the [vLLM project](https://github.com/vllm-project/vllm) (Apache-2.0).
`turboquant.metal`, `pagedattention_tiled.metal`, and `mla.metal` are
vLLM-project Apache-2.0 sources. `pagedattention_nax.metal` adapts MLX's
`steel_attention_nax` (MIT, © 2025 Apple Inc.).
`gdn_recurrent_prefill.metal` adapts mlx-lm's packed recurrence layout
(MIT, © 2025 Apple Inc.).

## How the shaders are compiled

The source builders concatenate shaders in the order below, stripping local
`#include "…"` directives.

### 1. C++ `_paged_ops` extension: `__init__.py` and `build.py`

By default, `get_ops()` loads the prebuilt nanobind extension and
required `.metallib` libraries, plus optional NAX support. With
`VLLM_METAL_BUILD_FROM_SOURCE=1`, it rebuilds changed artifacts with
`mlx.extension.MetalExtension`, then loads the generated `.metallib` files.
Packaging uses the same build via `python -m vllm_metal.metal.build`.
Source builds require the `build` extra and the Xcode Metal toolchain.

| Library | Source builder | Concatenated sources (in order) |
|---------|----------------|----------------------------------|
| **v2 paged attention** | `_build_v2_paged_attention_source` | `#define VLLM_METAL_PARTITION_SIZE` · `#define VLLM_METAL_PA_WINDOW_ROWS` · `float8.metal` · `utils.metal` · `turboquant.metal` · `reshape_and_cache.metal` · `pagedattention.metal` · `pagedattention_tiled.metal` |
| **GDN state operations** | `_build_gdn_source` | `utils.metal` · `gdn_linear_attention.metal` · `gdn_state_scatter.metal` |
| **MLA** | `_build_mla_paged_attention_source` | `utils.metal` · `mla.metal` |
| **NAX prefill** | `_build_nax_source` | `pagedattention_nax.metal` (optional; macOS 26.2 SDK or newer to build) |

### 2. `mx.fast.metal_kernel` snippets: `attention/impls/gdn_lazy.py`

The lazy GDN decode and prefill paths compile the following shaders directly through
MLX, via `_read_v2_metal_source`:

| Shader | Compiled kernel name |
|--------|----------------------|
| `gdn_conv1d_silu_decode.metal` | `gdn_conv1d_silu_decode_v2` |
| `gdn_conv1d_silu_prefill.metal` | `gdn_conv1d_silu_prefill_v2` |
| `gdn_recurrent_decode.metal` | `gdn_recurrent_decode_v2` |
| `gdn_recurrent_prefill.metal` | `gdn_recurrent_prefill_v2` |

Recurrent prefill uses one source with a compile-time `Packed` parameter. Within
the existing lazy prefill eligibility checks, key head dimension 128 and value
head dimension divisible by eight automatically select eight value rows per
SIMD-group. Other supported shapes retain the generic one-row layout. Both
specializations share state access and the sequence loop, while preserving their
respective reduction order. Selection is independent of GPU family and NAX;
pure decode retains its existing kernel. There is no separate packed user switch.

## File reference

| File | Role |
|------|------|
| `float8.metal` | FP8 E4M3 encode/decode helpers. Concatenated before `utils.metal`. |
| `utils.metal` | Generic vector types and shared helpers; includes `float8.metal`. Adapted from Apple MLX. |
| `reshape_and_cache.metal` | Scatter projected K/V into the paged cache through an MLX Primitive. |
| `pagedattention.metal` | Per-token paged-attention kernel with online softmax and sink support. Adapted from Apple MLX + vLLM. |
| `pagedattention_tiled.metal` | Tiled Flash-Attention-style kernel using simdgroup 8×8 MMA; shares the paged-attention library. |
| `pagedattention_nax.metal` | Optional M5 NAX paged-prefill kernel using MPP tensor operations. |
| `turboquant.metal` | TurboQuant encode/scatter and dequantization helpers: asymmetric uniform K quantization and Lloyd-Max V quantization with FWHT rotation. |
| `mla.metal` | Single-pass paged Multi-head Latent Attention. |
| `gdn_linear_attention.metal` | GDN (gated delta-net) linear-attention kernel for hybrid models. |
| `gdn_state_scatter.metal` | Scatter or zero cache/state rows, including updates to shared upstream storage. |
| `gdn_conv1d_silu_decode.metal` | Lazy GDN decode: causal conv1d + SiLU. |
| `gdn_conv1d_silu_prefill.metal` | Lazy GDN prefill: causal conv1d + SiLU. |
| `gdn_recurrent_decode.metal` | Lazy GDN decode: recurrent state update. |
| `gdn_recurrent_prefill.metal` | Lazy GDN prefill: recurrent state updates over packed sequences. |

### D512 NAX prefill

The split pipeline divides D512's output columns across two threadgroups,
each accumulating 256 columns. Both read the full 512-column Q/K vectors
and repeat QK and online softmax, then write disjoint output slices. This
halves live output accumulators without scratch buffers or a reduction pass.
Complete KV tiles in full Q blocks omit bounds checks; partial Q blocks and
partial KV tiles retain the guarded loads and masks.

The existing host tile-work hint enables this path with at least four full
Q blocks and eight times as much full-block as partial-block KV-tile work.
Short or partial-dominated launches, callers without sequence metadata, and
the MPS launcher retain the original D512 loop. Other head sizes keep their
existing kernels and selection rules.
