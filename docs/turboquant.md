# TurboQuant KV Cache Compression

vllm-metal supports TurboQuant-based KV cache compression: a Walsh–Hadamard rotation followed by per-block quantization that shrinks the KV cache by 2.5x–5x with minimal quality loss. Quantize/dequantize runs natively on Apple Silicon via MLX + a Metal kernel.

## Quick Start

```bash
VLLM_METAL_USE_PAGED_ATTENTION=1 vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --additional-config '{"turboquant": true, "k_quant": "q8_0", "v_quant": "q3_0"}'
```

TurboQuant is controlled via vLLM's `--additional-config` JSON, not a separate environment variable. Paged attention (`VLLM_METAL_USE_PAGED_ATTENTION=1`) is required.

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `turboquant` | `false` | Enable TurboQuant KV cache compression |
| `k_quant` | `"q8_0"` | Key quantization type (see table below) |
| `v_quant` | `"q3_0"` | Value quantization type (Lloyd-Max) |

### Supported Key Quant Types

K uses per-block affine quantization with a Walsh–Hadamard rotation.

| `k_quant` | Bits | Notes |
|-----------|------|-------|
| `q8_0`, `int8`, `uint8` | 8 | Near-lossless |
| `q5_0` | 5 | Good quality / size trade-off |
| `q4_0`, `int4`, `uint4` | 4 | Matches TurboQuant paper config |
| `int2`, `uint2` | 2 | Aggressive; noticeable quality loss |

### Supported Value Quant Types

V uses Lloyd-Max (non-uniform) quantization with a Walsh–Hadamard rotation. Values are mapped to precomputed centroids per bitwidth.

| `v_quant` | Bits |
|-----------|------|
| `q2_0` | 2 |
| `q3_0` | 3 |
| `q4_0` | 4 |
| `q5_0` | 5 |
| `q8_0` | 8 |

## Compression

Measured on a Qwen3-0.6B-shaped KV cache (28 layers, 4 KV heads, head_dim=128, block_size=16) vs fp16:

| Config | Compression | K mse | V mse |
|--------|-------------|-------|-------|
| `k_quant=q8_0`, `v_quant=q3_0` (default) | **2.56x** | 0.00002 | 0.03241 |
| `k_quant=q5_0`, `v_quant=q3_0` | 3.37x | 0.00154 | 0.03241 |
| `k_quant=q4_0`, `v_quant=q3_0` | 3.76x | 0.00658 | 0.03241 |
| `k_quant=uint2`, `v_quant=q3_0` | 4.92x | 0.16639 | 0.03241 |

At `max_model_len=32768` on Llama-3.2-1B, the default `q8_0/q3_0` configuration frees roughly 2.5x more context for the same KV memory budget.

## Requirements and Caveats

- **Paged attention is required** (`VLLM_METAL_USE_PAGED_ATTENTION=1`). TurboQuant cannot run on the MLX KV cache path.
- **MHA and hybrid (SDPA + GDN linear attention) models are supported.** In hybrid models, only the SDPA layers are compressed; GDN recurrent state stays at fp16 (it has no paged KV cache to quantize).
- **MLA models are not supported.** Enabling `turboquant` on an MLA model raises `NotImplementedError` at startup rather than silently falling back.
- **Head dim must be 64, 128, or 256** — sizes supported by the FWHT Metal kernel. Models outside this set are not supported yet.
- Quality is model-dependent. For production use, spot-check perplexity with your target config before rolling out aggressive settings (`int2`, `q2_0`).

## Known Quality Floors

Not every supported `(k_quant, v_quant)` combination is production-grade. K precision is load-bearing in a way V precision is not — K participates in the softmax, so quantization error gets **exponentially** amplified by `exp(QK^T)`, while V errors get **linearly** averaged across the context and largely wash out. This asymmetry means aggressive K quantization fails long before aggressive V quantization does.

Observed behavior:

| Config | Compression | Qualitative quality | Recommended use |
|--------|-------------|---------------------|-----------------|
| `q8_0` / `q3_0` | 2.56x | Matches bf16 within noise | **Default** |
| `q8_0` / `q2_0` | ~2.9x | Usable; minor fluency dip | Tight-memory serving |
| `q4_0` / `q3_0` | 3.76x | Paper-validated config; mild quality regression | Memory-bound workloads |
| `int2` / `q3_0` | ~4.1x | **Degraded**: semi-coherent, topic-drift, numeric artefacts ("2018 2018") | Capacity benchmarks only |
| `int2` / `q2_0` | ~4.8x | **Broken**: degenerate repetition loops ("concept concept concept…") | Not for serving |

## Examples

### Normal Compression

For production use with minimal quality impact:

```bash
VLLM_METAL_USE_PAGED_ATTENTION=1 vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --additional-config '{"turboquant": true, "k_quant": "q8_0", "v_quant": "q3_0"}'
```

### Aggressive Compression

For memory-bound workloads where some quality loss is acceptable:

```bash
VLLM_METAL_USE_PAGED_ATTENTION=1 vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --additional-config '{"turboquant": true, "k_quant": "q4_0", "v_quant": "q3_0"}'
```
