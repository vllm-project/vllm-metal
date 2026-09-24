# ggml Backend (experimental)

vllm-metal can run the model forward pass on [ggml](https://github.com/ggml-org/ggml)'s
Metal backend instead of MLX. The engine is written in Rust
(`rust/ggml-engine`) and exposed to Python through PyO3; vLLM scheduling and
sampling are unchanged.

```bash
brew install ggml rust
python -m vllm_metal.ggml.build        # or: ./install.sh from a source checkout

VLLM_METAL_BACKEND=ggml vllm serve Qwen/Qwen3.5-0.8B
VLLM_METAL_BACKEND=ggml vllm serve google/gemma-4-E2B-it
```

## Supported models

The engine loads unquantized Hugging Face safetensors (BF16/F16/F32 weights kept
in their checkpoint dtype) and serves the text backbone of:

| Family | Checkpoints | Notes |
|---|---|---|
| Qwen3.5 (dense) | `Qwen/Qwen3.5-0.8B`, other dense sizes | Gated DeltaNet linear attention + gated full attention |
| Gemma 4 (dense) | `google/gemma-4-E2B-it`, E4B | sliding/full attention, proportional RoPE, cross-layer KV sharing, per-layer embeddings |

MoE variants, quantized checkpoints, multimodal inputs, LoRA, speculative
decoding and pipeline/data parallelism are rejected at startup.

## How it works

* **Paged KV cache.** vLLM sees one synthetic attention layer whose page size
  equals the engine's per-block bytes across all layers, so every request gets
  a single block table. New K/V are written with `ggml_set_rows`; each step
  gathers the context K/V from the block table (`ggml_get_rows`) and runs
  `ggml_flash_attn_ext` once per group of sequences with equal query length.
  Sliding-window layers gather only the window.
* **Recurrent state.** Gated DeltaNet conv/SSM state lives in per-sequence
  slots (`ggml_ssm_conv`, `ggml_gated_delta_net`). Slots are owned by the
  runner, so prefix caching is disabled for hybrid models.
* **Per-layer embeddings** (Gemma 4 E-models) are looked up on the host from
  the memory-mapped checkpoint; only the rows for the current batch reach the GPU.
* Steps that exceed `VLLM_METAL_GGML_ATTN_BUDGET_MB` of attention temporaries,
  or mix many different query lengths, are split into several graphs.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_METAL_BACKEND` | `mlx` | `ggml` selects this backend |
| `VLLM_METAL_GGML_ATTN_BUDGET_MB` | `512` | per-graph budget for gathered K/V + masks (reserved from the KV cache) |
| `VLLM_METAL_GGML_PROFILE` | `0` | log per-step phase timings |
| `VLLM_METAL_GGML_VERBOSE` | `0` | forward ggml info logs |
| `VLLM_METAL_GGML_DEVICE` | `metal` | `cpu` for debugging only |
| `VLLM_METAL_GGML_ALLOW_CPU_FALLBACK` | `0` | allow ops Metal cannot run to use the ggml CPU backend |
| `GGML_BACKEND_DIR` | Homebrew `libexec` | where `libggml-metal.so` is loaded from |

!!! warning "CPU backend and PyTorch"
    Homebrew's ggml CPU backend links its own `libomp`. Running CPU kernels in
    a process that has initialized PyTorch's bundled `libomp` aborts, so the
    engine refuses CPU fallback unless explicitly allowed.

## Validation

`tools/ggml_parity.py` compares logits against Hugging Face transformers
(fp32) across single-shot prefill, chunked prefill + decode and mixed batches:

```bash
python tools/ggml_parity.py Qwen/Qwen3.5-0.8B
python tools/ggml_parity.py google/gemma-4-E2B-it --long 700
```

Both models pass with top-1 agreement ≥ 0.99 and mean KL ≤ 1.4e-3 (HF's own
bf16 inference of Gemma 4 E2B sits at mean KL 6.5e-3 against fp32).
