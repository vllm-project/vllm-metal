# Supported Models

vllm-metal supports text language models and a small set of native multimodal
models on Apple Silicon. Multimodal support is currently vision-only and runs on
the paged backend.

## Legend

| Symbol | Meaning |
| --- | --- |
| ✅ | Supported model/feature |
| 🔵 | Experimental supported model/feature |
| ❌ | Not supported model/feature |
| 🟡 | Not tested or verified |

Each row tracks a model family. The **Example checkpoint** is one configuration
we have actually run on Metal — a starting point, not the only checkpoint that
works. Other sizes and quantizations of the same family generally work too;
per-machine details (chip, RAM, macOS, reference match) and the full change
history live in the project's PRs, not in this table. If a model or checkpoint
does not work, please open an issue rather than adding more rows or example
checkpoints.

<!-- Keep this a high-level support matrix. Add a feature column only once at
least one shipped model uses it (e.g. speculative decoding, tensor parallel) —
do not add columns for unimplemented features. -->

## Text Pooling

Metal V1 has experimental text-only pooling support. See
[Text Pooling](text_embedding_pooling.md) for scope, usage, and
validation guidance. The reranker requires Qwen3 sequence-classification
`hf_overrides`.

| Model | Support | Runner | Example checkpoint |
| --- | --- | --- | --- |
| Qwen3-Embedding | 🔵 | `pooling` / `embed` (paged) | `mlx-community/Qwen3-Embedding-0.6B-8bit` |
| Qwen3-Reranker | 🔵 | `pooling` / `classify` (paged) | `mku64/Qwen3-Reranker-0.6B-mlx-8Bit` |

## Multimodal Language Models

Native multimodal support currently targets image-only vision-language requests on the paged backend.

| Model | Support | Runner | Scope | Example checkpoint |
| --- | --- | --- | --- | --- |
| Qwen3-VL | 🔵 | native multimodal paged generation | image input, no video | `mlx-community/Qwen3-VL-4B-Instruct-4bit` |
| PaddleOCR-VL | 🔵 | native multimodal paged generation | image input, no video | `PaddlePaddle/PaddleOCR-VL-1.6` |

## Text-Only Language Models

`Automatic Prefix Cache` is the default behavior when you do not pass
`--enable-prefix-caching`. Since
[#283](https://github.com/vllm-project/vllm-metal/pull/283), unified paged-KV
models reuse shared prefixes by default. Upstream vLLM keeps it off for
hybrid/Mamba models, so those rows stay `❌`. These values describe default
engine behavior, not exhaustive per-model benchmarking on Metal.

HF AWQ checkpoints load through mlx-lm's `_transform_awq_weights` repack, with an
entry-point preflight that normalizes AutoAWQ aliases (`w_bit`, `q_group_size`,
uppercase `"GEMM"`) and rejects unsupported variants (`gemv`, `bits != 4`,
`group_size != 128`, `zero_point=false`) before model state is built. Verified
for Qwen2.5, Llama 3, and Mistral
([#340](https://github.com/vllm-project/vllm-metal/pull/340),
[#381](https://github.com/vllm-project/vllm-metal/pull/381)).

Local GGUF checkpoints serve by detection like AWQ, with no env flag (vLLM sets
`quantization=gguf` from the file). A `.gguf` carries weights only, so it pairs
with a companion config dir (`--tokenizer`) and needs the `gguf` extra; the
weights stay MLX-native quantized (Q8_0/Q4_0, not a dense fallback). Scope is dense
`qwen2`/`qwen3` with per-tensor `Q8_0`/`Q4_0`; K-quants, `Q4_1`, fused-QKV, MoE,
SSM/hybrid, vision, and remote `repo:quant` are rejected with a clear error.
Verified end-to-end on Qwen3-0.6B Q8_0
([#415](https://github.com/vllm-project/vllm-metal/issues/415)).

| Model | Support | Attention Kernel | Automatic Prefix Cache | Example checkpoint |
| --- | --- | --- | --- | --- |
| Qwen3 | ✅ | GQA (paged) | ✅ | `Qwen/Qwen3-0.6B` |
| Qwen3.5 / 3.6 | ✅ | Hybrid SDPA + GDN linear (3.6 adds MoE) | ❌ | `Qwen/Qwen3.5-0.8B` |
| Qwen3-Next | ✅ | Hybrid SDPA + GDN linear | ❌ | `mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit` |
| Gemma 4 | 🔵 | GQA + per-layer sliding window + YOCO | ✅ | `mlx-community/gemma-4-E2B-it` |
| Gemma 3 | ✅ | GQA (paged) | ✅ | `mlx-community/gemma-3-1b-it-qat-4bit` |
| Llama 3 | ✅ | GQA (paged) | ✅ | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` |
| Mistral-7B | ✅ | GQA (paged) | ✅ | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` |
| Mistral-Small-24B | 🔵 | GQA (paged) | ✅ | `mlx-community/Mistral-Small-24B-Instruct-2501-4bit` |
| GPT-OSS | 🔵 | Sink attention (paged) | ✅ | `openai/gpt-oss-20b` |
| GLM-4.5 | 🟡 | MLA (paged latent cache, MLX SDPA — no Metal kernel) | 🟡 | — |
| MiniCPM3-4B | ✅ | MLA (paged latent cache) | ✅ | `mlx-community/MiniCPM3-4B-4bit` |
| GLM-4.7-Flash | 🔵 | GQA (paged) | ✅ | `mlx-community/GLM-4.7-Flash-4bit` |
| DeepSeek-R1-Distill-Qwen | ✅ | GQA (paged) | ✅ | `mlx-community/DeepSeek-R1-Distill-Qwen-7B-3bit` |
| Phi-4-mini | ✅ | GQA packed qkv (paged) | ✅ | `microsoft/Phi-4-mini-instruct` |
| Phi-3.5-mini | ✅ | MHA packed qkv (paged) | ✅ | `mlx-community/Phi-3.5-mini-instruct-4bit` |
| Qwen2.5 | ✅ | GQA (paged) | ✅ | `mlx-community/Qwen2.5-7B-Instruct-4bit` |
| Qwen2-7B | ✅ | GQA (paged) | ✅ | `mlx-community/Qwen2-7B-Instruct-4bit` |
| Yi-1.5-9B | ✅ | GQA (paged, LlamaForCausalLM) | ✅ | `mlx-community/Yi-1.5-9B-Chat-4bit` |
| SmolLM3-3B | ✅ | GQA (paged) | ✅ | `mlx-community/SmolLM3-3B-4bit` |
| Granite 3.3 | 🔵 | GQA (paged) | ✅ | `mlx-community/granite-3.3-8b-instruct-4bit` |
