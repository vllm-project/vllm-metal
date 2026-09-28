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
| 🟡 | Verified only with a tiny model; real checkpoint unverified |

Each row tracks a model family. The **Example checkpoint** is one configuration
we have actually run on Metal — a starting point, not the only checkpoint that
works. For MLX, other sizes and quantizations of the same family generally work too;
per-machine details (chip, RAM, macOS, reference match) and the full change
history live in the project's PRs, not in this table. If a model or checkpoint
does not work, please open an issue rather than adding more rows or example
checkpoints.

The **PyTorch MPS** column covers the opt-in `VLLM_METAL_BACKEND=mps` path;
blank cells are unverified.

<!-- Keep this a high-level support matrix. Add a feature column only once at
least one shipped model uses it (e.g. speculative decoding, tensor parallel) —
do not add columns for unimplemented features. -->

## Text Pooling

Metal V1 has experimental text-only pooling support. See
[Text Pooling](text_embedding_pooling.md) for scope, usage, and
validation guidance. The reranker requires Qwen3 sequence-classification
`hf_overrides`.

| Model | MLX | PyTorch MPS | Runner | Example checkpoint |
| --- | --- | --- | --- | --- |
| Qwen3-Embedding | 🔵 | | `pooling` / `embed` (paged) | `mlx-community/Qwen3-Embedding-0.6B-8bit` |
| Qwen3-Reranker | 🔵 | | `pooling` / `classify` (paged) | `mku64/Qwen3-Reranker-0.6B-mlx-8Bit` |
| BGE-M3 | 🔵 | | `pooling` / `embed`, `token_classify` (encoder) | `BAAI/bge-m3` |
| Multilingual E5 Base | 🔵 | | `pooling` / `embed` (encoder) | `intfloat/multilingual-e5-base` |

## Multimodal Language Models

Native multimodal support currently targets image-only vision-language requests on the paged backend.

| Model | MLX | PyTorch MPS | Runner | Scope | Example checkpoint |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL | 🔵 | | native multimodal paged generation | image input, no video | `mlx-community/Qwen3-VL-4B-Instruct-4bit` |
| Qwen3.5 (dense) | 🔵 | | native multimodal paged generation | image input, no video; FP8 checkpoints stay text-only | `mlx-community/Qwen3.5-4B-MLX-4bit` |
| PaddleOCR-VL | 🔵 | | native multimodal paged generation | image input, no video | `PaddlePaddle/PaddleOCR-VL-1.6` |
| Gemma 4 | 🔵 | | mlx_lm text backbone + mlx-vlm vision sidecar, paged generation | image input, no video/audio, bidirectional attention inside image blocks on sliding-window layers | `mlx-community/unsloth-gemma-4-26B-A4B-it-qat-oQ4` |

Gemma 4 keeps its text path exactly as on the text-only table (mlx_lm model,
selective logits, intermediate forward); only `vision_tower` and `embed_vision`
are loaded from the checkpoint through mlx-vlm. The sidecar activates in
`VLLM_METAL_MULTIMODAL_MODE=auto` when the checkpoint resolves to a local
safetensors directory with `vision_tower.*` weights, the HF `Gemma4Processor`
builds (the checkpoint's `processor_config.json` needs a `video_processor`
block with transformers 5.14+), the text config has no per-layer inputs, and
no speculative decoding is configured; otherwise the model stays text-only and
the reason is logged. A repo id such as the example above only resolves when
it is already fully cached locally (`hf download <repo>` first) — the sidecar
never triggers a download itself, and an uncached repo id falls back to
text-only with a logged reason exactly like a nonexistent local path. Not
every conversion meets these conditions: `mlx-community/gemma-4-12B-it-4bit`
ships no vision weights and the `gemma-4-e2b-it-4bit` / `gemma-4-e4b-it-4bit`
conversions have per-layer inputs, so all three serve text-only.

Image soft tokens attend bidirectionally to each other inside their own image
block on sliding-window layers, matching HF's
`create_masks_for_vision_model` semantics; full-attention layers and text
tokens stay causal. By default the tiled Metal prefill kernel applies this
mask itself from a per-row range buffer (vLLM's `mm_prefix` contract) and the
engine logs `Metal: mm_prefix ranges on R row(s)` the first time a prefill
batch carries image-block rows; `VLLM_METAL_MM_PREFIX_PATH=recompute` selects
the reference path instead, which recomputes the block rows with MLX SDPA
after the kernel and logs `Metal: bidirectional image attention: N
segment(s), M block(s), R row(s)`. Both paths give the same mask; the kernel
path attends each row once. A native build that predates the kernel's
`mm_prefix` support takes the recompute path instead and says so once at
startup (`the compiled ops predate mm_prefix support`); a float32 KV cache
keeps the recompute too, since the tiled kernel has no float32 instantiation.
At startup the engine logs which path image blocks take, for example
`Metal: image blocks attend through the tiled prefill kernel
(VLLM_METAL_MM_PREFIX_PATH=kernel, bfloat16 KV cache)`. An image block that
does not fit inside one prefill
step falls back to causal attention for the rest of the request, with a
warning containing `falling back to causal attention`; raise
`--max-num-batched-tokens` or lower `--max-num-seqs` to keep the block inside
one step instead. `--max-num-batched-tokens` must be at least the image
soft-token count plus two (282 by default, for the boi/eoi tokens) so a block
fits one prefill step at all. TurboQuant KV cache compression is refused at
load time in sidecar mode because neither image-attention path supports it: the
tiled kernel has no TurboQuant variant, and the recompute reads K/V back from
the paged cache unquantized.

## Text-Only Language Models

Prefix caching is enabled by default where supported, as shown in the table
below. It is currently disabled for Nemotron-H and Granite 4.0 hybrid models.

HF AWQ checkpoints load through mlx-lm's `_transform_awq_weights` repack, with an
entry-point preflight that normalizes AutoAWQ aliases (`w_bit`, `q_group_size`,
uppercase `"GEMM"`) and rejects unsupported variants (`gemv`, `bits != 4`,
`group_size != 128`, `zero_point=false`) before model state is built. Verified
for Qwen2.5, Llama 3, and Mistral
([#340](https://github.com/vllm-project/vllm-metal/pull/340),
[#381](https://github.com/vllm-project/vllm-metal/pull/381)).

GGUF checkpoints serve by detection like AWQ, with no env flag:
vllm-metal's GGUF engine integration sets `quantization=gguf` from the file
(vLLM 0.24 moved its in-tree GGUF support to the CUDA/ROCm-only
[vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin)). Dense
`qwen2`/`qwen3`/`llama`/`mistral` checkpoints load from a local `.gguf`
(including llama.cpp's Q4_K_M and Q5_K_M exports) or a remote `repo_id:quant`
reference and stay quantized in memory. See [GGUF](gguf.md) for the supported
qtypes, exclusions, and serve examples. Verified end-to-end on Qwen3-0.6B,
Llama-3.2-1B-Instruct, and Mistral-7B-Instruct-v0.3 in Q8_0 and Q4_K_M
([#415](https://github.com/vllm-project/vllm-metal/issues/415),
[#761](https://github.com/vllm-project/vllm-metal/issues/761)).

Ling-3.0 Tiny supports the official BF16 checkpoint and MLX-native MXFP8
checkpoints converted from it. Direct loading of the official serialized
block-FP8 checkpoint is not supported.

| Model | MLX | PyTorch MPS | Attention Kernel | Automatic Prefix Cache | Example checkpoint |
| --- | --- | --- | --- | --- | --- |
| Qwen3 | ✅ | 🔵 | GQA (paged) | ✅ | `Qwen/Qwen3-0.6B` |
| Qwen3.5 / 3.6 / 3.8 | ✅ | | Hybrid SDPA + GDN linear (3.6 adds MoE) | 🔵 | `mlx-community/Qwen3.8-27B-8bit` |
| Qwen3-Next | ✅ | | Hybrid SDPA + GDN linear | 🔵 | `mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit` |
| LFM2 / LFM2.5 | ✅ | | Hybrid SDPA + ShortConv | ✅ | `LiquidAI/LFM2.5-1.2B-Instruct` |
| Nemotron-H (Nemotron 3.5 Lightning) | 🔵 | | Hybrid SDPA + Mamba-2 (MoE) | ❌ | `mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit` |
| Granite 4.0-h-micro | 🔵 | | Hybrid SDPA + Mamba-2 | ❌ | `mlx-community/granite-4.0-h-micro-4bit` |
| Gemma 4 | ✅ | | GQA + per-layer sliding window + YOCO | ✅ | `mlx-community/gemma-4-e2b-it-4bit` |
| Gemma 3 | ✅ | | GQA + per-layer sliding window (paged) | ✅ | `mlx-community/gemma-3-1b-it-qat-4bit` |
| Llama 3 | ✅ | | GQA (paged) | ✅ | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` |
| Mistral-7B | ✅ | | GQA (paged) | ✅ | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` |
| Mistral-Small-24B | 🔵 | | GQA (paged) | ✅ | `mlx-community/Mistral-Small-24B-Instruct-2501-4bit` |
| StableLM 2 | ✅ | | MHA + partial RoPE (paged) | ✅ | `mlx-community/stablelm-2-zephyr-1_6b-4bit` |
| Phi 1.5 / Phi 2 | ✅ | | MHA + partial RoPE (paged) | ✅ | `mlx-community/phi-2-hf-4bit-mlx` |
| GPT-OSS | 🔵 | | Sink attention (paged) | ✅ | `openai/gpt-oss-20b` |
| Ling-3.0 Tiny | 🔵 | | Hybrid MLA + KDA (paged latent + recurrent state) | 🔵 | `inclusionAI/Ling-3.0-tiny` (BF16 or converted MXFP8) |
| GLM-4.5 | Not verified | | MLA (paged latent cache, MLX SDPA — no Metal kernel) | Not verified | — |
| MiniCPM3-4B | ✅ | | MLA (paged latent cache) | ✅ | `mlx-community/MiniCPM3-4B-4bit` |
| GLM-4.7-Flash | 🔵 | | GQA (paged) | ✅ | `mlx-community/GLM-4.7-Flash-4bit` |
| DeepSeek-R1-Distill-Qwen | ✅ | | GQA (paged) | ✅ | `mlx-community/DeepSeek-R1-Distill-Qwen-7B-3bit` |
| Phi-4-mini | ✅ | | GQA packed qkv (paged) | ✅ | `microsoft/Phi-4-mini-instruct` |
| Phi-3.5-mini | ✅ | | MHA packed qkv (paged) | ✅ | `mlx-community/Phi-3.5-mini-instruct-4bit` |
| Qwen2.5 | ✅ | | GQA (paged) | ✅ | `mlx-community/Qwen2.5-7B-Instruct-4bit` |
| Qwen2-7B | ✅ | | GQA (paged) | ✅ | `mlx-community/Qwen2-7B-Instruct-4bit` |
| Yi-1.5-9B | ✅ | | GQA (paged, LlamaForCausalLM) | ✅ | `mlx-community/Yi-1.5-9B-Chat-4bit` |
| SmolLM3-3B | ✅ | | GQA (paged) | ✅ | `mlx-community/SmolLM3-3B-4bit` |
| Granite 3.3 | 🔵 | | GQA (paged) | ✅ | `mlx-community/granite-3.3-8b-instruct-4bit` |
| EXAONE 4.0 | 🔵 | | GQA (paged) | ✅ | `mlx-community/exaone-4.0-1.2b-4bit` |
| Laguna | ✅  | | GQA (paged) | ✅ | `poolside/Laguna-XS-2.1-NVFP4-mlx` |
| Hunyuan (dense) | ✅ | 🔵 | GQA + QK norm (paged) | ✅ | `mlx-community/Hunyuan-1.8B-Instruct-4bit` |
| MiniMax M2 | 🟡 | | GQA + full-projection QK norm (paged) | Not verified | — |
| OLMo 2 | ✅ | 🔵 | MHA + full-projection QK norm (paged) | ✅ | `allenai/OLMo-2-0425-1B-Instruct` |
| OLMoE | ✅ | | MHA + full-projection QK norm (paged) | ✅ | `mlx-community/OLMoE-1B-7B-0125-Instruct-4bit` |
| OLMo 3 | 🔵 | | MHA + per-layer sliding window (paged) | ✅ | `mlx-community/Olmo-3-7B-Instruct-4bit` |

sliding-window attention (SWA) is not fully optimized.
