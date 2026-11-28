# GGUF

vllm-metal supports dense decoder (including Gemma 3 and Gemma 4), Qwen3 MoE,
and Qwen3.5 GGUF checkpoints through the MLX runtime. A GGUF file carries
weights only, so it needs a Hugging Face config and tokenizer source. That
source can be a quantized checkpoint, such as an mlx-community 4-bit, FP8, or
AWQ repository; vllm-metal reads only its config and tokenizer.

## Local weights

Use a local `.gguf` file and point `--tokenizer` at the matching config and
tokenizer source:

```bash
vllm serve /path/to/model.gguf \
  --tokenizer Qwen/Qwen3-0.6B
```

If `config.json` is next to the `.gguf`, `--tokenizer` is optional.

## Remote weights

Remote references use the same `repo_id:quant` shape as vLLM's GGUF plugin:

```bash
vllm serve bartowski/Qwen_Qwen3-0.6B-GGUF:Q8_0 \
  --tokenizer Qwen/Qwen3-0.6B
```

The source priority is:

```text
--hf-config-path > --tokenizer > GGUF weights repository
```

vllm-metal downloads exactly one matching `.gguf` file from the remote
repository. A quant published both as one file and as that file's shards
resolves to the single file. Missing or ambiguous matches, and quants published
only as shards, fail before model load. A config from an unsupported model
family fails earlier, before the `.gguf` file is selected or downloaded.

With `HF_HUB_OFFLINE=1`, remote references select from the cached repo listing
of the requested revision, or from its cached files when the cache has no
listing. Cache the matching GGUF file and companion config/tokenizer first;
missing, ambiguous, or shard-only matches also fail before model load.

## Current scope

- Supported model families: Qwen2, Qwen3, Qwen3 MoE (`qwen3moe`), Llama,
  Mistral, Qwen3.5 (`qwen35`), Gemma 3 (`gemma3`), and Gemma 4 dense
  (`gemma4`: E2B, E4B, 31B).
- Gemma 3 and Gemma 4 dense serve text only (`mmproj` files are not loaded).
  Point `--tokenizer` at the matching text or conditional-generation
  repository, such as `google/gemma-3-4b-it` or `google/gemma-4-E2B-it`.
  Gemma 4 MoE (`26B-A4B`) is not supported.
- Qwen3.5 serves text only and images are refused. Point `--tokenizer` at the
  `Qwen3_5ForConditionalGeneration` repository, such as `Qwen/Qwen3.5-9B`.
  vllm-metal does not load `mmproj` files. Files that embed a
  multi-token-prediction block, such as unsloth's `-MTP-GGUF` repos, load
  with that block skipped. MTP speculative decoding is not supported.
- Supported qtypes: Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K,
  Q6_K, IQ4_NL, IQ4_XS, IQ3_S, IQ3_XXS, and IQ2_S, plus plain F32/F16/BF16
  tensors, mixed freely within one file, so llama.cpp's Q4_K_M, Q5_K_M, IQ3_M,
  IQ3_XS and IQ3_XXS exports, bartowski's Q4_K_L, and unsloth's UD-Q4_K_XL
  load. Dense IQ2_M exports such as Qwen3-1.7B's load; Qwen3-30B-A3B's IQ2_M
  also carries IQ2_XS tensors, so it does not. Remote reference tags
  accept Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K and Q2_K_S/M/L, Q3_K and
  Q3_K_S/M/L, Q4_K_S/M/L, Q5_K_S/M/L, Q6_K, Q6_K_L, IQ4_NL, IQ4_XS, IQ3_S,
  IQ3_M, and the plain types.
- IQ4_NL and IQ4_XS weights need about twice their file size in memory, so
  a mostly-IQ4 file uses more than the same model's Q4_K_M. IQ3_S weights
  also need more memory than their file size, so an IQ3_M file uses about as
  much as the same model's Q4_K_M.
- Q2_K/Q3_K/Q6_K/IQ3_XXS/IQ2_S weights stay in their GGUF blocks and run on
  custom Metal kernels. Small batches use a fused kernel on the packed blocks,
  and larger ones, such as a prefill, dequantize a transient dense copy of the
  weight for one GEMM and free it afterwards.
- When a tied model's file also carries `output.weight`, logits use it, as
  llama.cpp does: llama-quantize stores that table at the output qtype (for
  example Q6_K) even when `token_embd` uses a lower one. An `output.weight` in
  a qtype the loader cannot run is skipped and the head stays tied.
- Unsupported: IQ1, IQ2_XXS, IQ2_XS and T-quants, MoE models other than Qwen3
  MoE, SSM or hybrid models other than Qwen3.5, vision input, sharded GGUF
  files, and LoRA (`--enable-lora`).
