# GGUF

vllm-metal supports dense decoder GGUF checkpoints through the MLX runtime. A
GGUF file carries weights only, so it needs a Hugging Face config and tokenizer
source.

## Installation

GGUF support needs an optional dependency. Install it inside the environment
created by [Installation](installation.md):

```bash
source ~/.venv-vllm-metal/bin/activate
pip install 'vllm-metal[gguf]'
```

Without it, the engine fails at model load with `ModuleNotFoundError: No module
named 'gguf'`.

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
only as shards, fail before model load.

With `HF_HUB_OFFLINE=1`, remote references select from the cached repo listing
of the requested revision, or from its cached files when the cache has no
listing. Cache the matching GGUF file and companion config/tokenizer first;
missing, ambiguous, or shard-only matches also fail before model load.

## Current scope

- Supported model families: Qwen2, Qwen3, Llama, and Mistral.
- Supported qtypes: Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, and
  Q6_K, plus plain F32/F16/BF16 tensors, mixed freely within one file, so
  llama.cpp's Q4_K_M and Q5_K_M exports and bartowski's Q4_K_L load. Remote
  reference tags accept Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K and Q2_K_S/M/L,
  Q3_K and Q3_K_S/M/L, Q4_K_S/M/L, Q5_K_S/M/L, Q6_K, Q6_K_L, and the plain
  types.
- llama.cpp falls back from Q4_K/Q5_K to Q5_0/Q5_1 on rows that are not a
  multiple of 256 wide, as in Qwen2.5-0.5B's Q4_K_M and Q5_K_M files. Its
  Q2_K/Q3_K exports write IQ4_NL on those rows instead, which stays
  unsupported: 121 of the 291 tensors in Qwen2.5-0.5B-Instruct-GGUF's
  `q2_k.gguf` are IQ4_NL, the embedding among them, so that file is
  rejected while files without IQ4_NL rows load.
- Q2_K/Q3_K/Q6_K weights stay in their GGUF blocks and run on custom Metal
  kernels. Small batches use a fused kernel on the packed blocks, and larger
  ones, such as a prefill, dequantize a transient dense copy of the weight
  for one GEMM and free it afterwards.
- A tied model's unused `output.weight` is skipped whatever its qtype.
- Unsupported: IQ and T-quants, MoE, SSM or hybrid models, vision
  models, fused-QKV GGUFs, sharded GGUF files, and LoRA (`--enable-lora`).
