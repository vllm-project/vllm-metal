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
repository. Missing, ambiguous, or sharded matches fail before model load.

With `HF_HUB_OFFLINE=1`, remote references select from the requested revision
in the local Hugging Face cache. Cache the matching GGUF file and companion
config/tokenizer first; missing, ambiguous, or sharded cached matches also fail
before model load.

## Current scope

- Supported model families: Qwen2, Qwen3, Llama, and Mistral.
- Supported qtypes: Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, and Q6_K, plus plain
  F32/F16/BF16 tensors, mixed freely within one file, so llama.cpp's Q4_K_M
  and Q5_K_M exports and bartowski's Q4_K_L load. Remote reference tags accept
  Q8_0, Q4_0, Q4_1, and the plain types; K-quant files load from a local path.
- Q6_K weights stay in their GGUF blocks and run on custom Metal kernels.
  Small batches use a fused kernel on the packed blocks, and larger ones, such
  as a prefill, dequantize a transient dense copy of the weight for one GEMM
  and free it afterwards.
- A tied model's unused `output.weight` is skipped whatever its qtype.
- Unsupported: Q5_0/Q5_1 (llama.cpp falls back to them when a row width is not
  a multiple of 256, as in Qwen2.5-0.5B's Q4_K_M), Q2_K/Q3_K and IQ quants,
  MoE, SSM or hybrid models, vision models, fused-QKV GGUFs, and sharded GGUF
  files.
