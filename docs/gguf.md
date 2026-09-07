# GGUF

vllm-metal supports dense decoder GGUF checkpoints through the MLX runtime. A
GGUF file carries weights only, so it needs a Hugging Face config and tokenizer
source.

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
- Supported qtypes: Q8_0, Q4_0, and Q4_1.
- Unsupported: K-quants, MoE, SSM or hybrid models, vision models, fused-QKV
  GGUFs, and sharded GGUF files.
