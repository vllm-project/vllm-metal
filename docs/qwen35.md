# Qwen3.5 Model Support

Qwen3.5 models (dense and MoE variants) are supported via the `[qwen35]` optional extra.

## Requirements

| Dependency | Minimum | Why |
|---|---|---|
| mlx-lm | 0.31.0 | Native `qwen3_5` / `qwen3_5_moe` model modules |
| mlx-vlm | 0.3.12 | Qwen3.5 VLM support |
| transformers | 5.0.0 | Qwen3.5 config compatibility |
| vllm | 0.17.0 | `Qwen3_5MoeForConditionalGeneration` model registry |

## Installation

```bash
# Step 1: install vllm 0.17.0 (required for Qwen3.5 model registry)
VLLM_VERSION=0.17.0 ./install.sh

# Step 2: install Qwen3.5 dependencies
pip install 'vllm-metal[qwen35]'
```

## Verified models

| Model | Type | Tested |
|---|---|---|
| Qwen3.5-35B-A3B | MoE, multimodal | Yes |

## Usage

```bash
vllm serve Qwen/Qwen3.5-35B-A3B --max-model-len 4096 --dtype auto
```

## Architecture notes

Qwen3.5 is a hybrid model with alternating `linear_attention` (Mamba/SSM)
and `full_attention` layers. The KV cache contains a mix of `ArraysCache`
(for linear attention) and `KVCache` (for full attention). As of mlx-lm
0.31.0, both sequential and batched decode work correctly with this layout.
