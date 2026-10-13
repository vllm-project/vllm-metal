# MoE Expert Offloading

Mixture-of-experts checkpoints are mostly expert weights, but each token runs
only `top_k` of them. With expert offloading, the experts stay in the
checkpoint on disk and each MoE layer reads the experts its tokens route to,
through a cache in unified memory. This serves MoE models that do not fit in
memory when fully loaded.

The compute path is unchanged. The experts a step touches are stacked and the
layer runs mlx-lm's own `SwitchGLU` forward on them, so outputs are identical
to the fully resident model.

## Quick Start

```bash
vllm serve mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --additional-config '{"moe_expert_offload": {"cache_gb": 1.5}}'
```

| Key | Description |
|-----|-------------|
| `moe_expert_offload.cache_gb` | Memory for cached experts, in GiB. Required to enable offloading. |

The KV cache budget reserves `cache_gb` before sizing, so the resident model,
the expert cache and the KV cache fit `--gpu-memory-utilization` together.

## Choosing the cache size

The cache holds the most recently used experts across all layers. With a
single request, each decode step reads `top_k` experts per layer; a cache that
holds a few steps' worth keeps most reads warm. Prefill and batched decode
route to many more experts, so each step reads more from disk. Offloading
targets one to a few concurrent requests on Macs that cannot hold the model.

Expect lower decode throughput than a resident model, because every MoE layer
waits for its routing before it reads experts, and misses read from the SSD.

## Supported checkpoints

- mlx-lm text models whose experts are mlx-lm `SwitchGLU` layers, in two
  safetensors layouts: experts stacked under `<block>.switch_mlp` (mlx-community
  conversions such as Qwen3-30B-A3B) and one tensor per expert under
  `<block>.experts.<j>` (OLMoE and Qwen1.5-MoE conversions).
- Shared experts, attention and embeddings stay resident.

Not supported: GGUF, AWQ and multimodal checkpoints, pipeline parallelism,
LoRA, speculative decoding, and checkpoints that name expert projections
differently (Mixtral's `w1`/`w2`/`w3`).
