# Supported Models

vllm-metal currently focuses on text-only language models on Apple Silicon. Multi-modal (vision / audio input) models are not yet supported.

## Legend

| Symbol | Meaning |
| --- | --- |
| ✅ | Supported model/feature |
| 🔵 | Experimental supported model/feature |
| ❌ | Not supported model/feature |
| 🟡 | Not tested or verified |

## Text-Only Language Models

| Model | Support | Attention Kernel | Automatic Prefix Cache | PRs | Notes |
| --- | --- | --- | --- | --- | --- |
| Qwen3 | ✅ | GQA (paged) |  | [#232](https://github.com/vllm-project/vllm-metal/pull/232), [#237](https://github.com/vllm-project/vllm-metal/pull/237) |  |
| Qwen3.5 | ✅ | Hybrid SDPA + GDN linear |  | [#210](https://github.com/vllm-project/vllm-metal/pull/210), [#226](https://github.com/vllm-project/vllm-metal/pull/226), [#230](https://github.com/vllm-project/vllm-metal/pull/230), [#235](https://github.com/vllm-project/vllm-metal/pull/235), [#239](https://github.com/vllm-project/vllm-metal/pull/239), [#243](https://github.com/vllm-project/vllm-metal/pull/243), [#259](https://github.com/vllm-project/vllm-metal/pull/259), [#265](https://github.com/vllm-project/vllm-metal/pull/265), [#194](https://github.com/vllm-project/vllm-metal/issues/194) |  |
| Qwen3.6 | ✅ | Hybrid SDPA + GDN linear (MoE) |  |  |  |
| Qwen3-Next | ✅ | Hybrid SDPA + GDN linear |  | [#240](https://github.com/vllm-project/vllm-metal/pull/240) |  |
| Gemma 4 | 🔵 | GQA + per-layer sliding window + YOCO |  | [#251](https://github.com/vllm-project/vllm-metal/pull/251), [#260](https://github.com/vllm-project/vllm-metal/pull/260), [#269](https://github.com/vllm-project/vllm-metal/pull/269), [#275](https://github.com/vllm-project/vllm-metal/pull/275), [#277](https://github.com/vllm-project/vllm-metal/pull/277), [#278](https://github.com/vllm-project/vllm-metal/pull/278), [#282](https://github.com/vllm-project/vllm-metal/pull/282), [#276](https://github.com/vllm-project/vllm-metal/issues/276), [#279](https://github.com/vllm-project/vllm-metal/pull/279), [#281](https://github.com/vllm-project/vllm-metal/issues/281) |  |
| Gemma 3 | 🟡 | GQA (paged) |  |  |  |
| Llama 3 | 🟡 | GQA (paged) |  |  |  |
| Mistral-Small-24B | 🔵 | GQA (paged) |  | [#166](https://github.com/vllm-project/vllm-metal/pull/166), [#190](https://github.com/vllm-project/vllm-metal/pull/190) |  |
| GPT-OSS | 🔵 | Sink attention (paged) |  | [#190](https://github.com/vllm-project/vllm-metal/pull/190), [#221](https://github.com/vllm-project/vllm-metal/pull/221), [#212](https://github.com/vllm-project/vllm-metal/issues/212) |  |
| GLM-4.5 | 🟡 | MLA (paged latent cache, MLX SDPA — no Metal kernel) |  | [#213](https://github.com/vllm-project/vllm-metal/pull/213), [#233](https://github.com/vllm-project/vllm-metal/pull/233) |  |
| GLM-4.7-Flash | 🔵 | GQA (paged) |  | [#190](https://github.com/vllm-project/vllm-metal/pull/190) |  |
