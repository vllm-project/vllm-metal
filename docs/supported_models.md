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

`Automatic Prefix Cache` describes the default behavior when the user does
not pass `--enable-prefix-caching`. After
[#283](https://github.com/vllm-project/vllm-metal/pull/283), unified paged-KV
models on Metal can reuse shared prefixes by default. Upstream vLLM still
keeps the default off for hybrid/Mamba models, so those rows remain `❌`
unless prefix caching is explicitly forced. These values describe the
default engine behavior, not exhaustive model-by-model benchmarking on
Metal. Qwen3 is explicitly covered by the paged prefix-cache e2e test.

| Model | Support | Attention Kernel | Automatic Prefix Cache | PRs | Notes |
| --- | --- | --- | --- | --- | --- |
| Qwen3 | ✅ | GQA (paged) | ✅ | [#232](https://github.com/vllm-project/vllm-metal/pull/232), [#237](https://github.com/vllm-project/vllm-metal/pull/237), [#283](https://github.com/vllm-project/vllm-metal/pull/283) | Validated by the paged prefix-cache e2e test |
| Qwen3.5 | ✅ | Hybrid SDPA + GDN linear | ❌ | [#210](https://github.com/vllm-project/vllm-metal/pull/210), [#226](https://github.com/vllm-project/vllm-metal/pull/226), [#230](https://github.com/vllm-project/vllm-metal/pull/230), [#235](https://github.com/vllm-project/vllm-metal/pull/235), [#239](https://github.com/vllm-project/vllm-metal/pull/239), [#243](https://github.com/vllm-project/vllm-metal/pull/243), [#259](https://github.com/vllm-project/vllm-metal/pull/259), [#265](https://github.com/vllm-project/vllm-metal/pull/265), [#194](https://github.com/vllm-project/vllm-metal/issues/194) | Upstream keeps automatic prefix caching off for hybrid/Mamba models |
| Qwen3.6 | ✅ | Hybrid SDPA + GDN linear (MoE) | ❌ |  | Upstream keeps automatic prefix caching off for hybrid/Mamba models |
| Qwen3-Next | ✅ | Hybrid SDPA + GDN linear | ❌ | [#240](https://github.com/vllm-project/vllm-metal/pull/240) | Upstream keeps automatic prefix caching off for hybrid/Mamba models |
| Gemma 4 | 🔵 | GQA + per-layer sliding window + YOCO | ✅ | [#251](https://github.com/vllm-project/vllm-metal/pull/251), [#260](https://github.com/vllm-project/vllm-metal/pull/260), [#269](https://github.com/vllm-project/vllm-metal/pull/269), [#275](https://github.com/vllm-project/vllm-metal/pull/275), [#277](https://github.com/vllm-project/vllm-metal/pull/277), [#278](https://github.com/vllm-project/vllm-metal/pull/278), [#282](https://github.com/vllm-project/vllm-metal/pull/282), [#276](https://github.com/vllm-project/vllm-metal/issues/276), [#279](https://github.com/vllm-project/vllm-metal/pull/279), [#281](https://github.com/vllm-project/vllm-metal/issues/281), [#283](https://github.com/vllm-project/vllm-metal/pull/283) | Default-on for non-hybrid paged models; overall model support remains experimental |
| Gemma 3 | ✅ | GQA (paged) | ✅ | [#283](https://github.com/vllm-project/vllm-metal/pull/283) | tested on gemma-3-1b-it-qat-4bit; gemma-3-4b-it-4bit verified for text-only generation with VLM image inputs bypassed |
| Llama 3 | ✅ | GQA (paged) | ✅ | [#294](https://github.com/vllm-project/vllm-metal/pull/294) | tested on llama3.2-1B |
| Mistral-Small-24B | 🔵 | GQA (paged) | ✅ | [#166](https://github.com/vllm-project/vllm-metal/pull/166), [#190](https://github.com/vllm-project/vllm-metal/pull/190), [#283](https://github.com/vllm-project/vllm-metal/pull/283) | Default-on for non-hybrid paged models |
| GPT-OSS | 🔵 | Sink attention (paged) | ✅ | [#190](https://github.com/vllm-project/vllm-metal/pull/190), [#221](https://github.com/vllm-project/vllm-metal/pull/221), [#212](https://github.com/vllm-project/vllm-metal/issues/212), [#283](https://github.com/vllm-project/vllm-metal/pull/283) | Default-on for non-hybrid paged models |
| GLM-4.5 | 🟡 | MLA (paged latent cache, MLX SDPA — no Metal kernel) | 🟡 | [#213](https://github.com/vllm-project/vllm-metal/pull/213), [#233](https://github.com/vllm-project/vllm-metal/pull/233) | Automatic prefix caching is not yet verified on the MLX MLA path |
| GLM-4.7-Flash | 🔵 | GQA (paged) | ✅ | [#190](https://github.com/vllm-project/vllm-metal/pull/190), [#283](https://github.com/vllm-project/vllm-metal/pull/283) | Default-on for non-hybrid paged models |
