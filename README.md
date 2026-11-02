<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/vllm-metal-dark.svg">
    <img src="docs/assets/vllm-metal-light.svg" alt="vLLM Metal — Run vLLM on your Apple Silicon GPU" width="900">
  </picture>
</p>

<p align="center">
| <a href="https://docs.vllm.ai/projects/vllm-metal/en/latest/"><b>Documentation</b></a> | <a href="https://docs.vllm.ai/projects/vllm-metal/en/latest/recipes/"><b>Recipes by Mac memory</b></a> | <a href="https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0"><b>Blog</b></a> | <a href="https://slack.vllm.ai/"><b>Developer Slack</b></a> (#hw-metal) |
</p>

vLLM Metal is a plugin that enables vLLM to run on Apple Silicon Macs using MLX as the primary compute backend. It unifies MLX and PyTorch under a single lowering path.

---
*Latest News* 🔥

- [2026/10] Experimental [Recursive Flint INT4](https://huggingface.co/RecursiveSI/flint-int4) support is now available on the MLX backend, including concurrent serving.
- [2026/10] DiffusionGemma now supports Jev-like System 1 decisions on Metal, with median decision latency under 0.2 s on an M5 Pro. See the [structured decisions guide](https://docs.vllm.ai/projects/vllm-metal/en/latest/structured_decisions/).
- [2026/10] Desktop inference competes with your other applications for memory. Sleep mode lets an idle model server return memory to those apps without shutting down. Read [Abhigith Neil Abraham's walkthrough](https://abneil.dev/blog/sleep-mode-on-a-mac-with-kv-offloading.html).
- [2026/10] Run local models with vLLM Metal through [llmman](https://github.com/llmmanorg/llmman). See the [setup recipe](https://docs.vllm.ai/projects/vllm-metal/en/latest/recipes/#or-with-llmman).
- [2026/09] [SiliconBench](https://arxiv.org/abs/2609.19169), which evaluates vLLM Metal and other serving engines on speed, memory use, and output fidelity, was accepted at [NeurIPS 2026's Evaluations and Datasets Track](https://neurips.cc/virtual/2026/poster/139658).
- [2026/09] Read our [official vLLM announcement](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0), covering concurrent serving on Apple Silicon and v0.28.0.
- [2026/08] vLLM Metal now uses M5 NAX tensor units to accelerate MHA, GQA, and MQA prefill.
- [2026/08] Qwen3.8 now runs on Metal! `mlx-community/Qwen3.8-27B-8bit` serves a 27B hybrid SDPA + GDN linear model on a single Apple Silicon Mac.
- [2026/04] We released the new version v0.2.0! Unified paged varlen Metal kernel is now the default attention backend. 83x TTFT, 3.6x throughput compared to v0.1.0.

---

## Architecture

Upstream vLLM supplies the API server, scheduler, and paged block manager; mlx_lm supplies the token-wise model layers; vllm-metal owns the request-aware attention path — the paged varlen kernel, M5 NAX prefill, and speculative decoding.

![vllm-metal in the vLLM stack](https://raw.githubusercontent.com/vllm-project/vllm-metal/main/docs/assets/architecture.svg)

## Requirements

- macOS 15 (Sequoia) or later, on Apple Silicon

## Supported Models

vllm-metal supports a growing set of models on Apple Silicon. See the full matrix in [docs/supported_models.md](docs/supported_models.md).

## Installation

- **Stable release:** Install with Homebrew, then run `vllm` without activating an environment.

  ```bash
  brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
  brew install vllm-project/vllm-metal/vllm-metal
  ```

- **Latest development build:** Use the curl installer, then activate its environment. No compiler needed.

  ```bash
  curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
  source ~/.venv-vllm-metal/bin/activate
  ```

- **Contributing Python or kernel code:** Follow the [source setup](docs/CONTRIBUTING.md#development-setup) for an editable checkout and native build tools.
