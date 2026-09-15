# vLLM Metal Plugin

> **High-performance LLM inference on Apple Silicon using MLX and vLLM**

vLLM Metal is a plugin that enables vLLM to run on Apple Silicon Macs using MLX as the primary compute backend. It unifies MLX and PyTorch under a single lowering path.

**Documentation**: https://docs.vllm.ai/projects/vllm-metal/en/latest/

---
*Latest News* 🔥

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
