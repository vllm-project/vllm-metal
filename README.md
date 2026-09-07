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

An opt-in [PyTorch/MPS compatibility backend](docs/pytorch_backend.md) executes
upstream vLLM model implementations directly. It is experimental and currently
targets unquantized dense text generation on one Mac.

![vllm-metal in the vLLM stack](https://raw.githubusercontent.com/vllm-project/vllm-metal/main/docs/assets/architecture.svg)

## Requirements

- macOS 15 (Sequoia) or later, on Apple Silicon
- Native arm64 Python 3.12. Rosetta/x86_64 Python is not supported.

## Supported Models

vllm-metal supports a growing set of models on Apple Silicon. See the full matrix in [docs/supported_models.md](docs/supported_models.md).

## Installation

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

Using the install script above, the following will be installed under the `~/.venv-vllm-metal` directory (the default).
- vllm-metal plugin
- vllm core
- Related libraries

If you run `source ~/.venv-vllm-metal/bin/activate`, the `vllm` CLI becomes available and you can access the vLLM right away.

For how to use the `vllm` CLI, please refer to the official vLLM guide.
https://docs.vllm.ai/en/latest/cli/
