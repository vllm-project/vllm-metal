# vLLM Metal Plugin

> **High-performance LLM inference on Apple Silicon using MLX and vLLM**

vLLM Metal is a plugin that enables vLLM to run on Apple Silicon Macs using MLX as the primary compute backend. It unifies MLX and PyTorch under a single lowering path.

**Documentation**: https://docs.vllm.ai/projects/vllm-metal/en/latest/

---
*Latest News* 🔥

- [2026/04] We released the new version v0.2.0! Unified paged varlen Metal kernel is now the default attention backend. 83x TTFT, 3.6x throughput compared to v0.1.0.

---

## Requirements

- macOS on Apple Silicon
- Native arm64 Python 3.12. Rosetta/x86_64 Python is not supported.
- Xcode Command Line Tools (`xcode-select --install`). vLLM core is compiled from source via `clang++`. The Metal kernels ship **prebuilt**, so no Metal compiler or toolchain is needed to run them.

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
