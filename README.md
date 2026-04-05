# vLLM Metal Plugin

> **High-performance LLM inference on Apple Silicon using MLX and vLLM**

vLLM Metal is a plugin that enables vLLM to run on Apple Silicon Macs using MLX as the primary compute backend. It unifies MLX and PyTorch under a single lowering path.

## Features

- **MLX-accelerated inference**: faster than PyTorch MPS on Apple Silicon
- **Unified memory**: True zero-copy operations leveraging Apple Silicon's unified memory architecture
- **vLLM compatibility**: Full integration with vLLM's engine, scheduler, and OpenAI-compatible API
- **Paged attention** *(experimental)*: Efficient KV cache management for long sequences — opt-in via `VLLM_METAL_USE_PAGED_ATTENTION=1`; default path uses MLX-managed KV cache. When enabled, expect significantly better serving performance (~82x TTFT, ~3.75x throughput in early benchmarks on Qwen3-0.6B). Other models may have rough edges.
- **GQA support**: Grouped-Query Attention for efficient inference

## Supported Models

Text and vision-language models tested on vllm-metal:

| Model family | Example | Backend |
|---|---|---|
| Llama 3 / 3.1 / 3.2 | `meta-llama/Llama-3.2-3B-Instruct` | mlx-lm |
| Qwen3 | `Qwen/Qwen3-0.6B` | mlx-lm |
| Qwen3.5 (hybrid) | `Qwen/Qwen3.5-7B-Instruct` | mlx-lm |
| Qwen3-VL | `Qwen/Qwen3-VL-7B-Instruct` | mlx-vlm |
| Gemma 4 | `google/gemma-4-E4B-it`, `google/gemma-4-31B-it` | mlx-vlm |
| LLaVA | `llava-hf/llava-1.5-7b-hf` | mlx-vlm |
| Whisper (STT) | `openai/whisper-large-v3` | mlx-whisper |

> **Note**: Gemma 4 requires `mlx-lm >= 0.31.2` and `mlx-vlm` with Gemma 4 support (not yet on PyPI as of 2026-04-05; see [Installation](#installation) for the local-checkout workaround).

## Limitations

- **Gemma 4 — text-only inference**: Gemma 4 models (`google/gemma-4-*`) are routed through `mlx-vlm` for all modalities. Vision (image) and audio inputs are supported by mlx-vlm's Gemma 4 implementation. To force text-only inference via `mlx-lm`, add `"gemma4"` to `_MLX_LM_MULTIMODAL_MODELS` in `vllm_metal/v1/model_runner.py` — note that this uses `gemma4.Model.sanitize()` which drops vision/audio weights at load time.
- **mlx-vlm Gemma 4 — PyPI release pending**: Until a tagged mlx-vlm release that includes Gemma 4 is published on PyPI, install mlx-vlm from a local checkout: `pip install -e <path/to/mlx-vlm>`. The same applies to `mlx-lm >= 0.31.2` if no PyPI release is available yet (use `pip install git+https://github.com/ml-explore/mlx-lm@4469ad4`).

## Requirements

- macOS on Apple Silicon

## Installation
Using the install script, the following will be installed under the `~/.venv-vllm-metal` directory (the default).
- vllm-metal plugin
- vllm core
- Related libraries

If you run `source ~/.venv-vllm-metal/bin/activate`, the `vllm` CLI becomes available and you can access the vLLM right away.

For how to use the `vllm` CLI, please refer to the official vLLM guide.
https://docs.vllm.ai/en/latest/cli/

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

## Reinstallation and Update
If any issues occur, please use the following command to switch to the latest release version and check if the problem is resolved.
If the issue continues to occur in the latest release, please report the details of the issue.
(If you have installed it in a directory other than the default `~/.venv-vllm-metal`, substitute that path and run the command accordingly.)

```bash
rm -rf ~/.venv-vllm-metal && curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

## Uninstall
Please delete the directory that was installed by the installation script.
(If you have installed it in a directory other than the default `~/.venv-vllm-metal`, substitute that path and run the command accordingly.)
```bash
rm -rf ~/.venv-vllm-metal
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                          vLLM Core                          │
│          Engine, Scheduler, API Server, Tokenizers          │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   vllm_metal Plugin Layer                   │
│ ┌─────────────────┐ ┌────────────────┐ ┌──────────────────┐ │
│ │ MetalPlatform   │ │ MetalWorker    │ │ MetalModelRunner │ │
│ │ (Platform)      │ │ (Worker)       │ │ (ModelRunner)    │ │
│ └─────────────────┘ └────────────────┘ └──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Unified Compute Backend                   │
│ ┌───────────────────────────┐ ┌───────────────────────────┐ │
│ │   MLX Backend             │ │   PyTorch Backend         │ │
│ │   (Primary)               │ │   (Model Loading/Interop) │ │
│ │                           │ │                           │ │
│ │ • SDPA Attention          │ │ • HuggingFace Loading     │ │
│ │ • RMSNorm                 │ │ • Weight Conversion       │ │
│ │ • RoPE                    │ │ • Tensor Bridge           │ │
│ │ • Cache Ops               │ │                           │ │
│ └───────────────────────────┘ └───────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                       Metal GPU Layer                       │
│          Apple Silicon Unified Memory Architecture          │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

Environment variables for customization:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_METAL_MEMORY_FRACTION` | `auto` | `auto` allocates just enough memory plus a minimal KV cache, or `0.?` for fraction of memory |
| `VLLM_METAL_USE_MLX` | `1` | Use MLX for compute (1=yes, 0=no) |
| `VLLM_MLX_DEVICE` | `gpu` | MLX device (`gpu` or `cpu`) |
| `VLLM_METAL_BLOCK_SIZE` | `16` | KV cache block size |
| `VLLM_METAL_USE_PAGED_ATTENTION` | `0` | Enable experimental paged KV cache |
| `VLLM_METAL_DEBUG` | `0` | Enable debug logging |
| `VLLM_USE_MODELSCOPE` | `False` | Set True to change model registry to <https://www.modelscope.cn/> |
| `VLLM_METAL_MODELSCOPE_CACHE` | None | Specify the absolute path of the local model |
| `VLLM_METAL_PREFIX_CACHE` | (unset) | Set to enable prefix caching for shared prompt reuse |
| `VLLM_METAL_PREFIX_CACHE_FRACTION` | `0.05` | Fraction of MLX working set for prefix cache (0, 1] |

## Paged KV vs MLX KV memory settings

- MLX path (`VLLM_METAL_USE_PAGED_ATTENTION=0`): `VLLM_METAL_MEMORY_FRACTION` must be `auto`.
- Paged KV path (`VLLM_METAL_USE_PAGED_ATTENTION=1`): `VLLM_METAL_MEMORY_FRACTION` can be `auto` or a numeric fraction in `(0, 1]`.
- For paged KV with `VLLM_METAL_MEMORY_FRACTION=auto`, vllm-metal uses a default fraction of `0.9`.

`VLLM_METAL_MEMORY_FRACTION` | `VLLM_METAL_USE_PAGED_ATTENTION` | Valid? | Notes
-- | -- | -- | --
`auto` | `0` | Yes | MLX path (default)
`auto` | `1` | Yes | Paged KV path; defaults to 0.9 internally
`0.7` | `1` | Yes | Paged KV path with explicit memory budget
`0.7` | `0` | No | Explicit fraction without paged KV is invalid

## Acknowledgements

- The Metal paged attention kernels are currently adapted from [mistral.rs](https://github.com/EricLBuehler/mistral.rs) (MIT license), via [HuggingFace kernels-community](https://github.com/huggingface/kernels-community). We plan to develop custom kernels in the future.
