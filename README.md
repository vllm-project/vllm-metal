# vLLM Metal Plugin

> **High-performance LLM inference on Apple Silicon using MLX and vLLM**

vLLM Metal is a plugin that enables vLLM to run on Apple Silicon Macs using MLX as the primary compute backend. It unifies MLX and PyTorch under a single lowering path.

## Features

- **MLX-accelerated inference**: faster than PyTorch MPS on Apple Silicon
- **Unified memory**: True zero-copy operations leveraging Apple Silicon's unified memory architecture
- **vLLM compatibility**: Full integration with vLLM's engine, scheduler, and OpenAI-compatible API
- **Paged attention** *(default)*: Efficient KV cache management with native Metal kernels — enabled by default. Set `VLLM_METAL_USE_PAGED_ATTENTION=0` to fall back to the legacy MLX-managed KV cache path.
- **GQA support**: Grouped-Query Attention for efficient inference

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
| `VLLM_METAL_USE_PAGED_ATTENTION` | `1` | Paged KV cache with native Metal kernels (set `0` to use legacy MLX KV cache) |
| `VLLM_METAL_DEBUG` | `0` | Enable debug logging |
| `VLLM_USE_MODELSCOPE` | `False` | Set True to change model registry to <https://www.modelscope.cn/> |
| `VLLM_METAL_MODELSCOPE_CACHE` | None | Specify the absolute path of the local model |
| `VLLM_METAL_PREFIX_CACHE` | (unset) | Set to enable prefix caching for shared prompt reuse |
| `VLLM_METAL_PREFIX_CACHE_FRACTION` | `0.05` | Fraction of MLX working set for prefix cache (0, 1] |

## Paged KV vs MLX KV memory settings

- Paged KV path (default, `VLLM_METAL_USE_PAGED_ATTENTION=1`): `VLLM_METAL_MEMORY_FRACTION` can be `auto` or a numeric fraction in `(0, 1]`.
- For paged KV with `VLLM_METAL_MEMORY_FRACTION=auto`, vllm-metal uses a default fraction of `0.9`.
- Legacy MLX path (`VLLM_METAL_USE_PAGED_ATTENTION=0`): `VLLM_METAL_MEMORY_FRACTION` must be `auto`.

`VLLM_METAL_MEMORY_FRACTION` | `VLLM_METAL_USE_PAGED_ATTENTION` | Valid? | Notes
-- | -- | -- | --
`auto` | `1` | Yes | Paged KV path (default); uses 0.9 internally
`0.7` | `1` | Yes | Paged KV path with explicit memory budget
`auto` | `0` | Yes | Legacy MLX path
`0.7` | `0` | No | Explicit fraction without paged KV is invalid

## Acknowledgements

- The Metal paged attention kernels are currently adapted from [mistral.rs](https://github.com/EricLBuehler/mistral.rs) (MIT license), via [HuggingFace kernels-community](https://github.com/huggingface/kernels-community). We plan to develop custom kernels in the future.
