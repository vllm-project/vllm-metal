# vLLM Metal Plugin

> **High-performance LLM inference on Apple Silicon using MLX and vLLM**

vLLM Metal is a plugin that enables vLLM to run on Apple Silicon Macs using MLX as the primary compute backend. It unifies MLX and PyTorch under a single lowering path.

## Features

- **MLX-accelerated inference**: faster than PyTorch MPS on Apple Silicon
- **Unified memory**: True zero-copy operations leveraging Apple Silicon's unified memory architecture
- **vLLM compatibility**: Full integration with vLLM's engine, scheduler, and OpenAI-compatible API
- **Paged attention**: Efficient KV cache management for long sequences
- **GQA support**: Grouped-Query Attention for efficient inference

## Requirements

- macOS on Apple Silicon

## Installation

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
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

## Vision-Language Models (VLM)

vLLM Metal supports vision-language models like Qwen2-VL using [mlx-vlm](https://github.com/Blaizzy/mlx-vlm).

### Supported VLM Models

- Qwen2-VL (e.g., `mlx-community/Qwen2-VL-2B-Instruct-4bit`)
- Qwen2.5-VL
- LLaVA
- Pixtral
- Idefics

### Usage with Vision Models

Start the server with a VLM:

```bash
vllm-metal --model mlx-community/Qwen2-VL-2B-Instruct-4bit
```

Send requests with images using the OpenAI-compatible API:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
      }
    ],
    "max_tokens": 100
  }'
```

### Python API

```python
from vllm_metal.model_runner import MetalModelRunner

# Create a minimal config for VLM
class Config:
    class ModelConfig:
        model = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    model_config = ModelConfig()

runner = MetalModelRunner(Config())
runner.load_model()

# Generate with an image
output = runner.generate(
    prompt="Describe this image",
    images=["https://example.com/image.jpg"],
    max_tokens=100,
)
print(output)
```

## Configuration

Environment variables for customization:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_METAL_MEMORY_FRACTION` | `auto` | `auto` allocates just enough memory plus a minimal KV cache, or `0.?` for fraction of memory |
| `VLLM_METAL_USE_MLX` | `1` | Use MLX for compute (1=yes, 0=no) |
| `VLLM_MLX_DEVICE` | `gpu` | MLX device (`gpu` or `cpu`) |
| `VLLM_METAL_BLOCK_SIZE` | `16` | KV cache block size |
| `VLLM_METAL_DEBUG` | `0` | Enable debug logging |

