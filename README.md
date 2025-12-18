# vLLM Metal Plugin

> **High-performance LLM inference on Apple Silicon using MLX and vLLM**

vLLM Metal is a hardware plugin that enables vLLM to run on Apple Silicon Macs (M1/M2/M3/M4/M5) using MLX as the primary compute backend. It unifies MLX and PyTorch under a single lowering path, similar to how [tpu-inference](https://github.com/vllm-project/tpu-inference) unifies JAX and PyTorch for TPUs.

## Features

- **MLX-accelerated inference**: 10-25x faster than PyTorch MPS on Apple Silicon
- **Unified memory**: True zero-copy operations leveraging Apple Silicon's unified memory architecture
- **vLLM compatibility**: Full integration with vLLM's engine, scheduler, and OpenAI-compatible API
- **Paged attention**: Efficient KV cache management for long sequences
- **GQA support**: Grouped-Query Attention for efficient inference

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4/M5)
- Python 3.11+
- MLX 0.20.0+
- vLLM 0.12.0+

## Installation

### Quick Install

```bash
./install.sh
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install MLX
pip install mlx mlx-lm

# Install vLLM (without CUDA deps)
pip install --no-deps vllm

# Install compatible dependencies
pip install transformers accelerate safetensors numpy psutil pydantic \
    cbor2 msgspec cloudpickle prometheus-client fastapi uvicorn uvloop \
    pillow tiktoken aiohttp openai einops tokenizers cachetools

# Install vLLM Metal
pip install -e .
```

## Usage

### Serve a Model

```bash
vllm serve Qwen/Qwen3-0.6B
```

### Python API

```python
from vllm import LLM, SamplingParams

# vLLM automatically detects and uses the Metal backend on Apple Silicon
llm = LLM(model="Qwen/Qwen3-0.6B")

prompts = ["Hello, my name is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

### OpenAI-Compatible API

```bash
# Start the server
vllm serve Qwen/Qwen3-0.6B --port 8000

# In another terminal
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM Core (Unchanged)                    │
│         Engine, Scheduler, API Server, Tokenizers           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 vllm_metal Plugin Layer                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ MetalPlatform│  │ MetalWorker │  │ MetalModelRunner    │ │
│  │ (Platform)   │  │ (Worker)    │  │ (ModelRunner)       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Unified Compute Backend                        │
│  ┌──────────────────────┐  ┌──────────────────────────────┐│
│  │   MLX Backend        │  │   PyTorch Backend            ││
│  │   (Primary)          │  │   (Model Loading/Interop)    ││
│  │                      │  │                              ││
│  │ • SDPA Attention     │  │ • HuggingFace Loading        ││
│  │ • RMSNorm            │  │ • Weight Conversion          ││
│  │ • RoPE               │  │ • Tensor Bridge              ││
│  │ • Cache Ops          │  │                              ││
│  └──────────────────────┘  └──────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Metal GPU Layer                          │
│         Apple Silicon Unified Memory Architecture           │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

Environment variables for customization:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_METAL_MEMORY_FRACTION` | `0.9` | Fraction of memory to use |
| `VLLM_METAL_USE_MLX` | `1` | Use MLX for compute (1=yes, 0=no) |
| `VLLM_MLX_DEVICE` | `gpu` | MLX device (`gpu` or `cpu`) |
| `VLLM_METAL_BLOCK_SIZE` | `16` | KV cache block size |
| `VLLM_METAL_DEBUG` | `0` | Enable debug logging |

## Supported Models

Any model supported by vLLM that uses standard transformer architectures:

- **Llama family**: Llama 2, Llama 3, Code Llama
- **Qwen family**: Qwen, Qwen2, Qwen2.5
- **Mistral family**: Mistral, Mixtral
- **Phi family**: Phi-2, Phi-3
- **And many more...**

## Limitations

- **No tensor parallelism**: Single GPU only (Apple Silicon limitation)
- **No CUDA graphs**: Metal doesn't support graph capture
- **No FP8**: FP8 quantization not supported
- **Memory bound**: Limited by unified memory (8GB-192GB depending on chip)

## Development

```bash
# Install dev dependencies
pip install -e '.[dev]'

# Run tests
pytest tests/ -v

# Run linters
ruff check .
ruff format .
mypy vllm_metal
```

## Performance

Expected performance on Apple Silicon (varies by model and chip):

| Chip | Memory | Throughput (7B model) |
|------|--------|----------------------|
| M1 Pro | 16GB | ~30 tok/s |
| M1 Max | 32GB | ~50 tok/s |
| M2 Ultra | 128GB | ~100 tok/s |
| M3 Max | 64GB | ~80 tok/s |
| M4 Max | 128GB | ~120 tok/s |

*Note: Actual performance depends on model, sequence length, and batch size.*

## Contributing

Contributions are welcome! Please see the [contributing guidelines](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE) for details.

## Acknowledgements

- [vLLM](https://github.com/vllm-project/vllm) - The high-throughput LLM serving engine
- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework for Apple Silicon
- [tpu-inference](https://github.com/vllm-project/tpu-inference) - Inspiration for the plugin architecture
