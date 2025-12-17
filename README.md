# vllm-metal

This plugin enables vLLM to run on Apple Silicon Macs using Metal Performance Shaders (MPS) for GPU acceleration.

## Installation

### Quick Install

Install the latest release with a single command:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/main/install.sh | bash
```

### From Source

```bash
git clone https://github.com/vllm-project/vllm-metal.git
cd vllm-metal
uv pip install -e .
```

## Features

- **Native Apple Silicon Support**: Run LLMs on Apple Silicon Macs
- **MPS Acceleration**: Leverages PyTorch's MPS backend for GPU operations
- **Paged Attention**: Full support for vLLM's paged attention mechanism
- **Memory Efficient**: Optimized for unified memory architecture
- **Drop-in Replacement**: Works with existing vLLM APIs

## Requirements

- macOS 12.3 or later
- Apple Silicon Mac
- Python 3.11 or later
- PyTorch 2.1.0 or later with MPS support
- vLLM 0.12.0 or later

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_METAL_DEVICE_ID` | `0` | MPS device ID |
| `VLLM_METAL_MEMORY_FRACTION` | `0.9` | Fraction of memory to use |
| `VLLM_METAL_ATTENTION_BACKEND` | `mps` | Attention backend (`mps` or `eager`) |
| `VLLM_METAL_EAGER_MODE` | `1` | Use eager mode (disable graph compilation) |
| `VLLM_METAL_MAX_BATCH_SIZE` | `256` | Maximum batch size |
| `VLLM_METAL_KV_CACHE_DTYPE` | `None` | KV cache dtype (default: model dtype) |
| `VLLM_METAL_ENABLE_PROFILING` | `0` | Enable profiling |

### Example Configuration

```bash
# Use 80% of available memory
export VLLM_METAL_MEMORY_FRACTION=0.8

# Enable profiling
export VLLM_METAL_ENABLE_PROFILING=1

# Run vLLM
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --dtype float16
```

## Limitations

- **Single GPU Only**: MPS does not support multi-GPU configurations
- **No Distributed Inference**: Tensor and pipeline parallelism not supported
- **Limited Quantization**: Some quantization methods (FP8) not available
- **Memory Sharing**: GPU memory is shared with system memory

## Performance Tips

1. **Use Float16**: Metal works best with `dtype=float16`
2. **Adjust Memory Fraction**: If you encounter OOM errors, reduce `VLLM_METAL_MEMORY_FRACTION`
3. **Batch Size**: Larger batch sizes can improve throughput
4. **Model Size**: Unified memory allows larger models than discrete GPU memory

## Troubleshooting

### MPS Not Available

```
RuntimeError: Metal/MPS backend not available
```

Ensure you're running on Apple Silicon and have macOS 12.3+ installed.

### Out of Memory

```
RuntimeError: MPS backend out of memory
```

Try reducing `VLLM_METAL_MEMORY_FRACTION` or using a smaller model.

### Slow Performance

- Ensure you're using `dtype=float16`
- Check that MPS is being used (not CPU fallback)
- Consider enabling eager mode if graph compilation is slow

## Development

### Running CI

```bash
scripts/ci.sh
```

