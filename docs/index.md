# vllm-metal

**High-performance LLM inference on Apple Silicon using MLX and vLLM**

vLLM Metal is a plugin that enables vLLM to run on Apple Silicon Macs using MLX as the primary compute backend. It unifies MLX and PyTorch under a single lowering path.

## Features

- **MLX-accelerated inference**: faster than PyTorch MPS on Apple Silicon
- **Unified memory**: True zero-copy operations leveraging Apple Silicon's unified memory architecture
- **vLLM compatibility**: Full integration with vLLM's engine, scheduler, and OpenAI-compatible API
- **Paged attention** *(experimental)*: Efficient KV cache management for long sequences
- **GQA support**: Grouped-Query Attention for efficient inference

Check the sidebar for guides on installation, configuration, and supported features.
