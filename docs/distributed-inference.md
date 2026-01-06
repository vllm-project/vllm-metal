# Distributed Inference with vLLM + Metal

This guide covers setting up distributed LLM inference across multiple Apple Silicon Macs using vLLM-Metal.

**Note: Ray distributed execution is not currently supported on Metal platforms.** The Metal backend uses a uniprocessor execution model that runs on a single device. This means true distributed inference across multiple nodes using Ray is not available with vLLM-Metal.

## Current Limitations

- **Single Device Only**: vLLM-Metal is designed to run on a single Apple Silicon device
- **No Ray Support**: The `distributed_executor_backend="ray"` option is not supported
- **Tensor Parallelism**: Limited to single-device tensor parallelism if multiple GPUs are available on the same machine

## Alternative Approaches

### 1. Single-Device Tensor Parallelism

You can still leverage tensor parallelism on a single Apple Silicon device with multiple GPU cores:

```python
from vllm import LLM, SamplingParams

# Create vLLM instance (tensor parallelism limited to single device)
llm = LLM(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    tensor_parallel_size=1,  # Limited to 1 on Metal
)

# Generate
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Hello, how are you?"], sampling_params)
```

### 2. Multi-Process Deployment

For higher throughput, you can run multiple vLLM instances across different ports:

```bash
# Terminal 1 - Instance 1
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='mlx-community/Llama-3.2-3B-Instruct-4bit')
# Add your serving logic here
"

# Terminal 2 - Instance 2
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='mlx-community/Llama-3.2-3B-Instruct-4bit')
# Add your serving logic here
"
```

### 3. Model Sharding on Single Device

You can optimize memory usage on a single device:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    # Optimize memory usage
    gpu_memory_utilization=0.8,
    max_num_batched_tokens=4096,
    max_num_seqs=256,
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Hello, how are you?"], sampling_params)
```

## Future Support

Distributed inference with Ray may be added in future versions of vLLM-Metal. Currently, the Metal backend is focused on optimizing single-device performance.

