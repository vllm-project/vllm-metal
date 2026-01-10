# Distributed Inference with Ray

vLLM-Metal can be used with [Ray](https://ray.io/) for scaling across multiple Apple Silicon Macs. This enables tensor parallelism and data parallelism for larger models and higher throughput.

## Current Status

> ⚠️ **HIGHLY EXPERIMENTAL**: Ray integration on macOS with Metal is experimental and has significant limitations. Manual configuration is required and success is not guaranteed. Known issues include:
>
> 1. **gRPC communication issues** - Upstream Ray bugs causing connection problems on macOS
> 2. **IP address mismatch** - Ray and vLLM may use different node addresses, causing placement group scheduling to hang
> 3. **Resource allocation conflicts** - Custom resource keys (like "METAL") don't appear in Ray's accelerator IDs, causing KeyError in vLLM's Ray executor
> 4. **Upstream vLLM changes needed** - Proper Ray support for non-GPU platforms requires changes in vLLM's Ray executor to handle custom accelerator keys properly
>
> **Known Working Configurations**:
> - Single-node Ray cluster with explicit IP address configuration (see troubleshooting)
> - Simple model serving with minimal parallelism
> - Ray configured with custom "METAL" resource: `--resources='{"METAL": 1}'`
>
> **Not Yet Supported**:
> - Multi-node clusters across different machines
> - Advanced tensor/data parallelism configurations
> - Production deployments
> - Automatic resource detection and allocation
>
> The integration code is provided as-is for testing purposes. See troubleshooting section for manual configuration steps.
>
> **⚠️ EXPERIMENTAL FEATURE NOTICE**: This feature is marked as experimental and may change significantly in future releases. Use at your own risk.

## Installation

Install vLLM-Metal with Ray support:

```bash
# Install with Ray optional dependency
pip install vllm-metal[ray]

# Or install Ray separately
pip install ray>=2.0.0
```

## Single-Machine Usage

For single-machine parallelism (e.g., using Ray for process isolation):

```python
import ray
ray.init()

from vllm import LLM, SamplingParams

# Ray executor will be auto-detected when Ray is initialized
llm = LLM(
    model="mlx-community/Llama-3.2-1B-Instruct-4bit",
    max_model_len=4096,
)

# Or explicitly specify Ray executor
llm = LLM(
    model="mlx-community/Llama-3.2-1B-Instruct-4bit",
    distributed_executor_backend="ray",
    max_model_len=4096,
)

outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=100))
print(outputs[0].outputs[0].text)
```

## Multi-Mac Cluster Setup

To distribute inference across multiple Apple Silicon Macs:

### 1. Start Ray Head Node (Primary Mac)

```bash
# On the primary Mac
ray start --head --port=6379

# Note the address printed, e.g., ray://192.168.1.100:10001
```

### 2. Join Worker Nodes (Additional Macs)

```bash
# On each additional Mac
ray start --address='192.168.1.100:6379'
```

### 3. Run Distributed Inference

```python
import ray
ray.init(address="ray://192.168.1.100:10001")

from vllm import LLM, SamplingParams

# Use tensor parallelism across nodes
llm = LLM(
    model="mlx-community/Llama-3.2-70B-Instruct-4bit",
    tensor_parallel_size=2,  # Split across 2 nodes
    distributed_executor_backend="ray",
)

outputs = llm.generate(["Explain quantum computing"], SamplingParams(max_tokens=500))
```

## CLI Usage

```bash
# Start Ray cluster first
ray start --head

# Serve with Ray executor
vllm serve mlx-community/Llama-3.2-1B-Instruct-4bit \
    --distributed-executor-backend ray \
    --max-model-len 4096
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RAY_ADDRESS` | - | Ray cluster address (auto-detected if Ray is initialized) |

## How It Works

1. **Platform Detection**: MetalPlatform detects Ray when `ray.is_initialized()` returns True
2. **Executor Selection**: Automatically switches to Ray executor, or use `distributed_executor_backend="ray"`
3. **Resource Scheduling**: Uses `CPU` resources for Ray placement groups (Apple Silicon has unified memory)
4. **Process Isolation**: Uses `spawn` multiprocessing to avoid Metal/MLX state corruption

## Architecture with Ray

```
┌──────────────────────────────────────────────────────┐
│                     Ray Cluster                      │
│  ┌─────────────────────┐    ┌─────────────────────┐  │
│  │   Mac 1 (Head)      │    │   Mac 2 (Worker)    │  │
│  │  ┌───────────────┐  │    │  ┌───────────────┐  │  │
│  │  │ MetalWorker   │  │◄───┤  │ MetalWorker   │  │  │
│  │  │ (TP Rank 0)   │  │    │  │ (TP Rank 1)   │  │  │
│  │  └───────────────┘  │    │  └───────────────┘  │  │
│  │         │           │    │         │           │  │
│  │         ▼           │    │         ▼           │  │
│  │  ┌───────────────┐  │    │  ┌───────────────┐  │  │
│  │  │ MLX Backend   │  │    │  │ MLX Backend   │  │  │
│  │  │ (M4 Max GPU)  │  │    │  │ (M4 Max GPU)  │  │  │
│  │  └───────────────┘  │    │  └───────────────┘  │  │
│  └─────────────────────┘    └─────────────────────┘  │
│              │                        │              │
│              └────────────────────────┘              │
│                  Gloo Collective Ops                 │
└──────────────────────────────────────────────────────┘
```

## Troubleshooting

### Ray Connection Issues

```
Failed to connect to GCS at address 127.0.0.1:6379
```

This is a known Ray + macOS gRPC issue. Workarounds:
- Ensure Ray is fully stopped before restarting: `ray stop --force`
- Try setting: `RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1`
- Check firewall settings allow Ray ports (6379, 10001-10100)

### IP Address Mismatch

Ray and vLLM may use different node addresses, causing placement group scheduling to hang:

```
Error: No available node types can fulfill resource request {'CPU': 1.0, 'node:192.168.1.10': 0.001}
```

This occurs when Ray reports one IP address but vLLM requests resources on a different IP. To fix:

1. **Standardize IP addresses** - Explicitly set the Ray node IP to match what vLLM expects:
   ```bash
   ray stop -f
   ray start --head --node-ip-address=127.0.0.1 --port=6379 --disable-usage-stats
   VLLM_HOST_IP=127.0.0.1 vllm serve HuggingFaceTB/SmolLM2-135M --port 8000 --distributed-executor-backend ray
   ```

2. **Alternative: Use consistent network interface** - If you need to use a specific network interface:
   ```bash
   # Find your machine's IP address
   ipconfig getifaddr en0  # or en1, depending on your active interface

   # Use that IP consistently
   ray stop -f
   ray start --head --node-ip-address=192.168.1.10 --port=6379 --disable-usage-stats
   VLLM_HOST_IP=192.168.1.10 vllm serve HuggingFaceTB/SmolLM2-135M --port 8000 --distributed-executor-backend ray
   ```

3. **Check Ray node addresses**:
   ```bash
   python -c "import ray; ray.init(address='auto'); print([n['NodeManagerAddress'] for n in ray.nodes() if n.get('Alive')])"
   ```

4. **For multi-machine setups**, ensure all machines can reach each other on the specified IPs and that firewalls allow Ray traffic.

### Resource Allocation Conflicts

When using Ray with Metal, you may see errors like:
```
ValueError: Use the 'num_cpus' and 'num_gpus' keyword instead of 'CPU' and 'GPU' in 'resources' keyword
```
or
```
KeyError: 'METAL'
```

The first error occurred with the previous "CPU" resource key approach, which conflicted with Ray's special handling of CPU resources. The second error occurs because custom resources like "METAL" don't appear in Ray's accelerator IDs that vLLM's executor expects.

**Solution**: Configure Ray with the custom resource and be aware of the limitation:

```bash
# Start Ray with custom METAL resource
ray stop -f
ray start --head --node-ip-address=127.0.0.1 --port=6379 --disable-usage-stats --resources='{"METAL": 1}'
```

Note that this may still cause issues with vLLM's Ray executor expecting accelerator IDs. This requires upstream changes in vLLM to properly support non-GPU platforms.

### Metrics Agent Errors

```
Failed to establish connection to the metrics exporter agent
```

These are non-fatal warnings. Metrics won't be exported but inference will still work.

### Process Crashes (SIGSEGV)

If you see segmentation faults, ensure:
- vLLM-Metal is using `spawn` multiprocessing (this is now automatic)
- You're not mixing `fork` with MLX operations in the parent process

### Checking Cluster Status

```bash
# View Ray cluster status
ray status

# View connected nodes
python -c "import ray; ray.init(); print(ray.nodes())"
```

