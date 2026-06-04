# Distributed Inference with Ray

!!! warning
    This is a **contributor-facing** feature (scaffolding for upcoming multi-Mac support), not intended for general users — for normal single-Mac serving, use the default in-process executor.

vllm-metal can run under vLLM's **Ray distributed executor**, placing each Apple-Silicon worker as a Ray actor. This is the groundwork for multi-Mac serving; today the **single-node** path (one Mac, `--tensor-parallel-size 1`) is supported and validated. Multi-node tensor / pipeline parallelism is in progress (see [Limitations](#limitations)).

Apple GPUs are not a Ray-recognized accelerator type (unlike CUDA or TPU), so each node advertises a **custom Ray resource named `mlx`**, and vLLM's executor places one worker per `mlx` unit.

## Requirements

- `ray` installed alongside vllm-metal: `uv pip install ray` (or `pip install ray`).
- Each node started with one `mlx` resource (one Apple GPU per Mac).
- The Ray node IP must match the address vLLM resolves via `get_ip()` — do not mix loopback and LAN addresses.

## Quick start (single node)

```bash
# 1. Start a Ray head node advertising the Apple GPU as the "mlx" resource.
#    Pin the node IP to the address vLLM uses so placement-group binding matches.
IP=$(python -c "from vllm.utils.network_utils import get_ip; print(get_ip())")
ray start --head --node-ip-address="$IP" --resources='{"mlx": 1}'

# 2. Serve through the Ray executor (connects to the running cluster).
#    Plain `--distributed-executor-backend ray` uses the default Ray V2 executor.
RAY_ADDRESS=auto vllm serve Qwen/Qwen3-0.6B \
  --distributed-executor-backend ray \
  --tensor-parallel-size 1

# 3. Verify generation runs through Ray.
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
# → " Paris. The capital of Italy is Rome. The capital of Spain is Madrid."

# Tear down when done.
ray stop
```

If a node isn't advertising the `mlx` resource, the engine can't place workers: the Ray executor logs `No available node types can fulfill resource request {'mlx': 1.0, ...}` and hangs while creating the placement group. (A separate `current platform cpu does not support ray` error instead means the Metal plugin isn't active or `ray_device_key` is unset — not a missing resource.)

On a healthy boot, each worker logs `vllm_metal: patched Ray V2 worker get_node_and_gpu_ids on RayWorkerProc (Apple-GPU custom Ray resource)` — confirming the custom-resource override fired inside the Ray actor.

## How it works

- `MetalPlatform` sets `ray_device_key = "mlx"` and `device_control_env_var = "VLLM_METAL_VISIBLE_DEVICES"`, so vLLM's Ray executor takes its generic custom-resource placement path (the same one TPU uses) instead of the CUDA `num_gpus` path.
- A compatibility shim overrides the Ray worker's `get_node_and_gpu_ids` to read the assigned `mlx` resource — Ray never lists custom resources in `get_accelerator_ids()` — installed in each worker via a Ray `worker_process_setup_hook`.

## Limitations

- **Single node, `--tensor-parallel-size 1`** is what is validated today. Multi-node tensor / pipeline parallelism — where the cross-Mac collectives run through MLX (`mx.distributed`) — is not yet implemented.
- **Pipeline parallelism is rejected**: `--pipeline-parallel-size > 1` raises at startup until cross-stage activation hand-off lands.
- One `mlx` resource per Mac (a single Apple GPU per machine).
- On a multi-NIC host, set the Ray node IP and vLLM's `get_ip()` consistently (e.g. `ray start --node-ip-address=...`).
