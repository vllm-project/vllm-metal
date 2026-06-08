# Distributed Inference with Ray

!!! note
    For normal single-Mac serving you don't need any of this — use the default in-process executor. The Ray executor and pipeline parallelism below are for **multi-Mac** serving (running one model across several Macs). Single-node serving, the pipeline-parallel forward, and a two-Mac end-to-end run (Qwen3-0.6B over Thunderbolt) are validated; multi-Mac serving is still new — **verify it for your own models and setup before relying on it.**

vllm-metal can run under vLLM's **Ray distributed executor**, placing each Apple-Silicon worker as a Ray actor. This is the groundwork for multi-Mac serving; today the **single-node** path (one Mac, `--tensor-parallel-size 1`) is supported and validated. **Pipeline parallelism** — splitting a model across stages — is numerically validated on a single node (see [Pipeline parallelism](#pipeline-parallelism)) and has served a model end-to-end across two Macs over Thunderbolt (see [Limitations](#limitations)).

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

## Quick start (two Macs over Thunderbolt)

Run one model split across two Macs with [pipeline parallelism](#pipeline-parallelism): **stage 0** (first layers) on Mac A, **stage 1** (last layers + sampling) on Mac B. Ray is the control plane; the cross-stage activations travel over the **MLX ring** on a direct **Thunderbolt cable**. That high-bandwidth, low-latency link is what makes PP across machines worthwhile — Wi-Fi / Ethernet is too slow to serve over, so Thunderbolt is the supported transport.

!!! note
    Multi-Mac serving is new — validated end-to-end on Qwen3-0.6B; start with that small model to check the plumbing, then scale up. Both Macs must have the model cached, the Thunderbolt bridge reachable, and each Mac's firewall must allow the MLX ring ports (`32323`/`32324`).

!!! note
    A multi-node Ray cluster on **macOS** needs two extra env vars on every node, both exported in the commands below: `RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1` (macOS clustering is gated behind it), and — only if the two Macs aren't on the identical Python build — `RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=minor` (Ray otherwise refuses to join nodes whose Python differs even at the *patch* level; the same minor version is wire-compatible).

Connect the two Macs with a Thunderbolt / USB4 cable — macOS auto-creates a "Thunderbolt Bridge" interface — and give it a static IP on each (same subnet, different last octet):

```bash
# Mac A
sudo networksetup -setmanual "Thunderbolt Bridge" 10.0.0.1 255.255.255.0
# Mac B
sudo networksetup -setmanual "Thunderbolt Bridge" 10.0.0.2 255.255.255.0
```

From Mac A, confirm the cable is up: `ping -c3 10.0.0.2`.

gloo (vLLM's control plane) advertises whatever each node's **hostname** resolves to — and a macOS `.local` hostname resolves to loopback, which can't be reached from the other Mac. Map each node's hostname to its bridge IP, and leave `GLOO_SOCKET_IFNAME` unset:

```bash
# Mac A
sudo scutil --set HostName maca && echo "10.0.0.1  maca" | sudo tee -a /etc/hosts
# Mac B
sudo scutil --set HostName macb && echo "10.0.0.2  macb" | sudo tee -a /etc/hosts
# verify on each — must print the bridge IP, not 127.0.0.1:
python -c "import socket; print(socket.gethostbyname(socket.gethostname()))"
```

**Mac A** — start the Ray head, pinned to its Thunderbolt IP:

```bash
source .venv-vllm-metal/bin/activate
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1            # multi-node Ray on macOS
export RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=minor   # only if the Macs' Python patch versions differ
VLLM_HOST_IP=10.0.0.1 ray start --head \
  --node-ip-address=10.0.0.1 --resources='{"mlx": 1}'
```

**Mac B** — join over Mac A's Thunderbolt IP:

```bash
source .venv-vllm-metal/bin/activate
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
export RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=minor   # only if the Macs' Python patch versions differ
VLLM_HOST_IP=10.0.0.2 ray start --address=10.0.0.1:6379 \
  --node-ip-address=10.0.0.2 --resources='{"mlx": 1}'
```

**Mac A** — serve across both stages:

```bash
RAY_ADDRESS=auto VLLM_HOST_IP=10.0.0.1 VLLM_METAL_USE_PAGED_ATTENTION=1 \
  vllm serve Qwen/Qwen3-0.6B \
    --distributed-executor-backend ray \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 1 \
    --no-async-scheduling
```

`--no-async-scheduling` is required for PP — the first stage has no sampler and rebuilds the token stream from the scheduler, which async scheduling would leave empty (the engine fails loud if you omit it).

On a healthy boot each worker logs its stage — `Pipeline stage 0/2 (is_first=True, is_last=False)` and `Pipeline stage 1/2 (is_first=False, is_last=True)` — and the MLX ring bootstrap lists both Macs' Thunderbolt IPs.

**Query** (from Mac A):

```bash
curl -s http://10.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
```

Tear down with `ray stop` on **both** Macs, then revert the bridge: `sudo networksetup -setdhcp "Thunderbolt Bridge"`. Once the small model works end to end, serve a model too large for a single Mac — that is the point of pipeline parallelism.

`VLLM_HOST_IP` is the load-bearing piece: it makes vLLM's `get_ip()` return the Thunderbolt address, so the cross-stage hand-off forms over the cable.

## How it works

- `MetalPlatform` sets `ray_device_key = "mlx"` and `device_control_env_var = "VLLM_METAL_VISIBLE_DEVICES"`, so vLLM's Ray executor takes its generic custom-resource placement path (the same one TPU uses) instead of the CUDA `num_gpus` path.
- A compatibility shim overrides the Ray worker's `get_node_and_gpu_ids` to read the assigned `mlx` resource — Ray never lists custom resources in `get_accelerator_ids()` — installed in each worker via a Ray `worker_process_setup_hook`.

## Pipeline parallelism

Pipeline parallelism (PP) splits the model's transformer layers into contiguous
stages — one worker process per stage — and pipes the hidden state from one stage
to the next. It is the natural fit for multi-Mac serving: unlike tensor
parallelism (an all-reduce every layer), PP sends a single activation per stage
boundary, so it tolerates a Thunderbolt / Ethernet link between machines.

**Design.** The mlx_lm model files are untouched. vLLM's executor (Ray, or the
`mp` executor for a single node) owns the *control plane* — spawning one ranked
worker per stage. The *data plane* — the cross-stage activation hand-off — runs
over MLX's own `mx.distributed` **ring** backend (point-to-point `send` / `recv`),
not Ray. Rank 0 is the first stage (it embeds the tokens); rank `N-1` is the last
(final norm, head, sampling). Each stage owns a contiguous layer slice and only
the last stage produces logits. Tensor parallelism must be 1, so the global rank
equals the pipeline-stage index.

**Numerical validation (single node).** `tools/pp_parity_check.py` runs the PP
forward across `N` ring processes and compares the logits against a single-process
reference. Because every stage runs identical ops on its slice and the hidden
state crosses the wire unchanged, the result is **bit-exact**:

```bash
source .venv-vllm-metal/bin/activate

# 1. Reference logits (single process; writes /tmp/pp_ref_<model>.npy).
python tools/pp_parity_check.py Qwen/Qwen3-0.6B

# 2. Two-stage pipeline over the MLX ring, compared against the reference.
mlx.launch -n 2 --backend ring tools/pp_parity_check.py Qwen/Qwen3-0.6B
# → PARITY PASS max_abs_diff=0.000e+00   (validated on Qwen3-0.6B / 1.7B / 4B)
```

## Limitations

- **Pipeline parallelism is new — validated end-to-end on two Macs (Qwen3-0.6B over Thunderbolt), but exercised on a narrow set of models and setups.** The PP forward is numerically validated bit-exact via `tools/pp_parity_check.py`, and the full engine path (executor → ranked workers → MLX ring across real per-node IPs → per-stage forward → last-rank sampling) has served a complete two-Mac generation. Verify output for your own models and confirm the ring ports are reachable between Macs.
- **Single-node co-located stages contend for memory.** With `N` stages on one Mac, each worker reserves a full-machine KV budget and the Metal wired limit ignores `VLLM_METAL_MEMORY_FRACTION` — lower `VLLM_METAL_MEMORY_FRACTION` when stacking stages on one machine. Separate Macs each own their RAM, so this does not affect true multi-node runs.
- **PP combined with tensor parallelism is rejected**: only `--pipeline-parallel-size > 1` with `--tensor-parallel-size 1` is supported.
- **Tensor parallelism (`--tensor-parallel-size > 1`) is not implemented** — the per-layer MLX collective is not wired yet.
- One `mlx` resource per Mac (a single Apple GPU per machine).
- On a multi-NIC host, set the Ray node IP and vLLM's `get_ip()` consistently (e.g. `ray start --node-ip-address=...`).
