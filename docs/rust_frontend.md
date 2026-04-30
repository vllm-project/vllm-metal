# Rust Frontend (experimental)

vllm-metal supports the optional [`vllm-frontend-rs`](https://github.com/Inferact/vllm-frontend-rs) Rust frontend as a drop-in replacement for vLLM's Python serving layer. The Rust frontend is hardware-agnostic; vllm-metal continues to run the engine on Metal/MLX inside the spawned Python subprocess.

For architecture, flag reference, and usage, see the [upstream README](https://github.com/Inferact/vllm-frontend-rs#readme).

## Install

```bash
./install.sh --with-vllm-rs
```

See [Installation](installation.md) for prerequisites (Rust toolchain).

## Run

Activate the venv first so the spawned headless Python engine inherits vllm-metal:

```bash
source ~/.venv-vllm-metal/bin/activate
vllm-rs serve Qwen/Qwen3-0.6B
```

## Caveat: integration direction

Use `vllm-rs serve` (Rust binary spawns Python engine) on bundled vllm 0.20.0. The reverse direction `VLLM_USE_RUST_FRONTEND=1 vllm serve` requires an upstream vLLM hook that has not landed yet.
