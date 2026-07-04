# Rust Frontend (experimental)

vllm-metal supports the optional `vllm-rs` Rust frontend as a drop-in replacement for vLLM's Python serving layer. The Rust frontend is hardware-agnostic; vllm-metal continues to run the engine on Metal/MLX inside the spawned Python subprocess.

As of vLLM 0.24.0, the Rust frontend source is vendored in the main vLLM repository under `rust/`. The former `Inferact/vllm-frontend-rs` repository is kept only as a historical archive.

For architecture, flag reference, and usage, see the [upstream README](https://github.com/vllm-project/vllm/tree/main/rust#readme).

## Install

```bash
./install.sh --with-vllm-rs
```

See [Installation](installation.md) for prerequisites (Rust toolchain).

## Run

Activate the venv first so the spawned headless Python engine inherits vllm-metal:

```bash
source ~/.venv-vllm-metal/bin/activate
VLLM_USE_RUST_FRONTEND=1 \
  VLLM_RUST_FRONTEND_PATH="$HOME/.cargo/bin/vllm-rs" \
  vllm serve Qwen/Qwen3-0.6B
```

## Standalone frontend

The `vllm-rs` binary is also installed to `~/.cargo/bin`. It can be used directly for advanced frontend-only or externally managed engine flows documented upstream.
