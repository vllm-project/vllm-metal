# Rust Frontend (experimental)

vllm-metal can be paired with [`vllm-frontend-rs`](https://github.com/Inferact/vllm-frontend-rs), an early-stage Rust drop-in for vLLM's frontend (HTTP server, tokenizer, chat templating, OpenAI protocol).

The Rust frontend talks to the Python engine over **ZMQ + MessagePack** — the same boundary upstream vLLM uses internally — so it is hardware-agnostic and works alongside vllm-metal unchanged. vllm-metal still runs the model and KV cache on Metal/MLX inside the Python engine subprocess; only the northbound serving layer is replaced.

First, install vllm-metal with the Rust frontend opt-in (see [Installation](installation.md)):

```bash
./install.sh --with-vllm-rs
```

Requires the Rust toolchain (cargo + rustup) on `PATH`. The upstream pins a nightly toolchain via its `rust-toolchain.toml`, which `rustup` auto-fetches on first build. The `vllm-rs` binary lands in `~/.cargo/bin/`.

## Architecture

```
  HTTP request (OpenAI API)
        │
        ▼
  vllm-rs (Rust)            ← frontend: tokenizer, chat templating, protocol
        │
        │ ZMQ + MessagePack
        ▼
  Python EngineCore         ← scheduler, sampling
        │
        ▼
  vllm-metal plugin         ← MLX paged attention, native Metal kernels
```

## Quick Start

Activate the venv so `vllm-rs serve` finds the Python interpreter that has vllm and vllm-metal installed, then launch the server:

```bash
source ~/.venv-vllm-metal/bin/activate
vllm-rs serve Qwen/Qwen3-0.6B
# OpenAI-compatible API now on http://127.0.0.1:8000
```

`vllm-rs serve` spawns a managed headless Python engine, performs the data-parallel handshake over ZMQ, and starts the Rust OpenAI server once the engine is ready. Inside the engine, vllm-metal activates as the platform plugin exactly as it would under `vllm serve`.

Verify with a chat completion:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 32,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

## Useful Flags

`vllm-rs serve --help` lists the full set. Common options:

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | HTTP bind host |
| `--port` | `8000` | HTTP bind port |
| `--python` | `python3` | Python interpreter used for the managed engine (env: `VLLM_RS_PYTHON`) |
| `--max-model-len` | from config | Override the model's max context length |
| `--reasoning-parser` | `auto` | Tool/reasoning parser selection (`auto`, `none`, or a specific parser) |
| `--tool-call-parser` | `auto` | Tool-call parser selection |
| `--headless` | off | Only launch the managed engine, do not start the Rust frontend |

## Two Integration Directions

There are two ways the Rust frontend and Python engine can be wired together:

| Direction | Command | Status |
|---|---|---|
| **Rust-driven** — Rust binary spawns headless Python engine | `vllm-rs serve <MODEL>` | **Works today** with bundled vllm 0.20.0 |
| **Python-driven** — Python `vllm` spawns Rust binary as subprocess | `VLLM_USE_RUST_FRONTEND=1 vllm serve <MODEL>` | Requires an upstream vLLM hook that has not landed yet |

Use the Rust-driven direction (`vllm-rs serve`) on the bundled vllm 0.20.0. The Python-driven direction will become available once the upstream hook merges.

## Limitations

- This is an early-stage project. APIs and CLI flags may change without notice.
- The first `cargo install` compiles ~70 crates and typically takes several minutes on Apple Silicon.
- The toolchain is pinned to a specific nightly; if the upstream bumps the pin, `rustup` will fetch a different nightly on the next reinstall.
