# Installation

Requires Apple Silicon and macOS 15 or later. Both methods install Python 3.12,
vLLM core, and prebuilt vllm-metal wheels. No compiler is needed.
For an editable checkout, see [Contributing](CONTRIBUTING.md#development-setup).

## Homebrew

Install the stable release:

```bash
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal
```

Run `vllm serve <model>` directly.
Upgrade with `brew update && brew upgrade vllm-metal`; remove with `brew uninstall vllm-metal`.

## Install script

Install the latest development build and activate its environment:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
source ~/.venv-vllm-metal/bin/activate
```

The script installs `uv` if needed and uses `~/.venv-vllm-metal` for the Python
environment. Activate it in each new shell before running `vllm`.
For a stable release, replace `bash` with `bash -s -- --stable` in the command above.
To choose a separate environment, pass `--venv /absolute/path/to/venv` after
`bash -s --` and activate that directory's `bin/activate` instead.

To remove the installation:

```bash
rm -rf ~/.venv-vllm-metal
```

For a clean update, remove the environment and repeat the installation commands.
Model weights in the Hugging Face cache are kept.

`pip install vllm-metal` is not supported. For serving options, see the
[vLLM CLI guide](https://docs.vllm.ai/en/latest/cli/).

## Run your first request

After installing, start a server with the small `Qwen/Qwen3-0.6B` model:

```bash
vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization 0.3 --max-model-len 2048
```

If you used the install script, activate its environment first as shown above.
The first run downloads the model from Hugging Face unless it is already cached.
These settings limit the context length and use a smaller memory budget than the
default; available memory still needs to accommodate the model and its cache.
See [KV cache memory settings](configuration.md#kv-cache-memory-settings) for details.

Keep this terminal open and wait for `Application startup complete.` before
sending a request. In another terminal, run:

```bash
curl --fail --silent --show-error \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {
        "role": "user",
        "content": "Explain what an inference server does in one sentence."
      }
    ],
    "max_tokens": 64,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

`enable_thinking: false` disables Qwen3's thinking mode for this short example.
A successful request returns JSON with the generated text in
`choices[0].message.content` and token counts in `usage`. The wording may vary.
The second terminal only needs `curl`; it does not need the Python environment.

Press **Ctrl+C** in the server terminal when you are finished.
