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

To remove the installation:

```bash
rm -rf ~/.venv-vllm-metal
```

For a clean update, remove the environment and repeat the installation commands.
Model weights in the Hugging Face cache are kept.

`pip install vllm-metal` is not supported. For serving options, see the
[vLLM CLI guide](https://docs.vllm.ai/en/latest/cli/).
