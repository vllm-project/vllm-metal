# Installation

## Requirements

- macOS on Apple Silicon
- Native arm64 Python 3.12. Rosetta/x86_64 Python is not supported.
- Xcode Command Line Tools (`xcode-select --install`). The installer compiles vLLM core from source via `clang++`.

> **Metal kernels are prebuilt.** Release wheels ship the native paged-attention
> extension (`_paged_ops*.so`) and its precompiled Metal shader libraries
> (`*.metallib`), so `pip install` requires no compiler or Metal toolchain to
> *run* the kernels — nothing is compiled on first request. The Command Line
> Tools above are still needed because the installer builds vLLM core from
> source. Building the Metal kernels yourself is only for contributors editing
> them; see [Contributing](CONTRIBUTING.md).

`uv` is bootstrapped automatically.

Verify the Python architecture before installing:

```bash
python3 -c "import platform; print(platform.machine())"
file "$(which python3)"
```

The first command should print `arm64`. If it prints `x86_64`, switch to a native arm64 Python and remove `~/.venv-vllm-metal` before reinstalling.

## Install

Using the install script, the following will be installed under the `~/.venv-vllm-metal` directory (the default).
- vllm-metal plugin
- vllm core
- Related libraries

If you run `source ~/.venv-vllm-metal/bin/activate`, the `vllm` CLI becomes available and you can access the vLLM right away.

For how to use the `vllm` CLI, please refer to the [official vLLM guide](https://docs.vllm.ai/en/latest/cli/).

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

## Reinstallation and Update

If any issues occur, please use the following command to switch to the latest release version and check if the problem is resolved.
If the issue continues to occur in the latest release, please report the details of the issue.
(If you have installed it in a directory other than the default `~/.venv-vllm-metal`, substitute that path and run the command accordingly.)

```bash
rm -rf ~/.venv-vllm-metal && curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

## Uninstall

Please delete the directory that was installed by the installation script.
(If you have installed it in a directory other than the default `~/.venv-vllm-metal`, substitute that path and run the command accordingly.)

```bash
rm -rf ~/.venv-vllm-metal
```

## Building Documentation

```bash
uv pip install -r docs/requirements-docs.txt
mkdocs serve
```
